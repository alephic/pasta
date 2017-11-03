import json
import random
import signal
import os
import sys

from allennlp.data.fields import TextField
from allennlp.data import Dataset, Instance, Vocabulary
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer, CharacterTokenizer
from allennlp.data.tokenizers.word_splitter import LettersDigitsWordSplitter

import torch

import numpy as np

from tqdm import tqdm

from .model import LanguageModel

def ensure_path(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

OPTIM_CLASSES = {
    'sgd': torch.optim.SGD,
    'adam': torch.optim.Adam,
    'adagrad': torch.optim.Adagrad,
    'adadelta': torch.optim.Adadelta,
    'adamax': torch.optim.Adamax
}

MODEL_SAVE_DIR = 'trained_lm'

def load_dataset(json_filename, word_level=True):
    with open(json_filename) as f:
        text_list = json.load(f)
    if word_level:
        tokenizer = WordTokenizer(word_splitter=LettersDigitsWordSplitter() ,start_tokens=['@@SOS@@'], end_tokens=['@@EOS@@'])
    else:
        tokenizer = CharacterTokenizer(start_tokens=['@@SOS@@'], end_tokens=['@@EOS@@'])
    indexers = {'tokens': SingleIdTokenIndexer()}
    dataset = Dataset(list(tqdm((
        Instance({
            'text': TextField(
                tokens=tokenizer.tokenize(text),
                token_indexers=indexers
            )
        }) for text in text_list
    ), total=len(text_list))))
    return dataset

def partition_dataset(dataset: Dataset, ratio: float):
    split_index = int(len(dataset.instances)*ratio)
    return Dataset(dataset.instances[:split_index]), Dataset(dataset.instances[split_index:])

def infinite_shuffled_generator(dataset):
    order = list(range(len(dataset)))
    while True:
        random.shuffle(order)
        for i in order:
            yield dataset.instances[i]

def single_pass_generator(dataset):
    return iter(dataset.instances)

def lm_batch_generator(dataset_gen, vocab, batch_size, max_instance_length):
    indexers = {'tokens': SingleIdTokenIndexer()}
    source_instances = [next(dataset_gen) for i in range(batch_size)]
    source_offsets = [0 for i in range(batch_size)]
    active_slots = set(range(batch_size))
    padding_index = vocab.get_token_index(vocab._padding_token)
    while len(active_slots) > 0:
        rows = []
        for i in range(batch_size):
            if i in active_slots:
                source_instance = source_instances[i]
                source_offset = source_offsets[i]
                row = list(map(vocab.get_token_index, source_instance.fields['text'].tokens[source_offset:source_offset + max_instance_length]))
                row.extend([padding_index] * (max_instance_length - len(row)))
                rows.append(row)
                new_offset = source_offset + max_instance_length
                if new_offset >= len(source_instance.fields['text'].tokens):
                    source_instances[i] = next(dataset_gen, None)
                    source_offsets[i] = 0
                    if source_instances[i] is None:
                        active_slots.remove(i)
            else:
                rows.append([padding_index] * max_instance_length)
        yield np.array(rows)

def evaluate_metrics(model, dataset, metrics, batch_size, max_instance_length):
    model.eval()
    gen = single_pass_generator(dataset)
    total_metrics = {metric: 0 for metric in metrics}
    state = None
    for batch in lm_batch_generator(gen, model.vocab, batch_size, max_instance_length):
        curr_batch_size = batch.shape[0]
        batch_out = model(batch, init_state=state, teacher_forcing=1.0)
        state = batch_out['final_state']
        for metric in metrics:
            batch_metric = batch_out[metric]
            if isinstance(batch_metric, torch.autograd.Variable):
                total_metrics[metric] += batch_metric.data[0] * curr_batch_size if batch_metric.nelement() == 1 else batch_metric.data.sum()
            else:
                total_metrics[metric] += batch_metric*curr_batch_size
        remaining -= batch_size
    return {metric: total_score/samples for metric, total_score in total_metrics.items()}

def train_model(config):
    word_level = config.get('word_level', True)
    train_set_path = config.get('train_set_path', 'data/emojipasta_utf8_filtered.json')
    print('Loading training dataset:', train_set_path)
    train_set = load_dataset(train_set_path, word_level=word_level)
    validate_set_path = config.get('validate_set_path', None)
    if validate_set_path is None:
        train_set, validate_set = partition_dataset(train_set, config.get('train_partition', 0.95))
        print('Created train partition with %d examples' % len(train_set.instances))
        print('Created validation partition with %d examples' % len(validate_set.instances))
    else:
        print('Loading validation dataset:', validate_set)
        validate_set = load_dataset(validate_set_path)
    print('Generating vocabulary')
    max_vocab_size = {
        'tokens': config.get('max_token_vocab_size', 40000 if word_level else 2000),
    }
    vocab = Vocabulary.from_dataset(
        train_set,
        max_vocab_size=max_vocab_size
    )
    print('Vocabulary has %d token entries' % vocab.get_vocab_size())
    print('Initializing model')
    model = LanguageModel(vocab, config.get('model_config', {'word_level': word_level}))
    step = 0
    validate_record = []
    batch_size = config.get('batch_size', 40)
    max_instance_length = config.get('max_instance_length', 40)
    validate_metrics = config.get('validate_metrics', ['loss', 'accuracy'])
    validate_interval = config.get('validate_interval', 100)
    validate_samples = config.get('validate_samples', batch_size)
    optim_class = OPTIM_CLASSES[config.get('optim_class', 'adam')]
    optim_args = config.get('optim_args', {'lr':1e-4})
    optim = optim_class(model.parameters(), **optim_args)
    should_stop = False
    prev_accuracy = 0
    teacher_forcing = 1.0
    prev_loss = float('inf')
    def handler(signal, frame):
        nonlocal should_stop
        print("Stopping training at step", step)
        should_stop = True
    signal.signal(signal.SIGINT, handler)
    print('Starting training')
    state = None
    batch_gen = lm_batch_generator(infinite_shuffled_generator(train_set), vocab, batch_size, max_instance_length) 
    while not should_stop:
        print('\rStep %d, loss = %f' % (step, prev_loss), end='')
        if step % validate_interval == 0:
            print('\nValidation scores at step %d:' % step)
            scores = evaluate_metrics(model, validate_set, validate_metrics, batch_size, max_instance_length)
            for metric in validate_metrics:
                print('  %s: %f' % (metric, scores[metric]))
            scores['step'] = step
            validate_record.append(scores)
            if model.training:
                model.eval()
            example_batch = model(np.array([[vocab.get_token_index('@@SOS@@')]]), unroll_length=max_instance_length)
            print('  Sample:', example_batch['text'][0])

        batch = next(batch_gen)

        optim.zero_grad()

        if not model.training:
            model.train()
        batch_out = model(batch, init_state=state, teacher_forcing=teacher_forcing)
        state = batch_out['final_state']
        prev_accuracy = batch_out['accuracy'].data.sum() / batch_size
        teacher_forcing = 1.0 - prev_accuracy if prev_accuracy > 0.1 else 1.0
        loss = batch_out['loss']
        prev_loss = loss.data[0]
        loss.backward()

        optim.step()

        step += 1

    signal.signal(signal.SIGINT, signal.default_int_handler)

    if input('Save model? [y/N]: ').lower() == 'y':
        default_name_index = 0
        while os.path.exists('%s/%d.state.th' % (MODEL_SAVE_DIR, default_name_index)):
            default_name_index += 1
        name = input('Model save name ["%d"]: ' % default_name_index)
        if len(name) == 0:
            name = str(default_name_index)
        path = '%s/%s' % (MODEL_SAVE_DIR, name)
        save_model(model, path)
        with open(path+ '.eval.json', mode='w') as f:
            json.dump(validate_record, f)

def save_model(model: LanguageModel, path):
    ensure_path(path)
    with open(path + '.model.conf.json', mode='w') as f:
        json.dump(model.config, f)
    torch.save(model.state_dict(), path + '.state.th')
    model.vocab.save_to_files(path + '.vocab')

def load_model(path):
    v = Vocabulary.from_files(path + '.vocab')
    with open(path + '.model.conf.json') as f:
        config = json.load(f)
    m = LanguageModel(v, config)
    m.load_state_dict(torch.load(path + '.state.th'))
    return m

if __name__ == '__main__':
    config = {}
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            config = json.load(f)
    train_model(config)
