import json
import random
import signal
import os
import sys

from allennlp.data.fields import TextField
from allennlp.data import Dataset, Instance, Vocabulary
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.data.tokenizers import WordTokenizer

import torch

from tqdm import tqdm

from model import PastaEncoder

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

def load_dataset(json_filename):
    with open(json_filename) as f:
        text_list = json.load(f)
    tokenizer = WordTokenizer(start_tokens=['@@SOS@@'])
    indexers = {'tokens': SingleIdTokenIndexer(), 'token_characters': TokenCharactersIndexer()}
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

def slice_instance(instance: Instance, max_instance_length: int):
    start_index = random.randint(0, max(0, len(instance['text'].tokens) - max_instance_length))
    return Instance({'text': TextField(instance['text'].tokens[start_index:start_index + max_instance_length], instance['text'].token_indexers)})

def get_batch(dataset: Dataset, batch_size: int, max_instance_length: int):
    instances = random.sample(dataset.instances, batch_size)
    batch = Dataset([slice_instance(instance, max_instance_length) for instance in instances])
    return batch.as_array_dict(padding_lengths={'tokens': max_instance_length + 1})

def evaluate_metrics(model, dataset, metrics, samples, batch_size, max_instance_length):
    model.eval()
    remaining = samples
    total_metrics = {metric: 0 for metric in metrics}
    while remaining > 0:
        batch = get_batch(dataset, min(remaining, batch_size), max_instance_length)
        batch_out = model(batch)
        for metric in metrics:
            batch_metric = batch_out[metric]
            if isinstance(batch_metric, Variable):
                total_metrics[metric] += batch_metric.sum().data[0]
            else:
                total_metrics[metric] += batch_metric
        remaining -= batch_size
    return {metric: total_score/samples for metric, total_score in total_metrics.items()}

def train_model(config):
    train_set_path = config.get('train_set_path', 'data/emojipasta_utf8.json')
    print('Loading training dataset:', train_set_path)
    train_set = load_dataset(train_set_path)
    validate_set_path = config.get('validate_set_path', None)
    if validate_set_path is None:
        train_set, validate_set = partition_dataset(train_set, config.get('train_partition', 0.9))
        print('Created train partition with %d examples' % len(train_set.instances))
        print('Created validation partition with %d examples' % len(validate_set.instances))
    else:
        print('Loading validation dataset:', validate_set)
        validate_set = load_dataset(validate_set_path)
    print('Generating vocabulary')
    v = Vocabulary.from_dataset(train_set, max_vocab_size=config.get('max_vocab_size', 4000))
    print('Initializing model')
    model = PastaEncoder(v, config.get('model_config', {}))
    print('Indexing datasets')
    train_set.index_instances(v)
    validate_set.index_instances(v)
    step = 0
    validate_record = []
    batch_size = config.get('batch_size', 60)
    max_instance_length = config.get('max_instance_length', 100)
    validate_metrics = config.get('validate_metrics', ['accuracy', 'loss'])
    validate_interval = config.get('validate_interval', len(train_set.instances))
    validate_samples = config.get('validate_samples', 10*batch_size)
    optim_class = OPTIM_CLASSES[config.get('optim_class', 'adam')]
    optim_args = config.get('optim_args', {})
    optim = optim_class(model.parameters(), **optim_args)
    should_stop = False
    def handler(signal, frame):
        nonlocal should_stop
        print("Stopping training at step", step)
        should_stop = True
    signal.signal(signal.SIGINT, handler)
    print('Starting training')
    while not should_stop:
        print('\rStep %d', end='')
        if step % validate_interval == 0:
            print('\nValidation scores at step %d:' % step)
            scores = evaluate_metrics(model, validate_set, validate_metrics, validate_samples, batch_size, max_instance_length)
            for metric in validate_metrics:
                print('  %s: %f' % (metric, step, scores[metric]))
            scores['step'] = step
            validate_record.append(scores)

        batch = get_batch(train_set, batch_size, max_instance_length)

        optim.zero_grad()

        if not model.training:
            model.train()
        batch_out = model(batch)
        loss = batch_out['loss']
        loss.backward()

        optim.step()

        step += 1

    signal.signal(signal.SIGINT, signal.default_int_handler)

    if input('Save model? [y/N]: ').lower() == 'y':
        default_name_index = 0
        while os.path.exists('models/%d.state.th' % default_name_index):
            default_name_index += 1
        name = input('Model save name ["%d"]: ' % default_name_index)
        if len(name) == 0:
            name = str(default_name_index)
        path = 'models/' + name
        save_model(model, path)
        with open(path+ '.eval.json', mode='w') as f:
            json.dump(validate_record, f)

def save_model(model: PastaEncoder, path):
    ensure_path(path)
    with open(path + '.model.conf.json', mode='w') as f:
        json.dump(model.config, f)
    torch.save(model.state_dict(), path + '.state.th')
    model.vocab.save_to_files(path + '.vocab')

def load_model(path):
    v = Vocabulary.from_files(path + '.vocab')
    with open(path + '.model.conf.json') as f:
        config = json.load(f)
    m = PastaEncoder(v, config)
    m.load_state_dict(torch.load(path + '.state.th'))
    return m

if __name__ == '__main__':
    config = None
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            config = json.load(f)
    train_model(config)