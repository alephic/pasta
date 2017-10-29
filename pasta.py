import json
import random
import signal

from allennlp.data.fields import TextField
from allennlp.data import Dataset, Instance, Vocabulary
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.models.model import Model
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.modules.seq2vec_encoders.cnn_encoder import CnnEncoder
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.modules.text_field_embedders.basic_text_field_embedder import BasicTextFieldEmbedder
from allennlp.nn.util import sort_batch_by_length

from highway_lstm.highway_lstm_layer import HighwayLSTMLayer

import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from torch.autograd import Variable

import numpy as np

from tqdm import tqdm

from util import sequence_cross_entropy_with_logits, ensure_path

def get_kl_divergence_from_normal(mu, sigma):
    sigma_squared = sigma ** 2
    return -0.5 * torch.sum(1 + torch.log(sigma_squared) - mu**2 - sigma_squared)

def load(json_filename):
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

class PastaEncoder(Model):
    def __init__(self, vocab, params):
        super(PastaEncoder, self).__init__(vocab)
        self.params = params
        word_emb_size = params.get('word_emb_size', 256)
        self.word_dropout = params.get('word_dropout', 0.05)
        self.word_lstm_size = params.get('word_lstm_size', 512)
        word_lstm_layers = params.get('word_lstm_layers', 2)
        word_lstm_dropout = params.get('word_lstm_dropout', 0.1)
        char_emb_size = params.get('char_emb_size', 64)
        char_cnn_filters = params.get('char_lstm_size', 64)
        latent_size = params.get('latent_size', 256)
        dist_mlp_hidden_size = params.get('dist_mlp_hidden_size', latent_size)
        self.word_emb = Embedding(
            vocab.get_vocab_size(namespace='tokens'),
            word_emb_size,
            padding_index=0
        )
        self.char_emb = TimeDistributed(Embedding(
            vocab.get_vocab_size(namespace='token_characters'),
            char_emb_size,
            padding_index=0
        ))
        self.char_enc = TimeDistributed(CnnEncoder(
            char_emb_size,
            char_cnn_filters,
            output_dim=word_emb_size
        ))
        self.word_enc = HighwayLSTMLayer(
            word_emb_size,
            self.word_lstm_size,
            num_layers=word_lstm_layers,
            recurrent_dropout_prob=word_lstm_dropout
        )
        self.dist_mlp = nn.Sequential(
            nn.Linear(self.word_lstm_size, dist_mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(dist_mlp_hidden_size, 2*latent_size)
        )
        self.word_dec = HighwayLSTMLayer(
            word_emb_size + latent_size,
            self.word_lstm_size,
            num_layers=word_lstm_layers,
            recurrent_dropout_prob=word_lstm_dropout
        )
        self.word_dec_project = nn.Linear(
            self.word_lstm_size,
            vocab.get_vocab_size(namespace='tokens') + 1 # extra slot for EOS
        )
        #self.char_dec_lstm_cell = nn.modules.LSTMCell(self.word_lstm_size, char_lstm_size)
        #self.char_dec_project = nn.Linear(
        #    char_lstm_size,
        #    vocab.get_vocab_size(namespace='token_characters') + 1 # extra slot for EOS
        #)

        # Model is GPU-only
        self.cuda()

    def do_word_dropout(self, token_indices_array):
        if self.training:
            token_mask = (token_indices_array != 0).astype(int)
            drop_mask = (np.random.rand(*token_indices_array.shape) < self.word_dropout).astype(int)
            return (token_indices_array * (1-drop_mask)) + token_mask*drop_mask
        else:
            return token_indices_array

    def get_target_indices(self, token_indices_array, input_lengths):
        array = token_indices_array[:, 1:].copy()
        eos = self.vocab.get_vocab_size(namespace='tokens')
        for i, length in enumerate(input_lengths):
            if length <= array.shape[1]:
                array[i, length-1] = eos
        return Variable(torch.cuda.LongTensor(array), requires_grad=False)
    
    def embed_inputs(self, array_dict):
        tokens_array = self.do_word_dropout(array_dict['text']['tokens'])
        chars_array = array_dict['text']['token_characters']
        tokens_array_no_unks = (tokens_array != 1).astype(int) * tokens_array
        embedded = self.word_emb(Variable(torch.cuda.LongTensor(tokens_array_no_unks), requires_grad=False))
        chars_embedded = self.char_emb(Variable(torch.cuda.LongTensor(chars_array), requires_grad=False))
        chars_mask = Variable(torch.cuda.FloatTensor((chars_array != 0).astype(float)), requires_grad=False)
        chars_encoded = self.char_enc(chars_embedded, chars_mask)
        unk_mask = Variable(torch.cuda.FloatTensor((tokens_array == 1).astype(float)), requires_grad=False)
        return embedded + unk_mask.unsqueeze(2).expand_as(chars_encoded) * chars_encoded

    def get_latent_dist_params(self, inputs, input_lengths_var: Variable):
        packed_inputs = pack_padded_sequence(inputs, input_lengths_var.data.tolist(), batch_first=True)
        word_enc_out = self.word_enc(packed_inputs)[0]
        padded = pad_packed_sequence(word_enc_out, batch_first=True)[0]
        last_indices = (input_lengths_var - 1).view(padded.size()[0], 1, 1).expand(-1, -1, padded.size()[2])
        encodings = padded.gather(1, last_indices).squeeze(1)
        return self.dist_mlp(encodings).chunk(2, dim=1)

    def get_reconstruction_logits(self, latent_var, inputs, input_lengths_var: Variable):
        inputs_and_latent = torch.cat(
            (
                inputs[:, :-1],
                latent_var.unsqueeze(1).expand(-1, inputs.size()[1] - 1, -1)
            ),
            2
        )
        packed_inputs = pack_padded_sequence(inputs_and_latent, [min(length, inputs_and_latent.size()[1]) for length in input_lengths_var.data.tolist()], batch_first=True)
        word_dec_out = self.word_dec(packed_inputs)[0]
        packed_logits = PackedSequence(self.word_dec_project(word_dec_out.data), word_dec_out.batch_sizes)
        return pad_packed_sequence(packed_logits, batch_first=True)[0]

    def forward(self, batch, kl_weight=1.0, reconstruct=True):
        output_dict = {}
        lengths = (batch['text']['tokens'] != 0).astype(int).sum(axis=1).tolist()
        lengths_var = Variable(torch.cuda.LongTensor(lengths), requires_grad=False)
        embedded = self.embed_inputs(batch)
        sorted_embedded, sorted_lengths_var, unsort_indices = sort_batch_by_length(embedded, lengths_var)
        mu, sigma = self.get_latent_dist_params(sorted_embedded, sorted_lengths_var)
        output_dict['latent_mean'] = torch.index_select(mu, 0, unsort_indices)
        output_dict['latent_stdev'] = torch.index_select(sigma, 0, unsort_indices)
        if reconstruct:
            sampled_latent = torch.normal(mu, sigma)
            reconstruction_logits = self.get_reconstruction_logits(sampled_latent, sorted_embedded, sorted_lengths_var)
            reconstruction_logits = torch.index_select(reconstruction_logits, 0, unsort_indices)
            target_indices = self.get_target_indices(batch['text']['tokens'], lengths)
            xent_mask = (target_indices != 0).float()
            xent = sequence_cross_entropy_with_logits(reconstruction_logits, target_indices, xent_mask, batch_average=False)
            dkl = get_kl_divergence_from_normal(mu, sigma)
            loss_vec = kl_weight * dkl + xent
            _, reconstructed = torch.max(reconstruction_logits, 2)
            output_dict['reconstructed'] = reconstructed
            output_dict['target'] = target_indices
            output_dict['accuracy'] = torch.sum((reconstructed == target_indices).float() * xent_mask, 1).float() / lengths_var.float()
            output_dict['loss'] = torch.sum(loss_vec)
        return output_dict

    def decode(self, output_dict, max_length=4000):
        sampled_latent = torch.normal(output_dict['latent_mean'], output_dict['latent_stdev'])
        batch_size = sampled_latent.size()[0]
        eos = self.vocab.get_vocab_size(namespace='tokens')
        h_prev = None
        c_prev = None
        input_indices = Variable(torch.LongTensor([self.vocab.get_token_index('@@SOS@@')]*batch_size), requires_grad=False)
        lengths = [1]*batch_size
        open_seqs = set(range(batch_size))
        decoded_slices = []
        while len(open_seqs) > 0 and len(decoded_slices) < max_length:
            inputs = self.word_emb(input_indices).unsqueeze(1)
            _, h, c = self.word_dec(pack_padded_sequence(inputs, lengths, batch_first=True), h0=h_prev, c0=c_prev)
            h_prev = h[:, 0] # all layers, t=0
            c_prev = c[:, 0]
            out = h[-1, 0] # out.size(): (batch_size, word_lstm_size)
            logits = self.word_dec_project(out)
            _, argmax = torch.max(logits, 1)
            input_indices = argmax
            decoded_slices.append(argmax)
            for i in list(open_seqs):
                if argmax.data[i] == eos:
                    open_seqs.remove(i)
        output_dict['decoded'] = torch.stack(decoded_slices, 1)
        return output_dict

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

def train_model(train_set: Dataset, validate_set: Dataset, params=None):
    params = params or {}
    print('Generating vocabulary')
    v = Vocabulary.from_dataset(train_set, max_vocab_size=params.get('max_vocab_size', 4000))
    print('Creating model')
    model = PastaEncoder(v, params.get('model_params', {}))
    print('Indexing datasets')
    train_set.index_instances(v)
    validate_set.index_instances(v)
    step = 0
    batch_size = params.get('batch_size', 60)
    max_instance_length = params.get('max_instance_length', 100)
    validate_metrics = params.get('validate_metrics', ['accuracy', 'loss'])
    validate_interval = params.get('validate_interval', len(train_set.instances))
    validate_samples = params.get('validate_samples', 10*batch_size)
    optim_class = OPTIM_CLASSES[params.get('optim_class', 'adam')]
    optim_params = params.get('optim_params', {})
    optim = optim_class(model.parameters(), **optim_params)
    should_stop = False
    def handler(signal, frame):
        print("Stopping training")
        nonlocal should_stop
        should_stop = True
    signal.signal(signal.SIGINT, handler)
    while True:
        if step % validate_interval == 0:
            print('Validation scores at step %d:' % step)
            scores = evaluate_metrics(model, validate_set, validate_metrics, validate_samples, batch_size, max_instance_length)
            for metric in validate_metrics:
                print('  %s: %f' % (metric, step, scores[metric]))

        batch = get_batch(train_set, batch_size, max_instance_length)

        optim.zero_grad()

        if not model.training:
            model.train()
        batch_out = model(batch)
        loss = batch_out['loss']
        loss.backward()

        optim.step()

        step += 1

    if input('Save model? [y/N]: ').lower() == 'y':
        path = input('Model save path [models/0]: ')
        if len(path) == 0:
            path = 'models/0'
        save_model(model, path)
        with open(path + '.train.conf.json', mode='w') as f:
            json.dump(params, f)

    return model

def save_model(model: PastaEncoder, path):
    ensure_path(path)
    with open(path + '.conf.json', mode='w') as f:
        json.dump(model.params, f)
    torch.save(model.state_dict(), path + '.state.th')
    model.vocab.save_to_files(path + '.vocab')

def load_model(path):
    v = Vocabulary.from_files(path + '.vocab')
    with open(path + '.conf.json') as f:
        params = json.load(f)
    m = PastaEncoder(v, params)
    m.load_state_dict(torch.load(path + '.state.th'))
    return m

