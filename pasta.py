import json

from allennlp.data.fields import TextField
from allennlp.data import Dataset, Instance, Vocabulary
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.models.model import Model
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.modules.seq2vec_encoders.cnn_encoder import CnnEncoder
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.modules.text_field_embedders.basic_text_field_embedder import BasicTextFieldEmbedder
from allennlp.nn.util import sort_batch_by_length, sequence_cross_entropy_with_logits

from highway_lstm.highway_lstm_layer import HighwayLSTMLayer

import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from torch.autograd import Variable

import numpy as np

from tqdm import tqdm

from dropout_embedding import DropoutEmbedding


def get_kl_divergence_from_normal(mu, sigma):
    sigma_squared = sigma ** 2
    return -0.5 * torch.sum(1 + torch.log(sigma_squared) - mu**2 - sigma_squared)

def get_sequence_mask_from_lengths(lengths_var, max_length):
    indices = Variable(torch.cuda.LongTensor(np.arange(max_length)), requires_grad=False)
    indices = indices.unsqueeze(0).expand(lengths_var.size()[0], -1)
    return (indices < lengths_var.unsqueeze(1).expand(-1, max_length)).float()

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

class PastaEncoder(Model):
    def __init__(self, vocab, **params):
        super(PastaEncoder, self).__init__(vocab)
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
        array = np.roll(token_indices_array, -1, axis=1)
        array[:, -1] = 0
        eos = self.vocab.get_vocab_size(namespace='tokens')
        for i, length in enumerate(input_lengths):
            array[i, length-1] = eos
        return Variable(torch.cuda.LongTensor(array), requires_grad=False)
    
    def embed_inputs(self, array_dict):
        tokens_array = self.do_word_dropout(array_dict['text']['tokens'])
        chars_array = array_dict['text']['token_characters']
        tokens_array_no_unks = (tokens_array != 1).astype(int) * tokens_array
        embedded = self.word_emb(Variable(torch.cuda.LongTensor(tokens_array_no_unks), requires_grad=False))
        chars_embedded = self.char_emb(Variable(torch.cuda.LongTensor(chars_array), requires_grad=False))
        chars_encoded = self.char_enc(chars_embedded)
        print(embedded.size())
        print(chars_encoded.size())
        unk_mask = Variable(torch.cuda.FloatTensor((tokens_array == 1).astype(float)), requires_grad=False)
        return embedded + unk_mask.unsqueeze(2).expand_as(chars_encoded) * chars_encoded

    def get_latent_dist_params(self, inputs, input_lengths_var: Variable):
        packed_inputs = pack_padded_sequence(inputs, input_lengths.data.tolist(), batch_first=True)
        word_enc_out = self.word_enc(packed_inputs)[0]
        padded = pad_packed_sequence(word_enc_out, batch_first=True)[0]
        last_indices = (input_lengths_var - 1).view(padded.size()[0], 1, 1).expand(-1, -1, padded.size()[2])
        encodings = padded.gather(1, last_indices).squeeze(1)
        return self.dist_mlp(encodings).chunk(2, dim=1)

    def get_reconstruction_logits(self, latent_var, inputs, input_lengths_var: Variable):
        inputs_and_latent = torch.cat(
            (
                inputs,
                latent_var.unsqueeze(1).expand(-1, inputs.size()[1], -1)
            ),
            2
        )
        packed_inputs = pack_padded_sequence(inputs_and_latent, input_lengths_var.data.tolist(), batch_first=True)
        word_dec_out = self.word_dec(packed_inputs)[0]
        packed_logits = PackedSequence(self.word_dec_project(word_dec_out.data), word_dec_out.batch_sizes)
        return pad_packed_sequence(packed_logits, batch_first=True)[0]

    def forward(self, data: Dataset, kl_weight=1.0, reconstruct=True):
        output_dict = {}
        array_dict = data.as_array_dict()
        embedded = self.embed_inputs(array_dict)
        lengths = [len(instance.fields['text'].tokens) for instance in data.instances]
        lengths_var = Variable(torch.cuda.LongTensor(lengths), requires_grad=False)
        sorted_embedded, sorted_lengths_var, unsort_indices = sort_batch_by_length(embedded, lengths_var)
        mu, sigma = self.get_latent_dist_params(sorted_embedded, sorted_lengths_var)
        output_dict['latent_mean'] = mu
        output_dict['latent_stdev'] = sigma
        if reconstruct:
            sampled_latent = torch.normal(mu, sigma)
            reconstruction_logits = self.get_reconstruction_logits(sampled_latent, sorted_embedded, sorted_lengths_var)
            target_indices = self.get_target_indices(array_dict['text']['tokens'], lengths)
            xent_mask = get_sequence_mask_from_lengths(sorted_lengths_var, target_indices.size()[1])
            xent = sequence_cross_entropy_with_logits(reconstruction_logits, target_indices, xent_mask, batch_average=False)
            dkl = get_kl_divergence_from_normal(mu, sigma)
            loss_vec = kl_weight * dkl + xent
            output_dict['logits'] = reconstruction_logits
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

def test_forward():
    d = load('data/emojipasta.json')
    v = Vocabulary.from_dataset(d)
    b = Dataset(d.instances[:10])
    b.index_instances(v)
    enc = PastaEncoder(v).cuda()
    return enc.forward(b)