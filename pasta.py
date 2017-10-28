import json

from allennlp.data.fields import TextField
from allennlp.data import Dataset, Instance, Vocabulary
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.models.model import Model
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.modules.token_embedders import TokenCharactersEncoder
from allennlp.modules.seq2vec_encoders.pytorch_seq2vec_wrapper import PytorchSeq2VecWrapper
from allennlp.modules.text_field_embedders.basic_text_field_embedder import BasicTextFieldEmbedder
from allennlp.nn.util import sort_batch_by_length

from highway_lstm.highway_lstm_layer import HighwayLSTMLayer

import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

import numpy as np

from tqdm import tqdm

from dropout_embedding import DropoutEmbedding

def load(json_filename):
    with open(json_filename) as f:
        text_list = json.load(f)
    splitter = SpacyWordSplitter()
    indexers = {'tokens': SingleIdTokenIndexer(), 'token_characters': TokenCharactersIndexer()}
    dataset = Dataset(list(tqdm((
        Instance({
            'text': TextField(
                tokens=splitter.split_words(text), 
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
        word_lstm_size = params.get('word_lstm_size', 512)
        word_lstm_layers = params.get('word_lstm_layers', 2)
        word_lstm_dropout = params.get('word_lstm_dropout', 0.1)
        char_emb_size = params.get('char_emb_size', 64)
        char_lstm_size = params.get('char_lstm_size', 64)
        char_encoding_dropout = params.get('char_encoding_dropout', 0.1)
        latent_size = params.get('latent_size', 256)
        dist_mlp_hidden_size = params.get('dist_mlp_hidden_size', latent_size)
        self.text_field_embedder = BasicTextFieldEmbedder({
            'tokens': Embedding(
                vocab.get_vocab_size(namespace='tokens'),
                word_emb_size,
                padding_index=0
            ),
            'token_characters': TokenCharactersEncoder(
                Embedding(
                    vocab.get_vocab_size(namespace='token_characters'),
                    char_emb_size,
                    padding_index=0
                ),
                PytorchSeq2VecWrapper(
                    nn.modules.LSTM(
                        char_emb_size,
                        char_lstm_size,
                        batch_first=True
                    )
                ),
                dropout=char_encoding_dropout
            )
        })
        self.word_enc_lstm = HighwayLSTMLayer(
            self.text_field_embedder.get_output_dim(),
            word_lstm_size,
            num_layers=word_lstm_layers,
            recurrent_dropout_prob=word_lstm_dropout
        )
        self.dist_mlp = nn.Sequential(
            nn.Linear(word_lstm_size, dist_mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(dist_mlp_hidden_size, 2*latent_size)
        )
        self.word_dec_lstm_cells = nn.ModuleList([
            nn.modules.LSTMCell(latent_size, word_lstm_size) for i in range(word_lstm_layers)
        ])
        self.word_dec_project = nn.Linear(
            word_lstm_size,
            vocab.get_vocab_size(namespace='tokens')
        )
        self.char_dec_lstm_cell = nn.modules.LSTMCell(word_lstm_size, char_lstm_size)
        self.char_dec_project = nn.Linear(
            char_lstm_size,
            vocab.get_vocab_size(namespace='token_characters')
        )
        # GPU-only
        self.cuda()
    def do_word_dropout(self, token_indices_array):
        if self.training:
            token_mask = (token_indices_array != 0).astype(int)
            drop_mask = (np.random.rand(*token_indices_array.shape) < self.word_dropout).astype(int)
            return (token_indices_array * (1-drop_mask)) + token_mask*drop_mask
        else:
            return token_indices_array
    def get_latent_dist_params(self, data: Dataset):
        lengths_list = [len(instance.fields['text'].tokens) for instance in data.instances]
        lengths_var = Variable(torch.cuda.LongTensor(lengths_list), requires_grad=False)
        batch_size = len(data.instances)
        array_dict = data.as_array_dict()
        embedded = self.text_field_embedder({
            'tokens': Variable(torch.cuda.LongTensor(self.do_word_dropout(array_dict['text']['tokens'])), requires_grad=False),
            'token_characters': Variable(torch.cuda.LongTensor(array_dict['text']['token_characters']), requires_grad=False)
        })
        embedded_sorted, lengths_sorted, restoration_indices = sort_batch_by_length(embedded, lengths_var)
        packed = pack_padded_sequence(embedded_sorted, lengths_sorted.data.tolist(), batch_first=True)
        word_enc_out = self.word_enc_lstm(packed)[0]
        padded = pad_packed_sequence(word_enc_out, batch_first=True)[0]
        padded = padded.index_select(0, restoration_indices)
        last_indices = (lengths_var - 1).view(batch_size, 1, 1).expand(-1, -1, padded.size()[2])
        encodings = padded.gather(1, last_indices).squeeze(1)
        return self.dist_mlp(encodings)

def test_encode():
    d = load('data/emojipasta.json')
    v = Vocabulary.from_dataset(d)
    b = Dataset(d.instances[:10])
    b.index_instances(v)
    enc = PastaEncoder(v).cuda()
    return enc.get_latent_dist_params(b)