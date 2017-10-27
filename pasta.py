import json
from tqdm import tqdm

from allennlp.data.fields import TextField
from allennlp.data import Dataset, Instance, Vocabulary
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.models.model import Model
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.modules.token_embedders import TokenCharactersEncoder
from allennlp.modules.seq2vec_encoders.pytorch_seq2vec_wrapper import PytorchSeq2VecWrapper
from allennlp.modules.text_field_embedders.basic_text_field_embedder import BasicTextFieldEmbedder
from allennlp.nn.util import arrays_to_variables

from highway_lstm.highway_lstm_layer import HighwayLSTMLayer

import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

from dropout_embedding import DropoutEmbedding

def load(json_filename):
    with open(json_filename) as f:
        text_list = json.load(f)
    splitter = SpacyWordSplitter()
    indexers = {'tokens': SingleIdTokenIndexer(), 'token_characters': TokenCharactersIndexer()}
    dataset = Dataset(list(tqdm((
        Instance({
            'text': TextField(
                tokens=splitter.split_words(text)[0], 
                token_indexers=indexers
            )
        }) for text in text_list
    ), total=len(text_list))))
    return dataset

class PastaEncoder(Model):
    def __init__(self, vocab, **params):
        super(PastaEncoder, self).__init__(vocab)
        word_emb_size = params.get('word_emb_size', 256)
        word_emb_dropout = params.get('word_emb_dropout', 0.1)
        word_lstm_size = params.get('word_lstm_size', 512)
        word_lstm_layers = params.get('word_lstm_layers', 2)
        word_lstm_dropout = params.get('word_lstm_dropout', 0.1)
        char_emb_size = params.get('char_emb_size', 128)
        char_emb_dropout = params.get('char_emb_dropout', 0.1)
        char_lstm_size = params.get('char_lstm_size', 128)
        char_encoding_dropout = params.get('char_encoding_dropout', 0.1)
        latent_size = params.get('latent_size', 512)
        self.text_field_embedder = BasicTextFieldEmbedder({
            'tokens': DropoutEmbedding(Embedding(
                vocab.get_vocab_size(namespace='tokens'),
                word_emb_size,
                padding_index=0
            ), dropout_rate=word_emb_dropout),
            'token_characters': TokenCharactersEncoder(
                DropoutEmbedding(Embedding(
                    vocab.get_vocab_size(namespace='token_characters'),
                    char_emb_size,
                    padding_index=0
                ), dropout_rate=char_emb_dropout),
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
    def forward(self, data_unsorted: Dataset):
        data_sorted = Dataset(sorted(data_unsorted.instances, key=lambda instance: len(instance.fields['text'].tokens), reverse=True))
        lengths = [len(instance.fields['text'].tokens) for instance in data_sorted.instances]
        array_dict = data_sorted.as_array_dict()
        embedded = self.text_field_embedder({ # TODO fix trying to encode zero-length character sequences
            'tokens': Variable(torch.LongTensor(array_dict['text']['tokens']), requires_grad=False),
            'token_characters': Variable(torch.LongTensor(array_dict['text']['token_characters']), requires_grad=False)
        })
        packed = pack_padded_sequence(embedded, lengths, batch_first=True)
        word_enc_out, word_enc_hidden = self.word_enc_lstm(packed)
        return {'word_enc_out': word_enc_out, 'word_enc_hidden': word_enc_hidden}

