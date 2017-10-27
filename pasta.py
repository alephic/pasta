import json
from tqdm import tqdm

from allennlp.data.fields import TextField
from allennlp.data import Dataset, Instance, Vocabulary
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.models.model import Model
from allennlp.modules.augmented_lstm import AugmentedLstm
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.modules.token_embedders import TokenCharactersEncoder
from allennlp.modules.seq2vec_encoders.pytorch_seq2vec_wrapper import PytorchSeq2VecWrapper
from allennlp.modules.text_field_embedders.basic_text_field_embedder import BasicTextFieldEmbedder

import torch.nn as nn
import torch

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
        word_emb_size = params.get('word_emb_size', 256)
        word_emb_dropout = params.get('word_emb_dropout', 0.1)
        word_lstm_size = params.get('word_lstm_size', 512)
        word_lstm_layers = params.get('word_lstm_layers', 2)
        word_lstm_dropout = params.get('word_lstm_dropout', 0.1)
        char_emb_size = params.get('char_emb_size', 128)
        char_emb_dropout = params.get('char_emb_dropout', 0.1)
        char_lstm_size = params.get('char_lstm_size', 128)
        char_encoding_dropout = params.get('char_encoding_dropout', 0.1)
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
        self.word_lstm = AugmentedLstm(
            self.text_field_embedder.get_output_dim(),
            word_lstm_size,
            recurrent_dropout_probability=word_lstm_dropout
        )
            