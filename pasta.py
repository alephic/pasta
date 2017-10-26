import json

from allennlp.data.fields import TextField
from allennlp.data import Dataset, Instance, Vocabulary
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter

def load(json_filename):
    with open(json_filename) as f:
        text_list = json.load(f)
    splitter = SpacyWordSplitter()
    indexers = {'tokens': SingleIdTokenIndexer(), 'token_characters': TokenCharactersIndexer()}
    dataset = Dataset([
        Instance({
            'text': TextField(
                tokens=splitter.split_words(text)[0], 
                token_indexers=indexers
            )
        }) for text in text_list
    ])
    return dataset