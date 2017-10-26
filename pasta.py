import allennlp.data
import json

def load(json_filename):
    with open(json_filename) as f:
        text_list = json.load(f)
    tokenizer = allennlp.data.tokenizers.word_tokenizer.WordTokenizer()
    return list(map(tokenizer.tokenize, text_list))
