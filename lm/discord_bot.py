from .train import load_model
from allennlp.data.tokenizers.word_splitter import LettersDigitsWordSplitter
import numpy as np
import re
import discord

DEFAULT_MODEL_ID = '1'

SUMMON_COMMANDS = list(map((lambda s: '!%s' % s), ['pasta', 'ravioli', 'linguini', 'spaghetti', 'tortellini', 'fettucini', 'rotini', 'lasagna', 'macaroni', 'penne', 'rigatoni']))

def cleanup(text):
    text = re.sub(r"\[ [Ss]ource( :)? \] \(.*\)", r"", text)
    text = re.sub(r" ' (([tsm]|ve|ll)($| ))", r"'\1", text)
    text = re.sub(r" ([.?!:;,])", r"\1", text)
    text = re.sub(r'" (.+) "', r'"\1"', text)
    text = re.sub(r"\( (.+) \)", r"\(\1\)", text)
    text = re.sub(r"\[ (.+) \]", r"\[\1\]", text)
    text = re.sub(r'([^ \w"]+) (?=[^\w"])', r"\1", text)
    return text

def load_model_prompt():
    m = load_model('trained_lm/%s' % (input('Model ID [%s]: ' % DEFAULT_MODEL_ID) or DEFAULT_MODEL_ID))
    m.eval()
    return m

def get_pasta(model):
    batch_out = model(np.array([[model.vocab.get_token_index('@@SOS@@')]]), disallow_unk=True, unroll_length=100)
    text = batch_out['text'][0]
    state = batch_out['final_state']
    last_indices = batch_out['indices'][:, -1].data.numpy().reshape(1, 1)

    end = text.find('@@EOS@@')
    if end != -1:
        text = text[:end]
    else:
        while True:
            batch_out = model(last_indices, init_states=state, disallow_unk=True, unroll_length=100)
            new_text = batch_out['text'][0]
            end = new_text.find('@@EOS@@')
            state = batch_out['final_state']
            last_indices = batch_out['indices'][:, -1].data.numpy().reshape(1, 1)
            if end != -1:
                new_text = new_text[:end]
                text += new_text
                break
            else:
                text += new_text
    return cleanup(text)

def get_pasta_with_prompt(model, prompt_text):
    tokens = LettersDigitsWordSplitter().split_words(prompt_text)
    inp = np.array([[model.vocab.get_token_index('@@SOS@@')] + [model.vocab.get_token_index(t.text) for t in tokens]])
    batch_out = model(inp)
    state = batch_out['final_state']
    last_inp = inp[:, -1:]
    batch_out = model(last_inp, init_states=state, disallow_unk=True, unroll_length=100)
    text = batch_out['text'][0]
    state = batch_out['final_state']
    last_indices = batch_out['indices'][:, -1].data.numpy().reshape(1, 1)

    end = text.find('@@EOS@@')
    if end != -1:
        text = text[:end]
    else:
        while True:
            batch_out = model(last_indices, init_states=state, disallow_unk=True, unroll_length=100)
            new_text = batch_out['text'][0]
            end = new_text.find('@@EOS@@')
            state = batch_out['final_state']
            last_indices = batch_out['indices'][:, -1].data.numpy().reshape(1, 1)
            if end != -1:
                new_text = new_text[:end]
                text += new_text
                break
            else:
                text += new_text
    return '%s %s' % (prompt_text.rstrip(' '), cleanup(text))

    
if __name__ == "__main__":

    model = load_model_prompt()
    model.eval()

    client = discord.Client()

    @client.event
    async def on_ready():
        print('Logged in as', client.user.name, client.user.id)

    @client.event
    async def on_message(message):
        summoned = False
        if message.content.startswith('!'):
            for summon_command in SUMMON_COMMANDS:
                if message.content.startswith(summon_command):
                    if re.match(r'\w', message.content[len(summon_command):]):
                        msg = get_pasta_with_prompt(model, message.content[len(summon_command):].lstrip(' '))
                    else:
                        msg = get_pasta(model)
                    await client.send_message(message.channel, msg)
                    return
    
    client.run(input('Token: '))
