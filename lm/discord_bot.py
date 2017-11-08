from .train import load_model
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
                    summoned = True
                    break
        if summoned:
            await client.send_message(message.channel, get_pasta(model))
    
    client.run(input('Token: '))
