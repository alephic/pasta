from .train import load_model
import numpy as np
import discord

default_model_id = '1'

model = load_model('trained_lm/%s' % (input('Model ID [%s]: ' % default_model_id) or default_model_id))
model.eval()

def get_pasta():
    batch_out = model(np.array([[model.vocab.get_token_index('@@SOS@@')]]), disallow_unk=True, unroll_length=100)
    text = batch_out['text']

client = discord.Client()

@client.event
async def on_ready():
    print('Logged in as', client.user.name, client.user.id)

@client.event
async def on_message(message):
    if message.content.startswith('!pasta'):
