from allennlp.models.model import Model
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.modules.seq2vec_encoders.cnn_encoder import CnnEncoder
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.nn.util import sort_batch_by_length

from highway_lstm.highway_lstm_layer import HighwayLSTMLayer

import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from torch.autograd import Variable

import numpy as np

def sequence_cross_entropy_with_logits(logits: torch.FloatTensor,
                                       targets: torch.LongTensor,
                                       weights: torch.FloatTensor,
                                       batch_average: bool = True) -> torch.FloatTensor:
    """
    Computes the cross entropy loss of a sequence, weighted with respect to
    some user provided weights. Note that the weighting here is not the same as
    in the :func:`torch.nn.CrossEntropyLoss()` criterion, which is weighting
    classes; here we are weighting the loss contribution from particular elements
    in the sequence. This allows loss computations for models which use padding.
    Parameters
    ----------
    logits : ``torch.FloatTensor``, required.
        A ``torch.FloatTensor`` of size (batch_size, sequence_length, num_classes)
        which contains the unnormalized probability for each class.
    targets : ``torch.LongTensor``, required.
        A ``torch.LongTensor`` of size (batch, sequence_length) which contains the
        index of the true class for each corresponding step.
    weights : ``torch.FloatTensor``, required.
        A ``torch.FloatTensor`` of size (batch, sequence_length)
    batch_average : bool, optional, (default = True).
        A bool indicating whether the loss should be averaged across the batch,
        or returned as a vector of losses per batch element.
    Returns
    -------
    A torch.FloatTensor representing the cross entropy loss.
    If ``batch_average == True``, the returned loss is a scalar.
    If ``batch_average == False``, the returned loss is a vector of shape (batch_size,).
    """
    # shape : (batch * sequence_length, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # shape : (batch * sequence_length, num_classes)
    log_probs_flat = torch.nn.functional.log_softmax(logits_flat, dim=1)
    # shape : (batch * max_len, 1)
    targets_flat = targets.view(-1, 1).long()

    # Contribution to the negative log likelihood only comes from the exact indices
    # of the targets, as the target distributions are one-hot. Here we use torch.gather
    # to extract the indices of the num_classes dimension which contribute to the loss.
    # shape : (batch * sequence_length, 1)
    negative_log_likelihood_flat = - torch.gather(log_probs_flat, dim=1, index=targets_flat)
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood_flat.view(*targets.size())
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood * weights.float()
    # shape : (batch_size,)
    per_batch_loss = negative_log_likelihood.sum(1) / (weights.sum(1).float() + 1e-13)

    if batch_average:
        num_non_empty_sequences = ((weights.sum(1) > 0).float().sum() + 1e-13)
        return per_batch_loss.sum() / num_non_empty_sequences
    return per_batch_loss

def get_kl_divergence_from_normal(mu, sigma):
    sigma_squared = sigma ** 2
    return -0.5 * torch.sum(1 + torch.log(sigma_squared) - mu**2 - sigma_squared, 1)

class PastaEncoder(Model):
    def __init__(self, vocab, config):
        super(PastaEncoder, self).__init__(vocab)
        self.config = config
        word_emb_size = config.get('word_emb_size', 128)
        self.input_word_dropout = config.get('input_word_dropout', 0.05)
        self.decode_word_dropout = config.get('decode_word_dropout', 0.0)
        self.decode_teacher_force_rate = config.get('decode_teacher_force_rate', 0.05)
        self.word_lstm_size = config.get('word_lstm_size', 256)
        word_lstm_layers = config.get('word_lstm_layers', 2)
        self.word_lstm_dropout = config.get('word_lstm_dropout', 0.1)
        char_emb_size = config.get('char_emb_size', 32)
        char_cnn_filters = config.get('char_cnn_filters', 32)
        latent_size = config.get('latent_size', 256)
        dist_mlp_hidden_size = config.get('dist_mlp_hidden_size', latent_size)
        self.word_emb = Embedding(
            vocab.get_vocab_size(namespace='tokens'),
            word_emb_size,
            padding_index=0
        )
        self.char_emb = Embedding(
            vocab.get_vocab_size(namespace='token_characters'),
            char_emb_size,
            padding_index=0
        )
        self.char_enc = CnnEncoder(
            char_emb_size,
            char_cnn_filters,
            output_dim=word_emb_size
        )
        self.word_enc = HighwayLSTMLayer(
            word_emb_size,
            self.word_lstm_size,
            num_layers=word_lstm_layers,
            recurrent_dropout_prob=self.word_lstm_dropout
        )
        self.dist_mlp = nn.Sequential(
            nn.Linear(self.word_lstm_size, dist_mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(dist_mlp_hidden_size, 2*latent_size)
        )
        self.word_dec = HighwayLSTMLayer(
            word_emb_size + latent_size,
            self.word_lstm_size,
            num_layers=word_lstm_layers
        )
        self.word_dec_project = nn.Linear(
            self.word_lstm_size,
            vocab.get_vocab_size(namespace='tokens')
        )
        #self.char_dec_lstm_cell = nn.modules.LSTMCell(self.word_lstm_size, char_lstm_size)
        #self.char_dec_project = nn.Linear(
        #    char_lstm_size,
        #    vocab.get_vocab_size(namespace='token_characters')
        #)

        # Model is GPU-only
        self.cuda()

    def do_input_word_dropout(self, token_indices_array):
        if self.training:
            token_mask = (token_indices_array != 0).astype(int)
            drop_mask = (np.random.rand(*token_indices_array.shape) < self.input_word_dropout).astype(int)
            return (token_indices_array * (1-drop_mask)) + token_mask*drop_mask
        else:
            return token_indices_array
    
    def embed_inputs(self, array_dict):
        tokens_array = self.do_input_word_dropout(array_dict['text']['tokens'])
        tokens_array_flat = tokens_array.reshape(tokens_array.size)
        chars_array = array_dict['text']['token_characters']

        tokens_array_no_unks = (tokens_array_flat != 1).astype(int) * tokens_array_flat
        embedded_flat = self.word_emb(Variable(torch.cuda.LongTensor(tokens_array_no_unks), requires_grad=False))

        unk_indices = np.ma.array(np.arange(tokens_array.size), mask=tokens_array_flat != 1).compressed()
        chars_flat = chars_array.reshape(chars_array.shape[0]*chars_array.shape[1], chars_array.shape[2])[unk_indices]
        max_num_chars = (chars_flat != 0).astype(int).sum(axis=1).max()
        chars_flat = chars_flat[:, :max_num_chars]

        chars_embedded = self.char_emb(Variable(torch.cuda.LongTensor(chars_flat), requires_grad=False))
        chars_mask = Variable(torch.cuda.FloatTensor((chars_flat != 0).astype(float)), requires_grad=False)
        chars_encoded = self.char_enc(chars_embedded, chars_mask)

        embedded_flat[Variable(torch.cuda.LongTensor(unk_indices), requires_grad=False)] = chars_encoded
        return embedded_flat.view(tokens_array.shape[0], tokens_array.shape[1], embedded_flat.size()[1]).contiguous()

    def get_latent_dist_params(self, inputs, input_lengths_var: Variable):
        sorted_inputs, sorted_lengths_var, unsort_indices = sort_batch_by_length(inputs, input_lengths_var)
        packed_inputs = pack_padded_sequence(sorted_inputs, sorted_lengths_var.data.tolist(), batch_first=True)
        word_enc_out = self.word_enc(packed_inputs)[0]
        padded = pad_packed_sequence(word_enc_out, batch_first=True)[0]
        last_indices = (sorted_lengths_var - 1).view(padded.size()[0], 1, 1).expand(-1, -1, padded.size()[2])
        encodings = padded.gather(1, last_indices).squeeze(1).index_select(0, unsort_indices)
        return self.dist_mlp(encodings).chunk(2, dim=1)

    def forward(self, batch, kl_weight=1.0, reconstruct=True):
        output_dict = {}
        lengths = (batch['text']['tokens'] != 0).astype(int).sum(axis=1).tolist()
        lengths_var = Variable(torch.cuda.LongTensor(lengths), requires_grad=False)
        embedded = self.embed_inputs(batch)
        mu, sigma = self.get_latent_dist_params(embedded, lengths_var)
        output_dict['latent_mean'] = mu
        output_dict['latent_stdev'] = sigma
        if reconstruct or self.training:
            # Reconstruct
            target_indices = Variable(torch.cuda.LongTensor(batch['text']['tokens'][:, 1:].copy()), requires_grad=False)
            output_dict['target_indices'] = target_indices
            output_dict['decode_inputs'] = Variable(torch.cuda.LongTensor(batch['text']['tokens'][:, 0]), requires_grad=False)
            self.decode(output_dict)
            reconstructed_logits = output_dict['decoded_logits']
            xent_mask = (target_indices != 0).float()
            xent = sequence_cross_entropy_with_logits(reconstructed_logits, target_indices, xent_mask, batch_average=False)
            dkl = get_kl_divergence_from_normal(mu, sigma)
            loss_vec = kl_weight * dkl + xent
            output_dict['loss'] = torch.sum(loss_vec)
        return output_dict

    def decode(self, output_dict, max_length=200):
        target_indices = output_dict.get('target_indices')
        batch_size = target_indices.size()[0]
        length = target_indices.size()[1] if target_indices is not None else max_length
        sampled_latent = torch.normal(output_dict['latent_mean'], output_dict['latent_stdev']).unsqueeze(1)
        h_prev = None
        c_prev = None
        input_indices = output_dict['decode_inputs']
        lengths = [1]*batch_size
        decoded_logits_slices = []
        decoded_indices_slices = []
        # Pre-sample recurrent dropout
        if self.training:
            dropout_weights = torch.cuda.FloatTensor(self.word_dec.num_layers, batch_size, self.word_dec.hidden_size)
            dropout_weights.bernoulli_(1 - self.word_lstm_dropout)
            dropout_weights = Variable(dropout_weights, requires_grad=False)
        else:
            dropout_weights = None
        # Loop over timesteps
        for t in range(length):
            # Word dropout
            if self.decode_word_dropout > 0:
                input_drop_mask = torch.cuda.LongTensor(batch_size)
                input_drop_mask.bernoulli_(self.decode_word_dropout)
                input_drop_mask = Variable(input_drop_mask, requires_grad=False)
                input_indices = (input_indices * (1 - input_drop_mask)) + input_drop_mask

            inputs = torch.cat(
                (self.word_emb(input_indices).unsqueeze(1), sampled_latent),
                2
            )
            _, h, c = self.word_dec(
                pack_padded_sequence(inputs, lengths, batch_first=True),
                h0=h_prev,
                c0=c_prev,
                dropout_weights=dropout_weights
            )
            h_prev = h[:, 0] # all layers, t=0
            c_prev = c[:, 0]
            out = h[-1, 0] # out.size(): (batch_size, word_lstm_size)
            logits = self.word_dec_project(out)
            decoded_logits_slices.append(logits)

            _, argmax = torch.max(logits, 1)
            decoded_indices_slices.append(argmax)
            input_indices = argmax

            # Teacher forcing
            if self.training and self.decode_teacher_force_rate > 0:
                input_force_mask = torch.cuda.LongTensor(batch_size)
                input_force_mask.bernoulli_(self.decode_teacher_force_rate)
                input_force_mask = Variable(input_force_mask, requires_grad=False)
                gold_indices = Variable(torch.cuda.LongTensor(target_indices[:, t]), requires_grad=False)
                input_indices = (input_indices * (1 - input_force_mask)) + (input_force_mask * gold_indices)
        
        output_dict['decoded_logits'] = torch.stack(decoded_logits_slices, 1)
        output_dict['decoded_indices'] = torch.stack(decoded_indices_slices, 1)
        return output_dict

    def make_readable(self, decoded):
        out = []
        eos = self.vocab.get_token_index('@@EOS@@')
        for i in range(decoded.size()[0]):
            out_line = []
            for t in range(decoded.size()[1]):
                curr = decoded.data[i, t]
                if curr == eos:
                    break
                elif curr == 1:
                    out_line.append('_')
                else:
                    out_line.append(self.vocab.get_token_from_index(curr))
            out.append(' '.join(out_line))
        return out