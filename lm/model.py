
from torch.autograd import Variable
import torch

from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.models.model import Model

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

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

class LanguageModel(Model):
  def __init__(self, vocab, config=None):
    super().__init__(vocab)
    config = config or {}
    self.config = config
    emb_size = config.get('emb_size', 256)
    lstm_size = config.get('lstm_size', 512)
    lstm_layers = config.get('lstm_layers', 4)
    self.word_dropout = config.get('word_dropout', 0.05)
    self.emb = Embedding(
      vocab.get_vocab_size(),
      emb_size,
      padding_index=0
    )
    self.lstm = torch.nn.LSTM(
      emb_size,
      lstm_size,
      lstm_layers,
      dropout=config.get('lstm_interlayer_dropout', 0.1),
      batch_first=True
    )
    self.h0 = torch.nn.Parameter(torch.Tensor(lstm_layers, lstm_size).zero_())
    self.c0 = torch.nn.Parameter(torch.Tensor(lstm_layers, lstm_size).zero_())
    self.project = torch.nn.Linear(
      lstm_size,
      vocab.get_vocab_size()
    )

  def forward(self, input_array, init_states=None, unroll_length=None, teacher_forcing=0, disallow_unk=False):
    if next(self.parameters()).is_cuda:
      LongTensor = torch.cuda.LongTensor
      FloatTensor = torch.cuda.FloatTensor
    else:
      LongTensor = torch.LongTensor
      FloatTensor = torch.FloatTensor
    input_var = Variable(LongTensor(input_array), requires_grad=False, volatile=not self.training)
    if input_var.size(1) > 1:
      targets = input_var[:, 1:].contiguous()
      length = targets.size(1)
    else:
      targets = None
      length = unroll_length
    batch_size = input_var.size(0)
    if init_states is None:
      h = self.h0.unsqueeze(1).expand(self.h0.size(0), batch_size, self.h0.size(1)).contiguous()
      c = self.c0.unsqueeze(1).expand(self.c0.size(0), batch_size, self.c0.size(1)).contiguous()
    else:
      h, c = init_states
      reset_mask = (input_var[:, 0] == self.vocab.get_token_index('@@SOS@@')).float()
      reset_mask = reset_mask.view(1, batch_size, 1).expand(h.size(0), batch_size, h.size(2))
      h = h*(1.0 - reset_mask) + self.h0.unsqueeze(1).expand(self.h0.size(0), batch_size, self.h0.size(1))*reset_mask
      c = c*(1.0 - reset_mask) + self.c0.unsqueeze(1).expand(self.c0.size(0), batch_size, self.c0.size(1))*reset_mask
    if teacher_forcing < 1.0 or targets is None:
      # Need to forward each timestep's output to the next timestep's input
      input_indices = input_var[:, :1].contiguous()
      logit_slices = []
      index_slices = []
      for t in range(length):
        # Word dropout
        if self.word_dropout > 0:
            input_drop_mask = LongTensor(batch_size, 1)
            input_drop_mask.bernoulli_(self.word_dropout)
            input_drop_mask = Variable(input_drop_mask, requires_grad=False)
            input_indices = (input_indices * (1 - input_drop_mask)) + input_drop_mask

        embedded_inputs = self.emb(input_indices)
        out, (h, c) = self.lstm(embedded_inputs, (h, c))
        out = out[:, -1]
        logits = self.project(out)
        dist = torch.nn.functional.softmax(logits, dim=1)
        if disallow_unk:
          dist[:, 1] = 0
        indices = torch.multinomial(dist, 1, replacement=True).detach()
        logit_slices.append(logits)
        index_slices.append(indices)
        input_indices = indices

        # Teacher forcing
        if teacher_forcing > 0 and targets is not None:
            input_force_mask = LongTensor(batch_size, 1)
            input_force_mask.bernoulli_(teacher_forcing)
            input_force_mask = Variable(input_force_mask, requires_grad=False)
            gold_indices = targets[:, t].unsqueeze(1)
            input_indices = (input_indices * (1 - input_force_mask)) + (input_force_mask * gold_indices)
      all_logits = torch.stack(logit_slices, 1)
      all_indices = torch.cat(index_slices, 1)
    else:
      # Don't need to forward output to input
      # Word dropout
      if self.word_dropout > 0:
          input_drop_mask = LongTensor(batch_size, input_var.size(1))
          input_drop_mask.bernoulli_(self.word_dropout)
          input_drop_mask = Variable(input_drop_mask, requires_grad=False)
          input_var = (input_var * (1 - input_drop_mask)) + input_drop_mask

      embedded_inputs = self.emb(input_var[:, :-1])
      out, (h, c) = self.lstm(embedded_inputs, (h, c))
      all_logits = self.project(out.contiguous().view(batch_size * length, out.size(2))).view(batch_size, length, self.project.out_features)
      dist = torch.nn.functional.softmax(all_logits, dim=2)
      if disallow_unk:
        dist[:, :, 1] = 0
      all_indices = torch.multinomial(dist.view(batch_size * length, dist.size(2)), 1, replacement=True).view(batch_size, length).detach()

    output_dict = {
      'logits': all_logits,
      'indices': all_indices,
      'final_state': (h.detach(), c.detach())
    }
    if targets is not None:
      target_mask = (targets != 0).float()
      loss = sequence_cross_entropy_with_logits(
        all_logits,
        targets.contiguous(),
        target_mask
      )
      output_dict['loss'] = loss
      output_dict['accuracy'] = ((all_indices == targets).float() * target_mask).sum(1) / torch.min(target_mask.sum(1), Variable(FloatTensor(batch_size).fill_(1), requires_grad=False))
    else:
      sep = ' ' if self.config.get('word_level', True) else ''
      output_dict['text'] = [sep.join(self.vocab.get_token_from_index(i) for i in all_indices.data[j].tolist()) for j in range(batch_size)]
    return output_dict
