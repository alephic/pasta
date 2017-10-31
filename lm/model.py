
from torch.autograd import Variable
import torch

from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.models.model import Model

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from highway_lstm.highway_lstm_layer import HighwayLSTMLayer

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

class LanguageModel(Model):
  def __init__(self, vocab, config=None):
    super().__init__(vocab)
    config = config or {}
    emb_size = config.get('emb_size', 256)
    lstm_size = config.get('lstm_size', 512)
    lstm_layers = config.get('lstm_layers', 4)
    self.emb = Embedding(
      vocab.get_vocab_size()
      emb_size,
      padding_index=0
    )
    self.lstm = torch.nn.LSTM(
      emb_size,
      lstm_size,
      lstm_layers,
      batch_first=True
    )
    self.h0 = torch.nn.Parameter(torch.Tensor(lstm_layers, lstm_size))
    self.c0 = torch.nn.Parameter(torch.Tensor(lstm_layers, lstm_size))
    self.project = torch.nn.Linear(
      lstm_size,
      vocab.get_vocab_size()
    )
    self.cuda()

  def forward(self, input_array, unroll_length=200):
    input_var = Variable(torch.cuda.LongTensor(input_array), requires_grad=False, volatile=not self.training)
    targets = input_var[:, 1:]
    length = targets.size(1) if self.training else unroll_length
    batch_size = input_var.size(0)
    h = self.h0.unsqueeze(0).expand(batch_size, -1, -1)
    c = self.c0.unsqueeze(0).expand(batch_size, -1, -1)
    input_indices = input_var[:, :1]
    logit_slices = []
    index_slices = []
    for t in range(length):
      embedded_inputs = self.emb(input_indices)
      out, (h, c) = self.lstm(embedded_inputs, h, c)
      out = out[-1]
      logits = self.project(out)
      dist = torch.nn.functional.softmax(logits)
      indices = torch.multinomial(dist, 1, replacement=True)
      logit_slices.append(logits)
      index_slices.append(indices)
    all_logits = torch.stack(logit_slices, 1)
    all_indices = torch.cat(index_slices, 1)
    loss = sequence_cross_entropy_with_logits(
      all_logits,
      targets,
      Variable(torch.cuda.FloatTensor(*targets.size()).fill_(1), requires_grad=False)
    )
    return {
      'logits': all_logits,
      'indices': all_indices,
      'text': [''.join(self.vocab.get_token_from_index(i) for i in all_indices.data[j].tolist()) for j in range(batch_size)],
      'loss': loss
    }
