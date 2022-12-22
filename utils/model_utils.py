import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np


class RNNEncoder(nn.Module):
    """A RNN wrapper handles variable length inputs, always set batch_first=True.
    Supports LSTM, GRU and RNN. Tested with PyTorch 0.3 and 0.4
    """
    def __init__(self, word_embedding_size, hidden_size, bidirectional=True,
                 dropout_p=0, n_layers=1, rnn_type="lstm",
                 return_hidden=True, return_outputs=True,
                 allow_zero=False):
        super(RNNEncoder, self).__init__()
        """  
        :param word_embedding_size: rnn input size
        :param hidden_size: rnn output size
        :param dropout_p: between rnn layers, only useful when n_layer >= 2
        """
        self.allow_zero = allow_zero
        self.rnn_type = rnn_type
        self.n_dirs = 2 if bidirectional else 1
        # - add return_hidden keyword arg to reduce computation if hidden is not needed.
        self.return_hidden = return_hidden
        self.return_outputs = return_outputs
        self.rnn = getattr(nn, rnn_type.upper())(word_embedding_size, hidden_size, n_layers,
                                                 batch_first=True,
                                                 bidirectional=bidirectional,
                                                 dropout=dropout_p)

    def sort_batch(self, seq, lengths):
        sorted_lengths, perm_idx = lengths.sort(0, descending=True)
        if self.allow_zero:  # deal with zero by change it to one.
            sorted_lengths[sorted_lengths == 0] = 1
        reverse_indices = [0] * len(perm_idx)
        for i in range(len(perm_idx)):
            reverse_indices[perm_idx[i]] = i
        sorted_seq = seq[perm_idx]
        return sorted_seq, list(sorted_lengths), reverse_indices

    def forward(self, inputs, lengths):
        """
        inputs, sorted_inputs -> (B, T, D)
        lengths -> (B, )
        outputs -> (B, T, n_dirs * D)
        hidden -> (n_layers * n_dirs, B, D) -> (B, n_dirs * D)  keep the last layer
        - add total_length in pad_packed_sequence for compatiblity with nn.DataParallel, --remove it
        """
        assert len(inputs) == len(lengths)
        sorted_inputs, sorted_lengths, reverse_indices = self.sort_batch(inputs, lengths)
        packed_inputs = pack_padded_sequence(sorted_inputs, sorted_lengths, batch_first=True)
        outputs, hidden = self.rnn(packed_inputs)
        if self.return_outputs:
            # outputs, lengths = pad_packed_sequence(outputs, batch_first=True, total_length=int(max(lengths)))
            outputs, lengths = pad_packed_sequence(outputs, batch_first=True)
            outputs = outputs[reverse_indices]
        else:
            outputs = None
        if self.return_hidden:  #
            if self.rnn_type.lower() == "lstm":
                hidden = hidden[0]
            hidden = hidden[-self.n_dirs:, :, :]
            hidden = hidden.transpose(0, 1).contiguous()
            hidden = hidden.view(hidden.size(0), -1)
            hidden = hidden[reverse_indices]
        else:
            hidden = None
        return outputs, hidden


def pool_across_time(outputs, lengths, pool_type="max"):
    """ Get maximum responses from RNN outputs along time axis
    :param outputs: (B, T, D)
    :param lengths: (B, )
    :param pool_type: str, 'max' or 'mean'
    :return: (B, D)
    """
    if pool_type == "max":
        outputs = [outputs[i, :int(lengths[i]), :].max(dim=0)[0] for i in range(len(lengths))]
    elif pool_type == "mean":
        outputs = [outputs[i, :int(lengths[i]), :].mean(dim=0) for i in range(len(lengths))]
    else:
        raise NotImplementedError("Only support mean and max pooling")
    return torch.stack(outputs, dim=0)


def count_parameters(model, verbose=True):
    """Count number of parameters in PyTorch model,
    References: https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7.

    from utils.utils import count_parameters
    count_parameters(model)
    import sys
    sys.exit(1)
    """
    n_all = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print("Parameter Count: all {:,d}; trainable {:,d}".format(n_all, n_trainable))
    return n_all, n_trainable


def generate_gauss_weight(props_len, center, width, sigma=9):
    # pdb.set_trace()
    weight = torch.linspace(0, 1, props_len)
    weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)
    center = center.unsqueeze(-1)
    width = width.unsqueeze(-1).clamp(1e-2) / sigma

    w = 0.3989422804014327
    weight = w/width*torch.exp(-(weight-center)**2/(2*width**2))

    return weight/weight.max(dim=-1, keepdim=True)[0] # (center.shape[0], props_len)

def get_gauss_props_from_clip_indices(indices, num_gauss_center, num_gauss_width):
    row, col = indices.shape
    centers = torch.linspace(0, 1, steps=num_gauss_center)
    widthes = torch.linspace(0.05, 1.0, steps=num_gauss_width)
    gauss_center = torch.zeros_like(indices).type(torch.float32)
    gauss_width = torch.zeros_like(indices).type(torch.float32)
    for i in range(row):
        for j in range(col):
            idx = indices[i, j].item()
            center_idx, width_idx = np.unravel_index(idx, shape=(num_gauss_center, num_gauss_width))
            gauss_center[i, j], gauss_width[i, j] = centers[center_idx], widthes[width_idx]
    
    return gauss_center, gauss_width
