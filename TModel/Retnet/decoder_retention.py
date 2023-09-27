import math

import torch
import torch.nn as nn

from Externals.RetNet.src.xpos_relative_position import XPOS


class SimpleDecoderRetention(nn.Module):
    def __init__(self, hidden_size, gamma, head_size=None, double_v_dim=False):
        super(SimpleDecoderRetention, self).__init__()

        self.hidden_size = hidden_size
        if head_size is None:
            head_size = hidden_size
        self.head_size = head_size

        self.v_dim = head_size * 2 if double_v_dim else head_size
        self.gamma = gamma

        self.W_Q = nn.Parameter(torch.randn(hidden_size, head_size) / hidden_size)
        self.W_K = nn.Parameter(torch.randn(hidden_size, head_size) / hidden_size)
        self.W_V = nn.Parameter(torch.randn(hidden_size, self.v_dim) / hidden_size)

        self.xpos = XPOS(head_size)

    def forward(self, X, Mem):
        """
        Parallel (default) representation of the retention mechanism.
        X: (batch_size, sequence_length, hidden_size)
        """
        # Decoder has the key and value from the ouput of the encoder
        sequence_length = X.shape[1]
        sequence_length2 = Mem.shape[1]
        D = self._get_D(sequence_length, sequence_length2).to(self.W_Q.device)

        Q = (X @ self.W_Q)
        K = (Mem @ self.W_K)

        Q = self.xpos(Q)
        K = self.xpos(K, downscale=True)

        V = Mem @ self.W_V
        ret = (Q @ K.permute(0, 2, 1)) * D.unsqueeze(0)

        return ret @ V

    def forward_recurrent(self, x_n, s_n_1, n):
        raise NotImplemented()

    def forward_chunkwise(self, x_i, r_i_1, i):
        raise NotImplemented()

    def _get_D(self, sequence_length1, sequence_length2):
        n = torch.arange(sequence_length1).unsqueeze(1)
        m = torch.arange(sequence_length2).unsqueeze(0)

        # Broadcast self.gamma ** (n - m) with appropriate masking to set values where n < m to 0
        D = (self.gamma ** (n - m)) * (n >= m).float()  # this results in some NaN when n is much larger than m
        # fill the NaN with 0
        D[D != D] = 0

        return D


class MultiScaleDecoderRetention(nn.Module):
    def __init__(self, hidden_size, heads, double_v_dim=False):
        """
        Multi-scale retention mechanism based on the paper
        "Retentive Network: A Successor to Transformer for Large Language Models"[https://arxiv.org/pdf/2307.08621.pdf]
        """
        super(MultiScaleDecoderRetention, self).__init__()
        self.hidden_size = hidden_size
        self.v_dim = hidden_size * 2 if double_v_dim else hidden_size
        self.heads = heads
        assert hidden_size % heads == 0, "hidden_size must be divisible by heads"
        self.head_size = hidden_size // heads
        self.head_v_dim = hidden_size * 2 if double_v_dim else hidden_size

        self.gammas = (
                1 - torch.exp(torch.linspace(math.log(1 / 32), math.log(1 / 512), heads))).detach().cpu().tolist()

        self.swish = lambda x: x * torch.sigmoid(x)
        self.W_G = nn.Parameter(torch.randn(hidden_size, self.v_dim) / hidden_size)
        self.W_O = nn.Parameter(torch.randn(self.v_dim, hidden_size) / hidden_size)
        self.group_norm = nn.GroupNorm(heads, self.v_dim)

        self.retentions = nn.ModuleList([
            SimpleDecoderRetention(self.hidden_size, gamma, self.head_size, double_v_dim) for gamma in self.gammas
        ])

    def forward(self, X, Mem):
        """
        parallel representation of the multi-scale retention mechanism
        """

        # apply each individual retention mechanism to X
        Y = []
        for retention in self.retentions:
            Y.append(retention(X, Mem))

        Y = torch.cat(Y, dim=2)
        Y_shape = Y.shape
        Y = self.group_norm(Y.reshape(-1, self.v_dim)).reshape(Y_shape)

        return (self.swish(X @ self.W_G) * Y) @ self.W_O

    def forward_recurrent(self, x_n, s_n_1s, n):
        raise NotImplemented()

    def forward_chunkwise(self, x_i, r_i_1s, i):
        raise NotImplemented()
