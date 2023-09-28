import math
from typing import Optional, Union

import torch
import torch.nn as nn
from einops import rearrange, einsum
from torch import Tensor

from Externals.RetNet.src.xpos_relative_position import XPOS

def _build_decay_gammas(
        num_heads: int,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Decay values are different for each retention head, following the prescribed
    method in the paper.  Conceptually, I think of each head having a different
    "retention window", which is the effective number of steps back in time that
    the head can attend to.  Retention windows are effectively determined by
    these decay coefficients.

    See: https://arxiv.org/pdf/2307.08621v3.pdf, Section 3.1 (Setup)
    """
    xmin, xmax = math.log(1 / 32), math.log(1 / 512)
    x = torch.linspace(xmin, xmax, steps=num_heads, device=device, dtype=dtype)
    return 1 - torch.exp(x)


def _build_decay_mask(
        query_length: int,
        key_length: int,
        decay_gammas: Tensor,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """The decay mask is one of the key components that makes *parallel* retention
    equivalent to *recurrent* retention.  The decay coefficients are pre-computed
    and applied to the similarity matrix at once, rather than being applied to
    each element in the recurrent formulation.

    See: https://arxiv.org/pdf/2307.08621v3.pdf, Equation 5
    """
    query_pos = torch.arange(query_length, device=device, dtype=dtype)
    key_pos = torch.arange(key_length, device=device, dtype=dtype)

    distance = torch.abs(query_pos.unsqueeze(-1) - key_pos.unsqueeze(0))
    # Set the upper-triangular distances to infinity, so that only *past* keys
    # can affect the current query.  (Setting distance to infinity ensures that
    # the decay matrix is 0 for those positions, since x^(inf) = 0 when -1 < x < 1.
    distance_mask = torch.ones_like(distance, dtype=torch.bool).triu_(diagonal=1)
    distance = distance.masked_fill(distance_mask, float("inf"))

    distance = rearrange(distance, "n s -> () n s")
    decay_gammas = rearrange(decay_gammas, "h -> h () ()")
    return decay_gammas ** distance


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
        self.head_v_dim = self.head_size * 2 if double_v_dim else self.head_size

        self.xpos = XPOS(self.head_size)

        self.swish = nn.SiLU()

        self.W_Q = nn.Parameter(torch.randn(self.heads, hidden_size, self.head_size) / hidden_size)
        self.W_K = nn.Parameter(torch.randn(self.heads, hidden_size, self.head_size) / hidden_size)
        self.W_V = nn.Parameter(torch.randn(self.heads, hidden_size, self.head_v_dim) / hidden_size)

        self.gammas = _build_decay_gammas(self.heads, self.W_Q.device, self.W_Q.dtype)

        self.W_G = nn.Parameter(torch.randn(hidden_size, self.v_dim) / hidden_size)
        self.W_O = nn.Parameter(torch.randn(self.v_dim, hidden_size) / hidden_size)
        self.group_norm = nn.GroupNorm(heads, self.v_dim)


    def forward(self, X, Mem):
        """
        parallel representation of the multi-scale retention mechanism
        b n l dim
        """
        # b=batch l=sq_len d=hiddendim n=n-head h=headdim v=vdim
        batch, sq_len, _ = X.shape
        _, key_len, _ = Mem.shape
        q_proj = einsum(X, self.W_Q, "b l d, n d h -> b n l h")  # checked this multiply
        k_proj = einsum(Mem, self.W_K, "b l d, n d h -> b n l h")
        v_proj = einsum(Mem, self.W_V, "b l d, n d v -> b n l v")

        # to check the multiply up top
        # Q = []
        # for i in range(self.heads):
        #     Q.append(X @ self.W_Q[i, :, :])
        # Q = torch.stack(Q)
        # torch.all(rearrange(Q,"n b l h -> b n l h") == einsum(X, self.W_Q, "b l d, n d h -> b n l h"))

        q_proj = rearrange(q_proj, "b n l h -> (b n) l h")
        k_proj = rearrange(k_proj, "b n l h -> (b n) l h")
        q_proj = self.xpos(q_proj)
        k_proj = self.xpos(k_proj, downscale=True)
        decay = _build_decay_mask(sq_len, key_len, self.gammas, q_proj.device, dtype=q_proj.dtype).unsqueeze(0)
        ret = q_proj @ k_proj.permute(0, 2, 1)
        ret = rearrange(ret, "(b n) l h -> b n l h", b=batch)
        ret = (ret * decay) @ v_proj
        ret = rearrange(ret, "b n l h -> b l (n h)")
        # apply each individual retention mechanism to X
        ret_shape = ret.shape
        ret = self.group_norm(ret.reshape(-1, self.v_dim)).reshape(ret_shape)

        return (self.swish(X @ self.W_G) * ret) @ self.W_O

    def forward_recurrent(self, x_n, s_n_1s, n):
        raise NotImplemented()

    def forward_chunkwise(self, x_i, r_i_1s, i):
        raise NotImplemented()


def test():
    X = torch.rand((2, 30, 512), dtype=torch.float)
    Y = torch.rand((2,40,512),dtype=torch.float)
    msdr = MultiScaleDecoderRetention(512, 8, False)
    msdr.forward(X, Y)


if __name__ == '__main__':
    test()
