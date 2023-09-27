import math
from Externals.RetNet.src.xpos_relative_position import XPOS
from math import log
from typing import Callable, Literal, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from torch import Tensor, nn


class SimpleDecoderRetention(nn.Module):
    def __init__(self, hidden_size, gamma, head_size=None, double_v_dim=False):
        super(SimpleDecoderRetention, self).__init__()

        self.hidden_size = hidden_size
        if head_size is None:
            head_size = hidden_size
        self.head_size = head_size

        self.v_dim = head_size * 2 if double_v_dim else head_size
        # self.gamma = gamma

        self.W_Q = nn.Parameter(torch.randn(hidden_size, head_size) / hidden_size)
        self.W_K = nn.Parameter(torch.randn(hidden_size, head_size) / hidden_size)
        self.W_V = nn.Parameter(torch.randn(hidden_size, self.v_dim) / hidden_size)
        self.gamma = torch.tensor(gamma).to(self.W_V.device)
        self.xpos = XPOS(head_size)

    def forward(self, X, Mem):
        """
        Parallel (default) representation of the retention mechanism.
        X: (batch_size, sequence_length, hidden_size)
        """
        # Decoder has the key and value from the ouput of the encoder
        sequence_length = X.shape[1]
        sequence_length2 = Mem.shape[1]
        D = self._get_D(sequence_length, sequence_length2, X.device)
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

    def _get_D(self, sequence_length1, sequence_length2, device):
        n = torch.arange(sequence_length1, device=device).unsqueeze(1)
        m = torch.arange(sequence_length2, device=device).unsqueeze(0)

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


DEFAULT_DEVICE = torch.device("cpu")
ActivationString = Literal["swish", "gelu", "relu"]


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    """Return an activation function given a string"""
    if activation == "swish":
        return F.silu
    elif activation == "gelu":
        return F.gelu
    elif activation == "relu":
        return F.relu
    else:
        raise RuntimeError(
            f"Unsupported activation string '{activation}'. "
            "Supported: 'swish', 'gelu', 'relu'"
        )


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
    xmin, xmax = log(1 / 32), log(1 / 512)
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


# def _build_chunkwise_decay_mask(
#     chunk_size: int,
#     decay_gammas: Tensor,
#     device: Optional[Union[torch.device, str]] = None,
#     dtype: Optional[torch.dtype] = None,
# ) -> Tensor:
#     """The chunkwise decay mask is used in the chunkwise formulation of retention.
#     It is derived from the recurrent formulation -- we apply increasing amounts
#     of decay to the previous state, based on the position of each token in the
#     input chunk.  So, we add a 'sequence' dimension to the decay gammas, and
#     raise them to the power of the position of each token in the chunk.

#     See: https://arxiv.org/pdf/2307.08621v3.pdf, Equation 7
#     """
#     chunk_pos = torch.arange(chunk_size, device=device, dtype=dtype)
#     chunk_pos = rearrange(chunk_pos, "n -> () n")
#     decay_gammas = rearrange(decay_gammas, "h -> h ()")
#     return decay_gammas**chunk_pos


def _build_position_thetas(
        head_dim: int,
        scale: float = 10000,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Positional thetas are different for each value along head_dim, following the
    prescribed method in the paper.  These are used to update the positional
    embeddings in both the parallel and recurrent formulations of retention.
    See: https://arxiv.org/pdf/2307.08621v3.pdf, Section 2.1 (Retention)

    NOTE: The actual values for thetas are not specified in the paper, so I
    copied these values from the official implementation.
    See: https://github.com/microsoft/torchscale/blob/7d231743f4f96c460b7cf0aa0cf242bb192b34f8/torchscale/architecture/retnet.py#L27C1-L28C59
    """
    x = torch.linspace(0, 1, steps=head_dim // 2, device=device, dtype=dtype)
    thetas = 1 / (scale ** x)
    return repeat(thetas, "d -> (d n)", n=2)


# NOTE: For the purposes of positional embeddings, we view query/key Tensors as
# complex-valued, where the even-numbered indices are the real part, and the
# odd-numbered indices are the imaginary part.  This makes it easy to compute
# complex values without *actually* using complex dtypes in PyTorch.
# (Complex dtypes have limited support compared to real dtypes.)
#
# I don't re-explain this in the functions below, but it's important to keep in
# mind when reading the code.


def _multiply_by_i(x: Tensor) -> Tensor:
    """Multiply a complex-valued tensor by the imaginary unit 'i'."""
    return torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(start_dim=-2)


def _theta_shift(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    # TODO: Add docstring
    return (x * cos) + (_multiply_by_i(x) * sin)


def retention_parallel(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        scale: Optional[float] = None,
        decay_gammas: Optional[Tensor] = None,
        need_weights: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    if decay_gammas is None:
        decay_gammas = _build_decay_gammas(
            num_heads=query.shape[1], device=query.device, dtype=query.dtype
        )
    decay_mask = _build_decay_mask(
        query_length=query.shape[2],
        key_length=key.shape[2],
        decay_gammas=decay_gammas,
        device=query.device,
        dtype=query.dtype,
    )

    # einstein notation:
    # - b: batch_size
    # - h: num_heads
    # - n / s: seq_length
    # - d: hidden_dim
    if scale is None:
        scale = key.size(-1) ** 0.5
    key = key / scale

    similarity = einsum(query, key, "b h n d, b h s d -> b h n s")
    similarity = similarity * rearrange(decay_mask, "h n s -> () h n s")
    retention = einsum(similarity, value, "b h n s, b h s d -> b h n d")

    if need_weights:
        return retention, similarity
    else:
        return retention, None


class MultiScaleRetention(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            relative_position: bool = True,
            bias: bool = True,
            batch_first: bool = True,
            activation: Union[ActivationString, Callable[[Tensor], Tensor]] = "swish",
            group_norm_eps: float = 1e-6,
            device: Optional[Union[torch.device, str]] = None,
            dtype: Optional[torch.dtype] = None,
            # TODO???
            # add_bias_kv=False,
            # add_zero_attn=False,
            # kdim=None,
            # vdim=None,
    ):
        if not batch_first:
            raise NotImplementedError("batch_first=False is not yet supported")
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.relative_position = relative_position
        self.bias = bias
        self.activation = activation

        if embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        head_dim = embed_dim // num_heads
        if not head_dim % 8 == 0:
            raise ValueError(
                f"head_dim (embed_dim / num_heads = {head_dim}) must be divisible by 8"
            )

        # The q/k/v projection layers are the same as in vanilla MHA.
        self.q_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )
        self.k_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )
        self.v_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )
        self.group_norm = nn.GroupNorm(
            num_groups=num_heads,
            num_channels=num_heads,
            affine=False,
            eps=group_norm_eps,
            device=device,
            dtype=dtype,
        )
        # The output project is slightly different, due to the gated "swish" layer.
        self.g_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )
        self.out_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )
        self.xpos = XPOS(head_dim)
        # 'thetas' parameter for updating the relative position embeddings.
        thetas: Optional[Tensor] = None
        if relative_position:
            thetas = _build_position_thetas(
                head_dim=head_dim, device=device, dtype=dtype
            )
        self.thetas: Optional[Tensor]
        self.register_buffer("thetas", thetas)

        self._reset_parameters()

    def _reset_parameters(self):
        # TODO: Double-check that we're following the same initialization as in
        # the paper.  This is a generic initialization for MHA linear layers.
        nn.init.xavier_normal_(self.q_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0)
        nn.init.xavier_normal_(self.k_proj.weight)
        if self.k_proj.bias is not None:
            nn.init.constant_(self.k_proj.bias, 0)
        nn.init.xavier_normal_(self.v_proj.weight)
        if self.v_proj.bias is not None:
            nn.init.constant_(self.v_proj.bias, 0)
        nn.init.xavier_normal_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0)
        nn.init.xavier_normal_(self.g_proj.weight)
        if self.g_proj.bias is not None:
            nn.init.constant_(self.g_proj.bias, 0)

    def forward_parallel(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            need_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # einstein notation:
        # b - batch size
        # n - sequence length
        # h - number of heads
        # d - embedding dimension
        #
        # Input shape: (b, n, d)
        q: Tensor = self.q_proj(query)
        k: Tensor = self.k_proj(key)
        v: Tensor = self.v_proj(value)

        # Unfold 'd' dimension into 'h' separate retention heads.  Move the head
        # dimension to position 1 (makes matrix ops *much* faster).
        q = rearrange(q, "b n (h d) -> (b h) n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> (b h) n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

        if self.relative_position:
            q = self.xpos(q)
            k = self.xpos(k, downscale=True)

        q = rearrange(q, "(b h) n d -> b h n d", h=self.num_heads)
        k = rearrange(k, "(b h) n d -> b h n d", h=self.num_heads)

        # Apply retention then group norm.
        retention, weights = retention_parallel(q, k, v, need_weights=need_weights)
        # To apply group norm in an equivalent way to the recurrent formulation,
        # we fold the sequence dimension into the batch dimension.  Otherwise,
        # normalization would be applied over the entire input sequence.
        batch_size = retention.size(0)
        retention = rearrange(retention, "b h n d -> (b n) h d")
        retention = F.dropout(retention, p=self.dropout, training=self.training)
        retention = self.group_norm(retention)
        # Unfold 'n' from the batch dimension, and fold 'h' back into the embed dim.
        retention = rearrange(retention, "(b n) h d -> b n (h d)", b=batch_size)

        # NOTE: Unlike multihead attention, the retention paper applies a "swish"
        # gate to increase the non-linear capacity of the model.  (IMO this is likely
        # to make up for the lack of "softmax" activation in the retention mechanism.)
        #
        # The paper describes the gate as:
        #   g = swish(X * W_g)
        # where X is the input to the layer.  The authors use Retention in a
        # Decoder-only model, the q/k/v inputs are the same (i.e. X = q = k = v).
        # So, I assume that 'query' can equivalently be used as the input.
        gate = self.activation(self.g_proj(query))
        retention = self.out_proj(retention * gate)

        return retention, weights

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            need_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        return self.forward_parallel(query, key, value, need_weights=need_weights)
