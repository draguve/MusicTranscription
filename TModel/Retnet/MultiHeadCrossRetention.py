from typing import Optional, Tuple, Union, Callable
import torch
import torch.nn.functional as F
from einops import rearrange, einsum
from torch import Tensor, nn
from yet_another_retnet.retention import _build_decay_gammas, _build_decay_mask, ActivationString, \
    _get_activation_fn, _build_position_thetas

from Externals.RetNet.src.xpos_relative_position import XPOS


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

    # if scale is None:
    #     scale = key.size(-1) ** 0.5
    # key = key / scale

    similarity = einsum(query, key, "b h n d, b h s d -> b h n s")
    similarity = similarity * rearrange(decay_mask, "h n s -> () h n s")
    retention = einsum(similarity, value, "b h n s, b h s d -> b h n d")

    if need_weights:
        return retention, similarity
    else:
        return retention, None


class MultiScaleCrossRetention(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            relative_position: bool = True,
            bias: bool = True,
            batch_first: bool = True,
            activation: Union[ActivationString, Callable[[Tensor], Tensor]] = "swish",
            group_norm_eps: float = 1e-5,  # TODO check this?
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
        self.W_Q = nn.Parameter(torch.randn(self.num_heads, embed_dim, head_dim) / embed_dim)
        self.W_K = nn.Parameter(torch.randn(self.num_heads, embed_dim, head_dim) / embed_dim)
        self.W_V = nn.Parameter(torch.randn(self.num_heads, embed_dim, head_dim) / embed_dim)
        self.group_norm = nn.GroupNorm(
            num_groups=num_heads,
            num_channels=embed_dim,
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
        self._reset_parameters()

    def _reset_parameters(self):
        # TODO: Double-check that we're following the same initialization as in
        # the paper.  This is a generic initialization for MHA linear layers.
        # nn.init.xavier_normal_(self.q_proj.weight)
        # if self.q_proj.bias is not None:
        #     nn.init.constant_(self.q_proj.bias, 0)
        # nn.init.xavier_normal_(self.k_proj.weight)
        # if self.k_proj.bias is not None:
        #     nn.init.constant_(self.k_proj.bias, 0)
        # nn.init.xavier_normal_(self.v_proj.weight)
        # if self.v_proj.bias is not None:
        #     nn.init.constant_(self.v_proj.bias, 0)
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
        q = einsum(query, self.W_Q, "b l d, n d h -> b n l h")
        k = einsum(key, self.W_K, "b l d, n d h -> b n l h")
        v = einsum(value, self.W_V, "b l d, n d v -> b n l v")

        q = rearrange(q,"b h n d -> (b h) n d")
        k = rearrange(k,"b h n d -> (b h) n d")
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
        retention = rearrange(retention, "b h n d -> (b n) (h d)")
        retention = F.dropout(retention, p=self.dropout, training=self.training)
        retention = self.group_norm(retention)
        # Unfold 'n' from the batch dimension, and fold 'h' back into the embed dim.
        retention = rearrange(retention, "(b n) e -> b n e", b=batch_size)

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

    def forward_recurrent(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            seq_idx: int,
            prev_state: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError()

    def forward_chunkwise(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            start_idx: int,
            prev_state: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError()

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            need_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        return self.forward_parallel(query, key, value, need_weights=need_weights)
