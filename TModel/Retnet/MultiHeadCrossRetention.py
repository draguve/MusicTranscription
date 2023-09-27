from typing import Optional, Tuple, Union, Callable
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn
from yet_another_retnet.retention import MultiScaleRetention, _theta_shift, retention_parallel, ActivationString, \
    _get_activation_fn, _build_position_thetas, retention_recurrent, retention_chunkwise


def get_sin_cos(q, thetas):
    indices = torch.arange(q.size(2), device=q.device, dtype=q.dtype)
    indices = rearrange(indices, "n -> () () n ()")
    thetas = rearrange(thetas, "d -> () () () d")
    angles = indices * thetas
    sin = torch.sin(angles)
    cos = torch.cos(angles)
    return sin, cos


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
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

        if self.relative_position:
            assert self.thetas is not None
            q_sin, q_cos = get_sin_cos(q, self.thetas)
            k_sin, k_cos = get_sin_cos(k, self.thetas)
            q = _theta_shift(q, q_sin, q_cos)
            k = _theta_shift(k, k_sin, k_cos)

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
