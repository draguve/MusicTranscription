from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from yet_another_retnet.retention import MultiScaleRetention, _theta_shift, retention_parallel


def get_sin_cos(q, thetas):
    indices = torch.arange(q.size(2), device=q.device, dtype=q.dtype)
    indices = rearrange(indices, "n -> () () n ()")
    thetas = rearrange(thetas, "d -> () () () d")
    angles = indices * thetas
    sin = torch.sin(angles)
    cos = torch.cos(angles)
    return sin, cos


class MultiScaleCrossRetention(MultiScaleRetention):
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
