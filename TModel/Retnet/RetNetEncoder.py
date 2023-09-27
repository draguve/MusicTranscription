from copy import deepcopy
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
from einops import rearrange
from torch import Tensor, nn

from yet_another_retnet.retention import (
    ActivationString,
    _get_activation_fn,
)

from TModel.Retnet.MultiHeadCrossRetention import MultiScaleCrossRetention


class RetNetEncoderLayer(nn.Module):
    def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            activation: Union[ActivationString, Callable[[Tensor], Tensor]] = "swish",
            norm_first: bool = True,
            layer_norm_eps: float = 1e-5,
            device: Optional[Union[torch.device, str]] = None,
            dtype: Optional[torch.dtype] = None,
    ) -> None:
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.norm_first = norm_first
        # retention block

        self.self_retention = MultiScaleCrossRetention(  # type: ignore
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            activation=activation,
            device=device,
            dtype=dtype,
        )
        self.norm1 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.norm2 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype
        )

        # feedforward block
        self.linear1 = nn.Linear(d_model, dim_feedforward, device=device, dtype=dtype)
        self.linear2 = nn.Linear(dim_feedforward, d_model, device=device, dtype=dtype)

        self._reset_parameters()

    def _reset_parameters(self):
        # TODO: Check that we're following the same initialization as the paper
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.xavier_normal_(self.linear2.weight)
        nn.init.constant_(self.linear2.bias, 0)

    def _feedforward_block(self, x: Tensor) -> Tensor:
        x = self.activation(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x

    def forward_parallel(self, x: Tensor) -> Tensor:
        def _retention_block(inputT: Tensor) -> Tensor:
            inputT, _ = self.self_retention.forward_parallel(inputT, inputT, inputT)
            return self.dropout(inputT)

        if self.norm_first:
            x = x + _retention_block(self.norm1(x))
            x = x + self._feedforward_block(self.norm2(x))
        else:
            x = x + self.norm1(_retention_block(x))
            x = x + self.norm2(self._feedforward_block(x))

        return x

    def forward_recurrent(
            self, x: Tensor, seq_idx: int, prev_state: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError()

    def forward_chunkwise(
            self, x: Tensor, start_idx: int, prev_state: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError()

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_parallel(x)


class RetNetEncoder(nn.Module):
    def __init__(self, encoder_layer: RetNetEncoderLayer, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [deepcopy(encoder_layer) for _ in range(num_layers)]
        )

    def forward_parallel(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            assert isinstance(layer, RetNetEncoderLayer)
            x = layer.forward_parallel(x)
        return x

    def forward_recurrent(
            self, x: Tensor, seq_idx: int, prev_states: Sequence[Optional[Tensor]] = ()
    ) -> Tuple[Tensor, List[Tensor]]:
        raise NotImplementedError()

    def forward_chunkwise(
            self, x: Tensor, start_idx: int, prev_states: Sequence[Optional[Tensor]] = ()
    ) -> Tuple[Tensor, List[Tensor]]:
        raise NotImplementedError()

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_parallel(x)
