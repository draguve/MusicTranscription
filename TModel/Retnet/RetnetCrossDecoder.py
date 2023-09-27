from copy import deepcopy
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
from einops import rearrange
from torch import Tensor, nn

from yet_another_retnet.retention import (
    ActivationString,
    MultiScaleRetention,
    _get_activation_fn,
)

from TModel.Retnet.MultiHeadCrossRetention import MultiScaleCrossRetention


class RetNetCrossLayer(nn.Module):
    # NOTE: Mostly pulled from 'nn.TransformerDecoderLayer', but with changes:
    #   - use MultiScaleRetention instead of MultiheadAttention
    #   - no cross-attention layer, since retention doesn't play well with that

    def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            activation: Union[ActivationString, Callable[[Tensor], Tensor]] = "swish",
            norm_first: bool = True,
            layer_norm_eps: float = 1e-6,
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
        self.norm1 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.self_retention = MultiScaleRetention(  # type: ignore
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            activation=activation,
            device=device,
            dtype=dtype,
        )
        self.cross_retention = MultiScaleCrossRetention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            activation=activation,
            device=device,
            dtype=dtype,
        )
        # feedforward block
        self.norm2 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.norm3 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype
        )
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

    def forward_parallel(self, x: Tensor, mem: Tensor) -> Tensor:
        def _retention_block(x: Tensor) -> Tensor:
            x, _ = self.self_retention.forward_parallel(x, x, x)
            return self.dropout(x)

        def _cross_retention_block(x: Tensor, mem: Tensor):
            x, _ = self.cross_retention.forward_parallel(x, mem, mem)
            return self.dropout(x)

        if self.norm_first:
            x = x + _retention_block(self.norm1(x))
            x = x + _cross_retention_block(self.norm2(x), mem)
            x = x + self._feedforward_block(self.norm3(x))
        else:
            x = x + self.norm1(_retention_block(x))
            x = x + self.norm2(_cross_retention_block(x, mem))
            x = x + self.norm3(self._feedforward_block(x))

        return x

    def forward_recurrent(
            self, x: Tensor, seq_idx: int, prev_state: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError()

    def forward_chunkwise(
            self, x: Tensor, start_idx: int, prev_state: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError()

    def forward(self, x: Tensor, mem: Tensor) -> Tensor:
        return self.forward_parallel(x, mem)


class RetNetCrossDecoder(nn.Module):
    def __init__(self, decoder_layer: RetNetCrossLayer, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [deepcopy(decoder_layer) for _ in range(num_layers)]
        )

    def forward_parallel(self, x: Tensor, mem: Tensor) -> Tensor:
        for layer in self.layers:
            assert isinstance(layer, RetNetCrossLayer)
            x = layer.forward_parallel(x, mem)
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


class RetNetEncoderDecoder(nn.Module):
    def __init__(
            self,
            num_tokens: int,  # usually obtained from the tokenizer
            d_model: int = 512,
            nhead: int = 8,
            num_layers: int = 6,
            dropout: float = 0.1,
            activation: Union[ActivationString, Callable[[Tensor], Tensor]] = "swish",
            dim_feedforward: int = 2048,
            norm_first: bool = True,
            layer_norm_eps: float = 1e-6,
            device: Optional[Union[torch.device, str]] = None,
            dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = nn.Embedding(num_tokens, d_model, device=device, dtype=dtype)
        decoder_layer = RetNetCrossLayer(
            d_model,
            nhead,
            dropout=dropout,
            activation=activation,
            dim_feedforward=dim_feedforward,
            norm_first=norm_first,
            layer_norm_eps=layer_norm_eps,
            device=device,
            dtype=dtype,
        )
        self.decoder = RetNetCrossDecoder(decoder_layer, num_layers)
        self.out = nn.Linear(d_model, num_tokens, device=device, dtype=dtype)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_normal_(self.out.weight)
        nn.init.constant_(self.out.bias, 0)

    def forward_parallel(self, x: Tensor) -> Tensor:
        x = self.embedding(x)
        x = self.decoder.forward_parallel(x)
        x = self.out(x)
        return x

    def forward_recurrent(
            self, x: Tensor, seq_idx: int, prev_states: Sequence[Optional[Tensor]] = ()
    ) -> Tuple[Tensor, List[Tensor]]:
        x = self.embedding(x)
        x, states = self.decoder.forward_recurrent(
            x, seq_idx=seq_idx, prev_states=prev_states
        )
        x = self.out(x)
        return x, states

    def forward_chunkwise(
            self, x: Tensor, start_idx: int, prev_states: Sequence[Optional[Tensor]] = ()
    ) -> Tuple[Tensor, List[Tensor]]:
        x = self.embedding(x)
        x, states = self.decoder.forward_chunkwise(
            x, start_idx=start_idx, prev_states=prev_states
        )
        x = self.out(x)
        return x, states

    def forward(self, inputs: Tensor, labels: Tensor) -> Tensor:
        pred = self.forward_parallel(inputs)
        criterion = nn.CrossEntropyLoss()
        return criterion(rearrange(pred, "b n c -> (b n) c"), labels.flatten())
