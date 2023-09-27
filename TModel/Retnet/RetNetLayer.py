from torch import nn
from copy import deepcopy
from Externals.RetNet.src.retention import MultiScaleRetention
from TModel.Retnet.decoder_retention import MultiScaleDecoderRetention


class RetNetEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, ffn_size, heads, double_v_dim):
        super(RetNetEncoderLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.ffn_size = ffn_size
        self.heads = heads
        self.retention = MultiScaleRetention(hidden_dim, heads, double_v_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_size),
            nn.GELU(),
            nn.Linear(ffn_size, hidden_dim)
        )
        self.layer_norms_1 = nn.LayerNorm(hidden_dim)
        self.layer_norms_2 = nn.LayerNorm(hidden_dim)

    def forward(self, X):
        """
        X: (batch_size, sequence_length, hidden_size)
        """
        X = self.retention(self.layer_norms_1(X)) + X
        X = self.ffn(self.layer_norms_2(X)) + X
        return X


class RetNetDecoderLayer(nn.Module):
    def __init__(self, hidden_dim, ffn_size, heads, double_v_dim):
        super(RetNetDecoderLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.ffn_size = ffn_size
        self.heads = heads
        self.retention = MultiScaleRetention(hidden_dim, heads, double_v_dim)
        self.cross_retention = MultiScaleDecoderRetention(hidden_dim, heads, double_v_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_size),
            nn.GELU(),
            nn.Linear(ffn_size, hidden_dim)
        )
        self.layer_norms_1 = nn.LayerNorm(hidden_dim)
        self.layer_norms_2 = nn.LayerNorm(hidden_dim)
        self.layer_norms_3 = nn.LayerNorm(hidden_dim)

    def forward(self, X, mem):
        """
        X: (batch_size, sequence_length, hidden_size)
        """
        X = self.retention(self.layer_norms_1(X)) + X
        X = self.cross_retention(self.layer_norms_2(X), mem) + X
        X = self.ffn(self.layer_norms_2(X)) + X
        return X


class RetnetEncoderLayers(nn.Module):
    def __init__(self, encoder_layer: RetNetEncoderLayer, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [deepcopy(encoder_layer) for _ in range(num_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x


class RetnetDecoderLayers(nn.Module):
    def __init__(self, decoder_layer: RetNetDecoderLayer, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [deepcopy(decoder_layer) for _ in range(num_layers)]
        )

    def forward(self, x, mem):
        for layer in self.layers:
            x = layer.forward(x, mem)
        return x
