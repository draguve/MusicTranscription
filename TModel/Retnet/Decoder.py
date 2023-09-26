from torch import nn

from Externals.RetNet.src.retention import MultiScaleRetention
from decoder_retention import MultiScaleDecoderRetention
from TranscriptionDataset.TranscriptionDataset import getDataPipe


class RetNetDecoder(nn.Module):
    def __init__(self, layers, hidden_dim, ffn_size, heads, double_v_dim=False):
        super().__init__()
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.ffn_size = ffn_size
        self.heads = heads
        self.v_dim = hidden_dim * 2 if double_v_dim else hidden_dim

        self.self_retentions = nn.ModuleList([
            MultiScaleRetention(hidden_dim, heads, double_v_dim)
            for _ in range(layers)
        ])

        self.mem_retentions = nn.ModuleList([
            MultiScaleDecoderRetention(hidden_dim, heads, double_v_dim)
            for _ in range(layers)
        ])

        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, ffn_size),
                nn.GELU(),
                nn.Linear(ffn_size, hidden_dim)
            )
            for _ in range(layers)
        ])
        self.layer_norms_1 = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(layers)
        ])
        self.layer_norms_2 = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(layers)
        ])
        self.layer_norms_3 = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(layers)
        ])

    def forward(self, X, Mem):
        """
        X: (batch_size, sequence_length, hidden_size)
        """
        for i in range(self.layers):
            Y = self.self_retentions[i](self.layer_norms_1[i](X)) + X
            Z = self.mem_retentions[i](self.layer_norms_2[i](Y), Mem) + Y
            X = self.ffns[i](self.layer_norms_2[i](Z)) + Z
        return X

    def forward_recurrent(self, x_n, s_n_1s, n):
        raise NotImplemented()

    def forward_chunkwise(self, x_i, r_i_1s, i):
        raise NotImplemented()


def test():
    datasetLocation = r"C:\Users\ritwi\Github\MusicTranscription\Trainsets\S_Tier_1695619803_mTokens400_mNoS5.hdf5"
    dataset, pipe = getDataPipe(datasetLocation, 2)
    model = RetNetDecoder(2, 512, 2048, 8)
    itx = iter(pipe)
    batch = next(itx)
    all_mels, src_mask, src_pad_mask, all_tokens, tgt_mask, tgt_pad_mask, tokens_out = batch
    output = model(all_mels,all_mels)
    print(output)


if __name__ == '__main__':
    test()
