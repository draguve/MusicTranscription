import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
import torchshape
import math

from SongDataset import SongDataset


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: torch.Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class GuitarModel(nn.Module):
    def __init__(self, input_shape,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 multi_head_attention_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(GuitarModel, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.dropout2d = nn.Dropout1d(dropout)
        self.conv1 = nn.Conv2d(
            in_channels=2,
            out_channels=4,
            kernel_size=(5, 5)
        )
        self.relu = nn.GELU()
        self.maxPool1 = nn.MaxPool2d(
            kernel_size=(3, 3),
            stride=(3, 3)
        )
        self.conv2 = nn.Conv2d(
            in_channels=4,
            out_channels=10,
            kernel_size=(5, 5)
        )
        self.relu2 = nn.GELU()
        self.maxPool2 = nn.MaxPool2d(
            kernel_size=(2, 2),
            stride=(2, 2)
        )
        self.conv3 = nn.Conv2d(
            in_channels=10,
            out_channels=25,
            kernel_size=(5, 5)
        )
        self.relu3 = nn.GELU()
        self.maxPool3 = nn.MaxPool2d(
            kernel_size=(2, 2),
            stride=(2, 2)
        )
        output_shape = torchshape.tensorshape(self.conv1, input_shape)
        output_shape = torchshape.tensorshape(self.maxPool1, output_shape)
        output_shape = torchshape.tensorshape(self.conv2, output_shape)
        output_shape = torchshape.tensorshape(self.maxPool2, output_shape)
        output_shape = torchshape.tensorshape(self.conv3, output_shape)
        output_shape = torchshape.tensorshape(self.maxPool3, output_shape)
        self.flattenedSize = 1
        for i in output_shape:
            self.flattenedSize *= i
        # we add 8 for (6 tuning, 1 arrangement, 1 capo)
        self.fc1 = nn.Linear(in_features=self.flattenedSize + 8, out_features=emb_size)
        self.relu4 = nn.GELU()
        # TODO: need to do Token Embedding for the target/guitar tokens
        self.tgt_embedding = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positionalEncoding = PositionalEncoding(emb_size, dropout)
        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=multi_head_attention_size,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            dim_feedforward=dim_feedforward
        )
        self.generator = nn.Linear(emb_size, tgt_vocab_size)

    def forward(self, x, tuningAndArrangement, tgt, tgt_mask, tgt_pad_mask):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxPool1(x)
        x = self.dropout2d(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxPool2(x)
        x = self.dropout2d(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxPool3(x)
        x = self.dropout2d(x)
        x = torch.flatten(x)
        x = torch.cat((x, tuningAndArrangement))
        x = self.fc1(x)
        x = self.relu4(x)
        # x = self.dropout(x)
        # dropout in embedding as well
        print(x.shape)
        x = self.tgt_embedding(x)
        print(x.shape)
        x = self.positionalEncoding(x)
        print(x.shape)
        tgt = self.tgt_embedding(tgt)
        print(tgt.shape)
        tgt = self.positionalEncoding(tgt)
        print(tgt.shape)
        x = self.transformer(x, tgt, tgt_mask=tgt_mask)
        x = self.generator(x)
        return x


if __name__ == '__main__':
    SAMPLE_RATE = 44100
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=2048,
        hop_length=512,
        n_mels=128
    )
    dataset = SongDataset("../test.hdf5", mel_spectrogram, sampleRate=SAMPLE_RATE)
    print(dataset.maxTokens)
    spectrogram, tuningAndArrangement, tokens, token_padding_mask, target_mask = dataset[1]
    model = GuitarModel((1, 2, 128, 87),
                        emb_size=512,
                        num_encoder_layers=3,
                        num_decoder_layers=3,
                        multi_head_attention_size=4,
                        dim_feedforward=512,
                        tgt_vocab_size=dataset.vocabSize)
    output = model.forward(spectrogram, tuningAndArrangement, tokens, target_mask, token_padding_mask)
    print(output.shape)

    total_params = sum(
        param.numel() for param in model.parameters()
    )
    print(total_params)
