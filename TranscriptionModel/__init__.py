import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
import torchshape
import math

from SongDataset import SongDataset, GuitarCollater


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
                 transformation,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 multi_head_attention_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 pad_token: int = 6873):
        super(GuitarModel, self).__init__()
        self.pad_token = pad_token
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=10,
            kernel_size=(5, 5)
        )
        self.gelu = nn.GELU()
        self.maxPool = nn.MaxPool2d(
            kernel_size=(2, 2),
            stride=(2, 2)
        )
        self.transformation = transformation

        output_shape = torchshape.tensorshape(self.conv, input_shape)
        output_shape = torchshape.tensorshape(self.maxPool, output_shape)

        # we add 8 for (6 tuning, 1 arrangement, 1 capo)
        self.fc1 = nn.Linear(in_features=output_shape[1] * output_shape[2] + 8, out_features=emb_size)
        self.tuning_seq_len = output_shape[3]

        self.gelu2 = nn.GELU()

        # Token Embedding for the target/guitar tokens
        self.tgt_embedding = TokenEmbedding(tgt_vocab_size, emb_size)  # the sqrt is done inside the forward pass
        self.positionalEncoding = PositionalEncoding(emb_size, dropout)
        self.numHeads = multi_head_attention_size
        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=multi_head_attention_size,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            dim_feedforward=dim_feedforward
        )
        self.generator = nn.Linear(emb_size, tgt_vocab_size)

    def forward(self, x, tuning, tgt, tgt_mask, tgt_pad_mask):
        x = self.transformation(x)
        # current shape = (batch,2,128,87)
        x = self.conv(x)
        x = self.gelu(x)
        x = self.maxPool(x)
        x = self.dropout(x)

        x = x.permute(1, 2, 3, 0)  # convert shape to (chan,128,87,batch)
        x = x.view((-1, *x.shape[2:]))  # convert shape to (post_conv*chan,87-conv_kernel_size,batch)
        x = x.permute(2, 1, 0)  # convert shape to (batch,87-conv_kernel_size,post_conv*chan)

        # create shape of tuning to (batch,seq_len,8)
        tuning = tuning.unsqueeze(1).repeat(1, self.tuning_seq_len, 1)
        x = torch.cat((x, tuning), 2)  # convert shape of to (batch,87-conv_kernel_size,post_conv*chan+8)

        x = self.fc1(x)
        x = self.gelu2(x)
        x = self.dropout2(x)
        # current input shape = (batch_size, sequence length, dim_model)
        x = self.positionalEncoding(x)
        x = x.permute(1, 0, 2)  # current input shape = (sequence length,batch_size, dim_model)

        # current target shape = (batch_size, sequence length, dim_model)
        tgt = self.tgt_embedding(tgt)
        tgt = self.positionalEncoding(tgt)
        tgt = tgt.permute(1, 0, 2)  # to get the shape (sequence length, batch_size, dim_model),

        # all the tgt_masks are the same, removed extra dimensions
        x = self.transformer(x, tgt, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_pad_mask)
        x = self.generator(x)
        return x

    def create_masks(self, tgt_tokens):
        token_padding_mask = (tgt_tokens == self.pad_token)
        target_mask = torch.nn.Transformer.generate_square_subsequent_mask(tgt_tokens.size(1))
        return target_mask, token_padding_mask


if __name__ == '__main__':
    SAMPLE_RATE = 44100
    BATCH_SIZE = 64
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=2048,
        hop_length=512,
        n_mels=128
    )
    dataset = SongDataset("../massive.hdf5", sampleRate=SAMPLE_RATE)
    collate_fn = GuitarCollater(dataset.pad_token)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE,collate_fn=collate_fn)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=dataset.pad_token)
    data_iter = iter(loader)
    spectrogram, tuningAndArrangement, tokens = next(data_iter)
    model = GuitarModel((BATCH_SIZE, 2, 128, 87),
                        transformation=mel_spectrogram,
                        emb_size=512,
                        num_encoder_layers=3,
                        num_decoder_layers=3,
                        multi_head_attention_size=4,
                        dim_feedforward=512,
                        tgt_vocab_size=dataset.vocabSize,
                        pad_token=dataset.pad_token)
    y_input = tokens[:, :-1]
    y_expected = tokens[:, 1:]
    target_mask, token_padding_mask = model.create_masks(y_input)
    print(spectrogram.shape)
    print(tuningAndArrangement.shape)
    print(y_input.shape)
    print(target_mask.shape)
    print(token_padding_mask.shape)
    output = model.forward(spectrogram, tuningAndArrangement, y_input, target_mask, token_padding_mask)
    output = output.permute(1, 2, 0)
    # loss = loss_fn(output, y_expected)
    print(output.shape)
    # print(loss)
    total_params = sum(
        param.numel() for param in model.parameters()
    )
    print(total_params)
