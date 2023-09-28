from ctypes import Union

import torch
from torch import optim, nn, utils, Tensor
import lightning.pytorch as pl
from yet_another_retnet.retention import ActivationString
from torchinfo import summary

from TModel.PositionalEncoding import PositionalEncoding
from TModel.GuitarTokenEmbeddingModel import GuitarTokenEmbeddingModel
from TModel.Retnet.RetNetEncoder import RetNetEncoderLayer, RetNetEncoder
from TranscriptionDataset.TranscriptionDataset import getDataPipe
from yet_another_retnet.retnet import RetNetDecoder, RetNetDecoderLayer
from TModel.Retnet.RetnetCrossDecoder import RetNetCrossDecoder, RetNetCrossLayer
from typing import Callable, List, Optional, Sequence, Tuple, Union


class RetnetEncoderDecoder(pl.LightningModule):
    def __init__(
            self,
            vocabSize,
            d_model=512,
            d_ff=2048,
            nhead: int = 8,
            num_layers: int = 6,
            dropout: float = 0.1,
            activation: Union[ActivationString, Callable[[Tensor], Tensor]] = "swish",
            normFirst: bool = True,
            layer_norm_eps: float = 1e-5,
            embeddingCheckpoint=None
    ):
        super().__init__()
        # inputTransformation =
        self.d_model = d_model
        self.vocabSize = vocabSize
        self.dropout = dropout
        self.d_ff = d_ff
        encoder_layer = RetNetEncoderLayer(
            d_model,
            nhead,
            d_ff,
            dropout,
            activation,
            normFirst,
            layer_norm_eps,
            # device=device,
            # dtype=dtype,
        )
        self.encoder = RetNetEncoder(encoder_layer, num_layers)
        decoder_layer = RetNetCrossLayer(
            d_model,
            nhead,
            dropout=dropout,
            activation=activation,
            dim_feedforward=self.d_ff,
            norm_first=normFirst,
            layer_norm_eps=layer_norm_eps,
            # device=device,
            # dtype=dtype,
        )
        self.decoder = RetNetCrossDecoder(decoder_layer, num_layers)
        if embeddingCheckpoint is None:
            self.tgt_embedding = nn.Embedding(self.vocabSize, d_model)
        else:
            embeddingModel = GuitarTokenEmbeddingModel.load_from_checkpoint(embeddingCheckpoint)
            self.tgt_embedding = embeddingModel.embeddings
        self.out = nn.Linear(d_model, vocabSize)
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.init_weights()

    def init_weights(self) -> None:
        nn.init.xavier_normal_(self.out.weight)
        nn.init.constant_(self.out.bias, 0)

    def forward(self,
                src_emb: Tensor,
                src_padding_mask: Tensor,
                tgt: Tensor,
                tgt_padding_mask: Tensor):
        mem = self.encoder.forward_parallel(src_emb)
        tgt_emb = self.tgt_embedding(tgt)
        out = self.decoder.forward_parallel(tgt_emb, mem)
        return self.out(out)

    def training_step(self, batch, batch_idx):
        all_mels, src_mask, src_pad_mask, all_tokens, tgt_mask, tgt_pad_mask, tokens_out = batch
        logits = self.forward(all_mels, src_pad_mask, all_tokens, tgt_pad_mask)
        loss = self.loss(logits.reshape(-1, logits.shape[-1]), tokens_out.reshape(-1))
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        all_mels, src_mask, src_pad_mask, all_tokens, tgt_mask, tgt_pad_mask, tokens_out = batch
        logits = self.forward(all_mels, src_pad_mask, all_tokens, tgt_pad_mask)
        loss = self.loss(logits.reshape(-1, logits.shape[-1]), tokens_out.reshape(-1))
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.transformer_model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5)
        return optimizer


def test():
    datasetLocation = r"C:\Users\ritwi\Github\MusicTranscription\Trainsets\S_Tier_1695619803_mTokens400_mNoS5.hdf5"
    dataset, pipe = getDataPipe(datasetLocation, 2, batchFirst=True)
    model = RetnetEncoderDecoder(dataset.getVocabSize())
    print(
        summary(
            model,
            ((10, 50, 512), (10, 50), (10, 300), (10, 300)),
            dtypes=[torch.float, torch.bool, torch.long, torch.bool],
            depth=7
        )
    )
    # print(str(model.parameters()))
    # itx = iter(pipe)
    # x = next(itx)
    # test_loss = model.training_step(x, 10)
    # print(test_loss)


if __name__ == '__main__':
    test()
