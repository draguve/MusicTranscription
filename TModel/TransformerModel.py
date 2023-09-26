import torch
from torch import optim, nn, utils, Tensor
import lightning.pytorch as pl
from TModel.PositionalEncoding import PositionalEncoding
from TModel.GuitarTokenEmbeddingModel import GuitarTokenEmbeddingModel
from TranscriptionDataset.TranscriptionDataset import getDataPipe


class TranscriptionTransformerModel(pl.LightningModule):
    def __init__(self, vocabSize, d_model=512, d_ff=512, dropout=0.1, embeddingCheckpoint=None):
        super().__init__()
        # inputTransformation =
        self.d_model = d_model
        self.vocabSize = vocabSize
        self.dropout = dropout
        self.positionalEncoding = PositionalEncoding(emb_size=self.d_model, dropout=self.dropout)
        self.transformer_model = nn.Transformer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_ff,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dropout=self.dropout,
            activation=nn.functional.gelu,
        )
        if embeddingCheckpoint is None:
            self.tgt_embedding = nn.Embedding(self.vocabSize, d_model)
        else:
            embeddingModel = GuitarTokenEmbeddingModel.load_from_checkpoint(embeddingCheckpoint)
            self.tgt_embedding = embeddingModel.embeddings
        self.outputNorm = nn.Linear(d_model, vocabSize)
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=0)

    def init_weights(self) -> None:
        initrange = 0.1
        self.outputNorm.bias.data.zero_()
        self.outputNorm.weight.data.uniform_(-initrange, initrange)

    def forward(self,
                src_emb: Tensor,
                src_mask: Tensor,
                src_padding_mask: Tensor,
                trg: Tensor,
                tgt_mask: Tensor,
                tgt_padding_mask: Tensor):
        src_emb_pos = self.positionalEncoding(src_emb)
        tgt_emb_pos = self.positionalEncoding(self.tgt_embedding(trg))
        outs = self.transformer_model(
            src=src_emb_pos,
            src_mask=src_mask,
            src_key_padding_mask=src_padding_mask,
            tgt=tgt_emb_pos,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )
        return self.outputNorm(outs)

    def encode(self, src_embed: Tensor, src_mask: Tensor):
        return self.transformer_model.encoder(self.positional_encoding(src_embed), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer_model.decoder(self.positionalEncoding(
            self.tgt_embedding(tgt)), memory,
            tgt_mask)

    def training_step(self, batch, batch_idx):
        all_mels, src_mask, src_pad_mask, all_tokens, tgt_mask, tgt_pad_mask, tokens_out = batch
        logits = self.forward(all_mels, src_mask, src_pad_mask, all_tokens, tgt_mask, tgt_pad_mask)
        loss = self.loss(logits.reshape(-1, logits.shape[-1]), tokens_out.reshape(-1))
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        all_mels, src_mask, src_pad_mask, all_tokens, tgt_mask, tgt_pad_mask, tokens_out = batch
        logits = self.forward(all_mels, src_mask, src_pad_mask, all_tokens, tgt_mask, tgt_pad_mask)
        loss = self.loss(logits.reshape(-1, logits.shape[-1]), tokens_out.reshape(-1))
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.transformer_model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)
        optimizer = torch.optim.AdamW(self.transformer_model.parameters(), lr=1e-5)
        return optimizer


def test():
    datasetLocation = "../Trainsets/S_Tier_1695428558_mTokens1000_mNoS60.hdf5"
    dataset, pipe = getDataPipe(datasetLocation, 2)
    model = TranscriptionTransformerModel(dataset.getVocabSize())
    itx = iter(pipe)
    x = next(itx)
    test_loss = model.training_step(x, 10)
    print(test_loss)


if __name__ == '__main__':
    test()
