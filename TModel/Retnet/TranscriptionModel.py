import torch
from torch import optim, nn, utils, Tensor
import lightning.pytorch as pl
from TModel.PositionalEncoding import PositionalEncoding
from TModel.GuitarTokenEmbeddingModel import GuitarTokenEmbeddingModel
from TranscriptionDataset.TranscriptionDataset import getDataPipe
from Externals.RetNet.src.retnet import RetNet
from TModel.Retnet.Decoder import RetNetDecoder
from Externals.RetNet.src.xpos_relative_position import XPOS


class TranscriptionRetnetModel(pl.LightningModule):
    def __init__(self, vocabSize, d_model=512, d_ff=2048, num_layers=6, heads=8, dropout=0.1, embeddingCheckpoint=None):
        super().__init__()
        # inputTransformation =
        self.num_layers = num_layers
        self.d_model = d_model
        self.vocabSize = vocabSize
        self.dropout = dropout
        self.d_ff = d_ff
        self.heads = heads
        # self.positionalEncoding = PositionalEncoding(emb_size=self.d_model, dropout=self.dropout)
        self.x_pos = XPOS(self.d_model)
        self.encoder = RetNet(self.num_layers, self.d_model, self.d_ff, self.heads, False)
        self.decoder = RetNetDecoder(self.num_layers, self.d_model, self.d_ff, self.heads, False)
        if embeddingCheckpoint is None:
            self.tgt_embedding = nn.Embedding(self.vocabSize, d_model)
        else:
            embeddingModel = GuitarTokenEmbeddingModel.load_from_checkpoint(embeddingCheckpoint)
            self.tgt_embedding = embeddingModel.embeddings
        self.outputLinear = nn.Linear(d_model, vocabSize)
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=0)

    def init_weights(self) -> None:
        initrange = 0.1
        self.outputLinear.bias.data.zero_()
        self.outputLinear.weight.data.uniform_(-initrange, initrange)

    def forward(self,
                src_emb: Tensor,
                src_padding_mask: Tensor,
                trg: Tensor,
                tgt_padding_mask: Tensor):
        src_emb_pos = (src_emb)
        tgt_emb_pos = (self.tgt_embedding(trg))
        mem = self.encoder.forward(src_emb_pos)
        outs = self.decoder.forward(tgt_emb_pos, mem)
        return self.outputLinear(outs)

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
        optimizer = torch.optim.AdamW(self.transformer_model.parameters(), lr=1e-5)
        return optimizer


def test():
    datasetLocation = r"C:\Users\ritwi\Github\MusicTranscription\Trainsets\S_Tier_1695619803_mTokens400_mNoS5.hdf5"
    dataset, pipe = getDataPipe(datasetLocation, 2, batchFirst=True)
    model = TranscriptionRetnetModel(dataset.getVocabSize())
    itx = iter(pipe)
    x = next(itx)
    test_loss = model.training_step(x, 10)
    print(test_loss)


if __name__ == '__main__':
    test()
