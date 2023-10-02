import importlib

import torch
from lightning.pytorch.strategies import DeepSpeedStrategy
from torch import optim, nn, utils, Tensor
import lightning.pytorch as pl
from torchinfo import summary
from TModel.PositionalEncoding import PositionalEncoding
from TModel.GuitarTokenEmbeddingModel import GuitarTokenEmbeddingModel
from TModel.Retnet.RetNetLayer import RetNetEncoderLayer, RetnetEncoderLayers, RetNetDecoderLayer, RetnetDecoderLayers
from TranscriptionDataset.TranscriptionDataset import getDataPipe
from Tokenizer.loaderH5 import H5GuitarTokenizer
from Tokenizer.Onsets import convertTensorToF1

DEEPSPEED_ENABLED = False
if importlib.util.find_spec('deepspeed'):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

    DEEPSPEED_ENABLED = True

from Externals.RetNet.src.retnet import RetNet
from TModel.Retnet.Decoder import RetNetDecoder


class TranscriptionRetnetModel(pl.LightningModule):
    def __init__(self, vocabSize, tokenizer, d_model=512, d_ff=2048, num_layers=6, heads=8, dropout=0.1,
                 embeddingCheckpoint=None,
                 double_v_dim=False, lr_init=1e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, ):
        super().__init__()

        self.lr_init = lr_init
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.tokenizer = tokenizer

        # inputTransformation =
        self.num_layers = num_layers
        self.d_model = d_model
        self.vocabSize = vocabSize
        self.dropout = dropout
        self.d_ff = d_ff
        self.heads = heads
        self.v_dim = self.d_model * 2 if double_v_dim else self.d_model
        # self.positionalEncoding = PositionalEncoding(emb_size=self.d_model, dropout=self.dropout)
        encoderLayer = RetNetEncoderLayer(d_model, d_ff, heads, double_v_dim)
        self.encoder = RetnetEncoderLayers(encoderLayer, num_layers)
        decoderLayer = RetNetDecoderLayer(d_model, d_ff, heads, double_v_dim)
        self.decoder = RetnetDecoderLayers(decoderLayer, num_layers)
        if embeddingCheckpoint is None:
            self.tgt_embedding = nn.Embedding(self.vocabSize, d_model)
        else:
            embeddingModel = GuitarTokenEmbeddingModel.load_from_checkpoint(embeddingCheckpoint)
            self.tgt_embedding = embeddingModel.embeddings
        self.outputLinear = nn.Linear(d_model, vocabSize)
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.save_hyperparameters()

        self.total_loss = 0.0
        self.num_batches = 0
        self.total_f1 = 0.0

    def init_weights(self) -> None:
        initrange = 0.1
        self.outputLinear.bias.data.zero_()
        self.outputLinear.weight.data.uniform_(-initrange, initrange)

    def forward(self,
                src_emb: Tensor,
                trg: Tensor):
        src_emb_pos = src_emb
        tgt_emb_pos = (self.tgt_embedding(trg))
        mem = self.encoder.forward(src_emb_pos)
        outs = self.decoder.forward(tgt_emb_pos, mem)
        return self.outputLinear(outs)

    def training_step(self, batch, batch_idx):
        all_mels, tokens_in, tokens_out, lengths = batch
        logits = self.forward(all_mels, tokens_in)
        loss = self.loss(logits.reshape(-1, logits.shape[-1]), tokens_out.reshape(-1))
        self.total_loss += loss.item()
        self.num_batches += 1
        self.log("avg_train_loss", self.total_loss / self.num_batches, prog_bar=True)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        all_mels, tokens_in, tokens_out, lengths = batch
        logits = self.forward(all_mels, tokens_in)
        loss = self.loss(logits.reshape(-1, logits.shape[-1]), tokens_out.reshape(-1))
        # test = on
        self.total_f1 += convertTensorToF1(self.tokenizer, logits.cpu(), tokens_out, lengths)[0]
        self.total_loss += loss.item()
        self.num_batches += 1
        self.log("avg_test_loss", self.total_loss / self.num_batches, prog_bar=True)
        self.log("avg_test_f1", self.total_f1 / self.num_batches, prog_bar=True)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.transformer_model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)
        if DEEPSPEED_ENABLED:
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(self.parameters(), lr=self.lr_init, betas=self.betas, eps=self.eps,
                                        bias_correction=True, adamw_mode=True, weight_decay=self.weight_decay,
                                        amsgrad=False)
            return FusedAdam(self.parameters(), lr=self.lr_init, betas=self.betas, eps=self.eps,
                             bias_correction=True, adam_w_mode=True, weight_decay=self.weight_decay, amsgrad=False)
        return torch.optim.AdamW(self.parameters(), lr=self.lr_init, betas=self.betas, eps=self.eps,
                                 weight_decay=self.weight_decay)

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False

    def on_train_epoch_end(self):
        self.total_loss = 0.0
        self.num_batches = 0
        self.total_f1 = 0.0

    def on_test_epoch_end(self):
        self.total_loss = 0.0
        self.num_batches = 0
        self.total_f1 = 0.0


def main():
    datasetLocation = r"C:\Users\ritwi\Github\MusicTranscription\Trainsets\S_Tier_1695619803_mTokens400_mNoS5.hdf5"
    tokenizer = H5GuitarTokenizer(datasetLocation)
    dataset, pipe = getDataPipe(datasetLocation, 2, batchFirst=True)
    model = TranscriptionRetnetModel(dataset.getVocabSize(), tokenizer)
    # print(summary(model, ((10, 50, 512), (10, 50), (10, 300), (10, 300)),
    #               dtypes=[torch.float, torch.bool, torch.long, torch.bool]))
    itx = iter(pipe)
    x = next(itx)
    test_loss = model.training_step(x, 10)
    print(test_loss)


if __name__ == '__main__':
    main()
