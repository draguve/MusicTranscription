import torch
from torch import optim, nn, utils, Tensor
import lightning.pytorch as pl
import PositionalEncoding


class TranscriptionTransformerModel(pl.LightningModule):
    def __init__(self, vocabSize, d_model=512, d_ff=512, dropout=0.1):
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
            batch_first=True,
        )
        self.tgt_embedding = nn.Embedding(self.vocabSize, d_model)
        self.outputNorm = nn.Linear(d_model, vocabSize)
        self.loss = torch.nn.CrossEntropyLoss()

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
        src_emb_pos = self.positional_encoding(src_emb)
        tgt_emb_pos = self.positional_encoding(self.positionalEncoding(trg))
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
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.transformer_model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
        return optimizer
