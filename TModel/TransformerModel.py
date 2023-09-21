from torch import optim, nn, utils, Tensor
import lightning.pytorch as pl


class TranscriptionTransformerModel(pl.LightningModule):
    def __init__(self, vocabSize, d_model=512, d_ff=512):
        super().__init__()
        # inputTransformation =
        transformer_model = nn.Transformer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_ff,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dropout=0.1,
            activation=nn.functional.gelu
        )
        outputNorm = nn.Linear(d_model, vocabSize)

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
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
