from torch import optim, nn, utils, Tensor
import lightning.pytorch as pl

from TModel import NegativeLoss


class GuitarTokenEmbeddingModel(pl.LightningModule):
    def __init__(self, embedding_size=512, vocab_size=None, cfs=None, lr=1e-3):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.embeddings.weight.data.uniform_(-1, 1)
        self.loss = NegativeLoss.SGNSLoss(self.embeddings, vocab_size, cfs)
        self.lr = lr
        self.save_hyperparameters()

    def forward(self, center, context):
        center_embeds = self.embeddings(center)
        context_embeds = self.embeddings(context)
        return center_embeds, context_embeds

    def training_step(self, batch, batch_idx):
        context, center = batch
        center_embeds, context_embeds = self.forward(center, context)
        loss = self.loss(center_embeds, context_embeds)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    # def validation_step(self, val_batch, batch_idx):
    #     x, y = val_batch
    #     logits = self.forward(x)
    #     loss = self.loss(logits, y)
    #     self.log('val_loss', loss, prog_bar=True)
    #
    # def test_step(self, batch, batch_idx):
    #     x, y = batch
    #     logits = self.forward(x)
    #     loss = self.loss(logits, y)
    #     self.log('test_loss', loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
