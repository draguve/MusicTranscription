from torch import optim, nn, utils, Tensor
import lightning.pytorch as pl


class GuitarTokenEmbeddingModel(pl.LightningModule):
    def __init__(self, embedding_size=512, vocab_size=None):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.linear1 = nn.Linear(embedding_size, vocab_size)
        self.activation = nn.LogSoftmax(dim=1)
        self.loss = nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def forward(self, inputs):
        lookup_embeds = self.embeddings(inputs)
        embeds = lookup_embeds.sum(dim=1)
        out = self.linear1(embeds)
        out = self.activation(out)
        return out

    def training_step(self, batch, batch_idx):
        embedding, target = batch
        log_prob = self.forward(embedding)
        loss = self.loss(log_prob, target)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        self.log('val_loss', loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        self.log('test_loss', loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
