from torch import optim, nn, utils, Tensor
import lightning.pytorch as pl


class GuitarTokenEmbeddingModel(pl.LightningModule):
    def __init__(self, context_size=2, embedding_size=512, vocab_size=None):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.linear1 = nn.Linear(embedding_size, vocab_size)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, inputs):
        lookup_embeds = self.embeddings(inputs)
        embeds = lookup_embeds.sum(dim=0)
        out = self.linear1(embeds)
        out = nn.functional.log_softmax(out)
        return out

    def training_step(self, batch, batch_idx):
        embedding, target = batch
        log_prob = self.forward(embedding)
        loss = self.loss(log_prob, target)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
