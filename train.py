from TModel.TransformerModel import TranscriptionTransformerModel
from Tokenizer.loaderH5 import H5GuitarTokenizer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
from TUtils import random_string
import lightning.pytorch as pl
from lightning import Trainer
import torch
from TranscriptionDataset import TranscriptionDataset
from TModel.Retnet.TranscriptionModel import TranscriptionRetnetModel
torch.set_float32_matmul_precision('medium')
import importlib

datasetLocation = "Trainsets/S_Tier_1695619803_mTokens400_mNoS5.hdf5"
wandbProject = "TranscriptionModel_Test"
batchSize = 4
num_workers = 2
enabled_wandb = False
enabled_checkpoint = False

dataset,pipe = TranscriptionDataset.getDataPipe(
    datasetLocation,
    batchSize,
    batchFirst=True
)
train_pipe,test_pipe = pipe.random_split({"train":0.8,"test":0.2},42,total_length=len(dataset))

train_dl = DataLoader(dataset=train_pipe,batch_size=None,num_workers=num_workers)
test_dl = DataLoader(dataset=test_pipe, batch_size=batchSize,num_workers=num_workers)

model = TranscriptionRetnetModel(
    dataset.getVocabSize(),
    d_model=512,
    d_ff=2048,
    # embeddingCheckpoint="Models/GuitarToken/Max2Length.ckpt"
)
try:
    torch.compile(model)
except Exception:
    print("Could not compile model with jit")

wandb_logger = None
if enabled_wandb:
    wandb_logger = WandbLogger(project=wandbProject)
    wandb_logger.experiment.config.update(dataset.meta_data)
    wandb_logger.experiment.config["batchSize"] = batchSize

checkpoint_callback = None
if enabled_checkpoint:
    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',
        dirpath=f'Models/GuitarTranscription/{random_string(10)}/',
        filename='GuitarTranscriptionModel-{epoch:02d}-{train_loss:.2f}',
        every_n_train_steps=1000,
        save_top_k=3,
        mode='min',
    )

trainer = Trainer(
    default_root_dir='Models/',
    max_epochs=5,
    profiler="simple",
    # profiler="pytorch",
    # logger=wandb_logger,
    # callbacks=[checkpoint_callback],
    max_time="00:00:05:00",
    precision="bf16-mixed",
)

trainer.fit(model=model, train_dataloaders=train_dl,val_dataloaders=test_dl)
