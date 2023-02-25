import json
from typing import Any

import torch
import torchaudio
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchshape
import math
import h5py
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from TUtils import ArrangementUtils
from pytorch_lightning import LightningModule
from TranscriptionDataset import ArrangementDataset


class TuningModel(LightningModule):
    def __init__(self, tuning_size, capo_size, arrangement_size):
        super().__init__()

        # mnist images are (1, 28, 28) (channels, height, width)
        self.conv1 = nn.Conv1d(2, 35, kernel_size=80, stride=16)
        self.bn1 = nn.BatchNorm1d(35)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(35, 35, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(35)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(35, 70, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(70)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(70, 70, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(70)
        self.pool4 = nn.MaxPool1d(4)
        # self.fcTuning = nn.Linear(70, tuning_size)
        # self.fcCapo = nn.Linear(70, capo_size)
        self.fcArrangement = nn.Linear(70, arrangement_size)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        # tuning = self.fcTuning(x)
        # capo = self.fcCapo(x)
        arrangement = self.fcArrangement(x)
        # return F.log_softmax(tuning, dim=2), F.log_softmax(capo, dim=2), F.log_softmax(arrangement, dim=2)
        return F.log_softmax(arrangement, dim=2)

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    def training_step(self, batch, batch_idx):
        section, tuning, capo, arrangement = batch
        output = self(section).squeeze()
        loss = F.nll_loss(output, torch.argmax(arrangement.squeeze(), dim=1))
        acc = (torch.argmax(output, 1) == torch.argmax(arrangement.squeeze(), 1)).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, val_batch, batch_idx):
        section, tuning, capo, arrangement = val_batch
        output = self(section).squeeze()
        loss = F.nll_loss(output, torch.argmax(arrangement.squeeze(), dim=1))
        acc = (torch.argmax(output, 1) == torch.argmax(arrangement.squeeze(), 1)).float().mean()
        self.log('val_loss', loss)
        self.log('loss_acc', acc)


if __name__ == '__main__':
    SAMPLE_RATE = 16000
    dataset = "../Trainsets/massive_test2.hdf5"
    transform = ArrangementDataset.OneHotEncodeArrangement(dataset)
    dataset = ArrangementDataset.ArrangementDataset(dataset, sampleRate=SAMPLE_RATE, oneHotTransform=transform)
    dataset.start()
    dataset.load_all_data()
    loader = DataLoader(
        dataset=dataset,
        batch_size=7,
        # collate_fn=guitarCollate,
        worker_init_fn=ArrangementDataset.worker_init_fn,
        # num_workers=2
    )  # , num_workers=4)
    dataiter = iter(loader)
    section, tuning, capo, arrangement = next(dataiter)
    print(len(next(dataiter)))
    print(f"{section.shape} {tuning.shape} {capo.shape} {arrangement.shape}")
    test = TuningModel(
        tuning_size=transform.tuning_output_size,
        capo_size=transform.capo_output_size,
        arrangement_size=transform.arrangement_output_size
    )
    output = test.forward(section)
    test.training_step(next(dataiter), 0)
    pass
