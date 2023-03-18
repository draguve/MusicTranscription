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


class ArrangementModel(LightningModule):
    def __init__(self, arrangement_size):
        # self.count=0
        super().__init__()
        self.acc = torch.tensor([0])
        self.conv1 = nn.Conv1d(2, 35, kernel_size=80, stride=16)
        self.bn1 = nn.BatchNorm1d(35)
        self.pool1 = nn.MaxPool1d(4)
        self.batchNorm1 = nn.BatchNorm1d(35)
        self.conv2 = nn.Conv1d(35, 35, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(35)
        self.pool2 = nn.MaxPool1d(4)
        self.batchNorm2 = nn.BatchNorm1d(35)
        self.conv3 = nn.Conv1d(35, 70, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(70)
        self.pool3 = nn.MaxPool1d(4)
        self.batchNorm3 = nn.BatchNorm1d(70)
        self.conv4 = nn.Conv1d(70, 70, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(70)
        self.pool4 = nn.MaxPool1d(4)
        # self.fcTuning = nn.Linear(70, tuning_size)
        # self.fcCapo = nn.Linear(70, capo_size)
        self.fcArrangement = nn.Linear(70, arrangement_size)
        self.save_hyperparameters()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.batchNorm1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.batchNorm2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.batchNorm3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        # tuning = self.fcTuning(x)
        # capo = self.fcCapo(x)
        arrangement = self.fcArrangement(x)
        # return F.log_softmax(tuning, dim=2), F.log_softmax(capo, dim=2), F.log_softmax(arrangement, dim=2)
        return arrangement

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    def get_loss_and_acc(self, batch, batch_idx):
        section, tuning, arrangement = batch
        output = self(section).squeeze(1)
        # print(f"{self.count} {section.shape} {tuning.shape} {capo.shape} {arrangement.shape} {arrangement.squeeze(1).shape} {torch.argmax(arrangement.squeeze(1), dim=1).shape} {output.shape}")
        # self.count+=1
        loss = F.mse_loss(output, arrangement.squeeze(1))
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.get_loss_and_acc(batch, batch_idx)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss = self.get_loss_and_acc(val_batch, batch_idx)
        self.log('val_loss', loss)

    # def training_epoch_end(self, outputs):
    #     self.acc = torch.tensor([0])
    #
    # def validation_epoch_end(self, outputs):
    #     self.acc = torch.tensor([0])


if __name__ == '__main__':
    SAMPLE_RATE = 16000
    dataset = "../Trainsets/massive_test2.hdf5"
    # transform = ArrangementDataset.OneHotEncodeArrangement(dataset)
    dataset = ArrangementDataset.ArrangementDataset(dataset, sampleRate=SAMPLE_RATE, oneHotTransform=None)
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
    section, tuning, arrangement = next(dataiter)
    print(len(next(dataiter)))
    print(f"{section.shape} {tuning.shape} {arrangement.shape}")
    test = ArrangementModel(
        arrangement_size=6
    )
    output = test.forward(section)
    test.training_step(next(dataiter), 0)
    pass
