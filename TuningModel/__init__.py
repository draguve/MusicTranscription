import json

import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
import torchshape
import math
import h5py
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from TUtils import ArrangementUtils


class TuningModel(nn.Module):
    def __init__(self, ):
        super(TuningModel, self).__init__()

    def forward(self, input, tuning):
        return input


if __name__ == '__main__':
    pass

