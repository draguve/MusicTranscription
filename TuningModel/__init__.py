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
    all_tunings = {}
    h5file = h5py.File("../Trainsets/massive_test2.hdf5", "r")
    songsGroup = h5file["Songs"]
    data = list(json.loads(songsGroup.attrs["index"]).values())
    tunings = None
    capos = None
    arrangements = None
    for item in tqdm(data):
        songGroup = h5file[f"/Songs/{item['group']}"]
        tuning = str(songGroup["tuning"][0:])
        if tunings is None:
            tunings = songGroup["tuning"][0:]
            capos = np.array([int(songGroup.attrs["capo"])])
            thisArrangement = [int(ArrangementUtils.arrangementIndex[arr])+3 for arr in songGroup.attrs["allArrangements"]]
            arrangements = np.zeros(6)
            arrangements[thisArrangement] = 1
            print(arrangements)
        else:
            tunings = np.vstack((tunings, songGroup["tuning"][0:]))
            capo = np.array([int(songGroup.attrs["capo"])])
            capos = np.vstack((capos,capo))
            thisArrangement = [int(ArrangementUtils.arrangementIndex[arr]) + 3 for arr in
                               songGroup.attrs["allArrangements"]]
            arrangement = np.zeros(6)
            arrangement[thisArrangement] = 1
            arrangements = np.vstack((arrangements,arrangement))
        if tuning in all_tunings:
            all_tunings[tuning] += 1
        else:
            all_tunings[tuning] = 1
    print(capos.shape)
    oheTunings = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(tunings)
    oheCaps = OneHotEncoder(handle_unknown='ignore',sparse_output=False).fit(capos)
    print(arrangements)

