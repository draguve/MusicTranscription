import numpy as np
from torch.utils.data import Dataset
import h5py
import json


class TranscriptionDataset(Dataset):
    def __init__(self, dataset_file):
        self.h5file = h5py.File(dataset_file, "r")
        data = json.loads(self.h5file.get("Meta").attrs["section_index"])
        self.keys = np.array(list(data.keys()))
        self.lengthOfDataset = len(self.keys)

    def __getitem__(self, item):
        sectionGroup = self.h5file[f"/Songs/{self.keys[item]}"]
        mel = sectionGroup["mel"]
        tokens = sectionGroup["tokens"]
        return mel, tokens

    def __len__(self):
        return self.lengthOfDataset


def test():
    dataset = TranscriptionDataset("../Trainsets/S_Tier_1695286490_mTokens1000_mNoS60.hdf5")
    for i in range(len(dataset)):
        mel, token = dataset[i]
        print(mel.shape, token.shape)


if __name__ == '__main__':
    test()
