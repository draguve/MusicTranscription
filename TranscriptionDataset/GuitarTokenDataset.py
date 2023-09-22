import numpy as np
import torch
import torchdata
import torchdata.datapipes.iter as tdi
import h5py
import json
from Tokenizer.loaderH5 import H5GuitarTokenizer


class GuitarTokenDataset(torchdata.datapipes.iter.IterDataPipe):
    def __init__(self, datasetFile, contextSize=2):
        self.h5file = h5py.File(datasetFile, "r")
        data = json.loads(self.h5file.get("Meta").attrs["section_index"])
        self.keys = np.array(list(data.keys()))
        self.lengthOfDataset = 0
        self.contextSize = contextSize
        for key in data:
            self.lengthOfDataset += data[key]["numTokens"]

    def __iter__(self):
        for key in range(self.keys.shape[0]):
            sectionGroup = self.h5file[f"/Songs/{self.keys[key]}"]
            tokens = sectionGroup["tokens"][...]
            for i in range(self.contextSize, tokens.shape[0] - self.contextSize):
                yield np.concatenate((tokens[i - self.contextSize:i], tokens[i + 1:i + self.contextSize + 1])), tokens[
                    i]

    def __len__(self):
        return self.lengthOfDataset


def getDataPipe(datasetLocation, contextSize=2, shuffleBufferSize=50000, prefetchSize=None,
                pinMemory=False):
    pipe = GuitarTokenDataset(datasetLocation, contextSize)
    pipe = pipe.shuffle(buffer_size=shuffleBufferSize)
    if prefetchSize is not None:
        pipe = tdi.Prefetcher(pipe, prefetchSize)
    if pinMemory:
        pipe = tdi.PinMemory(pipe)
    return pipe


def test():
    dataset = "../Trainsets/S_Tier_1695289757_mTokens1000_mNoS60.hdf5"
    # tokenizer = H5GuitarTokenizer(dataset)
    pipe = getDataPipe(dataset, 10)
    for i in pipe:
        print(i)


if __name__ == '__main__':
    test()
