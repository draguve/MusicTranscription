import numpy as np
import torch
import torchdata
import torchdata.datapipes.iter as tdi
import h5py
import json
from torch.utils.data import DataLoader, get_worker_info

from Tokenizer.loaderH5 import H5GuitarTokenizer


class GuitarTokenDataset(torchdata.datapipes.iter.IterDataPipe):
    def __init__(self, datasetFile, contextSize=2):
        self.h5file = None
        self.datasetFile = datasetFile
        h5file = h5py.File(datasetFile, "r")
        meta = h5file.get("Meta")
        self.cfs = meta.get("cfs")[...]
        self.section_data = json.loads(meta.attrs["section_index"])
        self.meta_data = {}
        for key in meta.attrs.keys():
            if "index" not in key:
                self.meta_data[key] = meta.attrs[key]
        self.lengthOfDataset = 0
        self.contextSize = contextSize
        for key in self.section_data:
            self.lengthOfDataset += self.section_data[key]["numTokens"]

    def getVocabSize(self):
        return self.meta_data["vocab_size"]

    def getCfs(self):
        return self.cfs

    def __iter__(self):
        worker = get_worker_info()
        self.keys = np.array(list(self.section_data.keys()))
        if worker is not None:
            split_size = len(self.keys) // worker.num_workers
            self.keys = self.keys[worker.id * split_size:(worker.id + 1) * split_size]
        np.random.shuffle(self.keys)
        h5file = h5py.File(self.datasetFile, "r")
        for key in range(self.keys.shape[0]):
            sectionGroup = h5file[f"/Songs/{self.keys[key]}"]
            tokens = sectionGroup["tokens"][...]
            for i in range(self.contextSize, tokens.shape[0] - self.contextSize):
                # yield (np.concatenate((tokens[i - self.contextSize:i], tokens[i + 1:i + self.contextSize + 1])),
                #        torch.tensor(tokens[i], dtype=torch.long))
                for j in range(i - self.contextSize, i):
                    yield torch.tensor(tokens[j]), torch.tensor(tokens[i], dtype=torch.long)
                for j in range(i + 1, i + self.contextSize + 1):
                    yield torch.tensor(tokens[j]), torch.tensor(tokens[i], dtype=torch.long)


def getDataPipe(datasetLocation, contextSize=2,
                pinMemory=False):
    dataset = GuitarTokenDataset(datasetLocation, contextSize)
    pipe = dataset
    # pipe = dataset.shuffle(buffer_size=shuffleBufferSize)
    # if prefetchSize is not None:
    # pipe = tdi.Prefetcher(pipe, prefetchSize)
    if pinMemory:
        pipe = tdi.PinMemory(pipe)
    return dataset, pipe


def test():
    dataset = "../Trainsets/TokensOnly_1695456272_mTokens1000_mNoS60.hdf5"
    # tokenizer = H5GuitarTokenizer(dataset)
    dataset, pipe = getDataPipe(dataset, 2)
    print(dataset.cfs)
    train_dl = DataLoader(dataset=pipe, batch_size=64, num_workers=2)
    for i in train_dl:
        print(i)
        break


if __name__ == '__main__':
    test()
