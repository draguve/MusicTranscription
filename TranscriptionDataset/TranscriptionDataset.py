import math
from pprint import pprint

import numpy as np
import torch
import torchdata
import torchdata.datapipes.iter as tdi
import h5py
import json
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor
from torch.utils.data import get_worker_info, DataLoader
from tqdm import tqdm

from Tokenizer.loaderH5 import H5GuitarTokenizer


class TranscriptionDataset(torchdata.datapipes.iter.IterDataPipe):
    def __init__(self, datasetFile, batchSize=10):
        self.h5file = None
        self.datasetFile = datasetFile
        h5file = h5py.File(datasetFile, "r")
        meta = h5file.get("Meta")
        self.section_data = json.loads(meta.attrs["section_index"])
        self.meta_data = {}
        self.batchSize = batchSize
        for key in meta.attrs.keys():
            if "index" not in key:
                self.meta_data[key] = meta.attrs[key]
        self.lengthOfDataset = len(list(self.section_data.keys()))

    def getVocabSize(self):
        return self.meta_data["vocab_size"]

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
            mel = torch.from_numpy(sectionGroup["mel"][...])
            src_pad_mask = torch.full((mel.shape[0],), False, dtype=torch.bool)
            tokens = torch.from_numpy(sectionGroup["tokens"][...]).long()
            tokens_in = tokens[:-1]
            tokens_out = tokens[1:]
            tgt_pad_mask = torch.full((tokens_in.shape[0],), False, dtype=torch.bool)
            yield mel, src_pad_mask, tokens_in, tgt_pad_mask, tokens_out

    def __len__(self):
        return math.ceil(self.lengthOfDataset / self.batchSize) + 1


def bucketBatcherSort(data):
    return sorted(data, key=lambda x: x[0].shape[0])


def datasetCollateFn(all_data):
    all_mels = pad_sequence([i[0] for i in all_data])
    src_pad_mask = pad_sequence([i[1] for i in all_data], padding_value=True, batch_first=True)
    all_tokens = pad_sequence([i[2] for i in all_data])
    tgt_pad_mask = pad_sequence([i[3] for i in all_data], padding_value=True, batch_first=True)
    src_mask = generate_square_subsequent_mask(all_mels.shape[0])
    tgt_mask = generate_square_subsequent_mask(all_tokens.shape[0])
    tokens_out = pad_sequence([i[4] for i in all_data], padding_value=0)
    return all_mels, src_mask, src_pad_mask, all_tokens, tgt_mask, tgt_pad_mask, tokens_out


def getDataPipe(datasetLocation, batchSize=10, prefetchSize=None, pinMemory=False):
    dataset = TranscriptionDataset(datasetLocation, batchSize=batchSize)
    pipe = tdi.BucketBatcher(
        dataset,
        batch_size=batchSize,
        sort_key=bucketBatcherSort,
        use_in_batch_shuffle=True,
    )
    pipe = tdi.Collator(pipe, collate_fn=datasetCollateFn)
    if prefetchSize is not None:
        pipe = tdi.Prefetcher(pipe, prefetchSize)
    if pinMemory:
        pipe = tdi.PinMemory(pipe)
    return dataset, pipe


def generate_square_subsequent_mask(sz: int, device='cpu') -> Tensor:
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    return torch.triu(torch.full((sz, sz), True, device=device, dtype=torch.bool), diagonal=1)


def test():
    datasetLocation = "../Trainsets/S_Tier_1695428558_mTokens1000_mNoS60.hdf5"
    tokenizer = H5GuitarTokenizer(datasetLocation)
    dataset, pipe = getDataPipe(datasetLocation, 10)
    train_dl = DataLoader(dataset=pipe, num_workers=0)
    for i in tqdm(train_dl, total=len(dataset)):
        # pprint([(j.shape, j.dtype) for j in i])
        continue


if __name__ == '__main__':
    test()
