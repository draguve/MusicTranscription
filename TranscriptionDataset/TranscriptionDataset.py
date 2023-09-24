import math

import numpy as np
import torch
import torchdata
import torchdata.datapipes.iter as tdi
import h5py
import json
import torch.nn.utils.rnn.pad_sequence
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
            mel = sectionGroup["mel"][...]
            tokens = sectionGroup["tokens"][...]
            yield mel, tokens

    def __len__(self):
        return math.ceil(self.lengthOfDataset / self.batchSize) + 1


def bucketBatcherSort(data):
    return sorted(data, key=lambda x: x[0].shape[0])


def datasetCollateFn(all_data):
    max_mel_size = max([i[0].shape[0] for i in all_data])
    max_token_size = max([i[1].shape[0] for i in all_data])
    all_mels = []
    all_tokens = []
    all_src_masks = []
    all_tgt_masks = []
    src_mask = generate_square_subsequent_mask(max_mel_size)
    tgt_mask = generate_square_subsequent_mask(max_token_size)
    for item in all_data:
        mel, tokens = item
        src_seq_mask = np.zeros(max_mel_size, dtype=bool)
        src_seq_mask[mel.shape[0]:] = True
        all_src_masks.append(src_seq_mask)
        all_mels.append(np.concatenate((mel, np.zeros((max_mel_size - mel.shape[0], mel.shape[1])))))
        tgt_seq_mask = np.zeros(max_token_size, dtype=bool)
        tgt_seq_mask[tokens.shape[0]:] = True
        all_tgt_masks.append(tgt_seq_mask)
        all_tokens.append(np.concatenate((tokens, np.zeros(max_token_size - tokens.shape[0], dtype=tokens.dtype))))
    return np.stack(all_mels), src_mask, np.stack(all_src_masks), np.stack(all_tokens), tgt_mask, np.stack(
        all_tgt_masks)


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
    return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)


def test():
    datasetLocation = "../Trainsets/S_Tier_1695428558_mTokens1000_mNoS60.hdf5"
    tokenizer = H5GuitarTokenizer(datasetLocation)
    dataset, pipe = getDataPipe(datasetLocation, 10)
    train_dl = DataLoader(dataset=pipe)
    for i in tqdm(train_dl, total=len(dataset)):
        print([(j.shape, j.dtype) for j in i])


if __name__ == '__main__':
    test()
