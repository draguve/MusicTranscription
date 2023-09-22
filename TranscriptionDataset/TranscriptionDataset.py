import numpy as np
import torch
import torchdata
import torchdata.datapipes.iter as tdi
import h5py
import json
from Tokenizer.loaderH5 import H5GuitarTokenizer


class TranscriptionDataset(torchdata.datapipes.map.MapDataPipe):
    def __init__(self, dataset_file):
        self.h5file = h5py.File(dataset_file, "r")
        data = json.loads(self.h5file.get("Meta").attrs["section_index"])
        self.keys = np.array(list(data.keys()))
        self.lengthOfDataset = len(self.keys)

    def __getitem__(self, item):
        sectionGroup = self.h5file[f"/Songs/{self.keys[item]}"]
        mel = sectionGroup["mel"][...]
        tokens = sectionGroup["tokens"][...]
        return mel, tokens

    def __len__(self):
        return self.lengthOfDataset


def bucketBatcherSort(data):
    return sorted(data, key=lambda x: x[0].shape[0])


def datasetCollateFn(all_data):
    max_mel_size = max([i[0].shape[0] for i in all_data])
    max_token_size = max([i[1].shape[0] for i in all_data])
    all_mels = []
    all_tokens = []
    all_src_masks = []
    all_tgt_masks = []
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
    return np.stack(all_mels), np.stack(all_src_masks), np.stack(all_tokens), np.stack(all_tgt_masks)


def getDataPipe(datasetLocation, batchSize=10, shuffleBufferSize=1000, postBucketShuffle=100, prefetchSize=None,
                pinMemory=False):
    pipe = TranscriptionDataset(datasetLocation)
    pipe = pipe.to_iter_datapipe().shuffle(buffer_size=shuffleBufferSize)
    pipe = tdi.BucketBatcher(
        pipe,
        batch_size=batchSize,
        sort_key=bucketBatcherSort,
        use_in_batch_shuffle=True,
    )
    pipe = pipe.shuffle(buffer_size=postBucketShuffle)
    pipe = tdi.Collator(pipe, collate_fn=datasetCollateFn)
    if prefetchSize is not None:
        pipe = tdi.Prefetcher(pipe, prefetchSize)
    if pinMemory:
        pipe = tdi.PinMemory(pipe)
    return pipe


def test():
    dataset = "../Trainsets/S_Tier_1695289757_mTokens1000_mNoS60.hdf5"
    tokenizer = H5GuitarTokenizer(dataset)
    pipe = getDataPipe(dataset, 10)
    for data in pipe:
        print([i.shape for i in data])
        for x in data[2]:
            for y in x:
                print(tokenizer.encoder.decode(y))
        break


if __name__ == '__main__':
    test()
