import json
from functools import partial
from typing import Iterator, Callable
from itertools import chain, cycle
import h5py
import numpy as np
import torch
import torchaudio
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.utils.data.dataset import T_co
import math
import tqdm
import torch.nn.functional as F
from TUtils import ArrangementUtils
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
import numpy


class OneHotEncodeArrangement(Callable):
    def __init__(self, datasetfile):
        all_tunings = {}
        h5file = h5py.File(datasetfile, "r")
        songsGroup = h5file["Songs"]
        data = list(json.loads(songsGroup.attrs["index"]).values())
        tunings = None
        capos = None
        arrangements = None
        for item in tqdm.tqdm(data, desc="Creating OneHotEncodings"):
            songGroup = h5file[f"/Songs/{item['group']}"]
            tuning = str(songGroup["tuning"][0:])
            if tunings is None:
                tunings = songGroup["tuning"][0:]
                capos = np.array([int(songGroup.attrs["capo"])])
                thisArrangement = [int(ArrangementUtils.arrangementIndex[arr]) + 3 for arr in
                                   songGroup.attrs["allArrangements"]]
                arrangements = np.zeros(6)
                arrangements[thisArrangement] = 1
            else:
                tunings = np.vstack((tunings, songGroup["tuning"][0:]))
                capo = np.array([int(songGroup.attrs["capo"])])
                capos = np.vstack((capos, capo))
                thisArrangement = [int(ArrangementUtils.arrangementIndex[arr]) + 3 for arr in
                                   songGroup.attrs["allArrangements"]]
                arrangement = np.zeros(6)
                arrangement[thisArrangement] = 1
                arrangements = np.vstack((arrangements, arrangement))
            if tuning in all_tunings:
                all_tunings[tuning] += 1
            else:
                all_tunings[tuning] = 1
        self.oheTunings = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(tunings)
        self.oheCaps = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(capos)
        self.oheArrangements = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(arrangements)
        self.tuning_output_size = sum(self.oheTunings._n_features_outs)
        self.capo_output_size = sum(self.oheCaps._n_features_outs)
        self.arrangement_output_size = sum(self.oheArrangements._n_features_outs)

    def __call__(self, tunings, capo, arrangement):
        return self.oheTunings.transform(np.expand_dims(tunings, 0)), self.oheCaps.transform(
            np.expand_dims(capo, 0)), self.oheArrangements.transform(np.expand_dims(arrangement, 0))


class ArrangementDataset(IterableDataset):
    def __init__(self, filename, sampleRate=44100, timeInSeconds=60.0, oneHotTransform=None):
        self.h5file = None
        self.file_number_samples_to_read = None
        self.data = None
        self.filename = filename
        self.sample_rate = sampleRate
        self.timeInSeconds = timeInSeconds
        self.oneHotTransform = oneHotTransform
        self.file_number_samples_to_read = int(self.timeInSeconds * self.sample_rate)
        # worker_info = torch.utils.data.get_worker_info()
        # if worker_info is None:
        #     self.start()

    def start(self):
        self.h5file = h5py.File(self.filename, "r")

    def get_all_data(self):
        h5file = h5py.File(self.filename, "r")
        songsGroup = h5file["Songs"]
        return list(json.loads(songsGroup.attrs["index"]).values())

    def load_all_data(self):
        self.data = self.get_all_data()

    def process_file(self, indexItem):
        songGroup = self.h5file[f"/Songs/{indexItem['group']}"]
        tuning = torch.Tensor(songGroup["tuning"][0:])
        info = torchaudio.info(indexItem["ogg"])
        file_sample_rate = info.sample_rate
        waveform, sample_rate = torchaudio.load(indexItem["ogg"], normalize=True)
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, orig_freq=file_sample_rate, new_freq=self.sample_rate)
            sample_rate = self.sample_rate
        thisArrangement = [int(ArrangementUtils.arrangementIndex[arr]) + 3 for arr in
                           songGroup.attrs["allArrangements"]]
        arrangement = np.zeros(6)
        arrangement[thisArrangement] = 1
        capo = np.array([int(songGroup.attrs["capo"])])
        if self.oneHotTransform:
            tuning, capo, arrangement = self.oneHotTransform(tuning, capo, arrangement)
        allSections = torch.split(waveform, self.file_number_samples_to_read, dim=1)
        for section in allSections:
            if section.size(1) != self.file_number_samples_to_read:
                section = F.pad(section, (self.file_number_samples_to_read - section.size(1), 0))
            yield section, tuning, capo, arrangement

    def get_stream(self, list_data):
        return chain.from_iterable(map(self.process_file, chain(list_data)))

    def __iter__(self) -> Iterator[T_co]:
        return self.get_stream(self.data)


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.start()
    # print(f"total {len(dataset.data)}")
    per_worker = int(math.ceil((len(dataset.data)) / float(worker_info.num_workers)))
    dataset.data = dataset.data[worker_id * per_worker:min((worker_id + 1) * per_worker, len(dataset.data))]
    # print(f"worker {len(dataset.data)} {worker_id}")


# def guitarCollate(batch):
#     pad_token = batch[0][3]
#     spec_batch = torch.stack([d[0] for d in batch])
#     tuning_batch = torch.stack([d[1] for d in batch])
#     padded_tokens = pad_sequence([d[2] for d in batch], padding_value=pad_token)
#     return spec_batch, tuning_batch, padded_tokens.permute(1, 0)

class ArrangementDataModule(pl.LightningDataModule):

    def __init__(self, location, batch_size=2, sample_rate=16000, num_workers=0, test_size=0.3):
        super().__init__()
        self.location = location
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.dataset_train = None
        self.dataset_test = None
        self.num_workers = num_workers
        self.test_size = test_size
        self.transform = OneHotEncodeArrangement(self.location)

    def setup(self, stage):
        self.dataset_train = ArrangementDataset(self.location, sampleRate=self.sample_rate,
                                                oneHotTransform=self.transform)
        total_set = numpy.array(self.dataset_train.get_all_data())
        train, test = train_test_split(total_set, test_size=self.test_size)
        self.dataset_train.data = train
        self.dataset_test = ArrangementDataset(self.location, sampleRate=self.sample_rate,
                                               oneHotTransform=self.transform)
        self.dataset_test.data = test

    def train_dataloader(self):
        if self.num_workers == 0:
            self.dataset_train.start()
        return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers,
                          worker_init_fn=worker_init_fn)

    def val_dataloader(self):
        if self.num_workers == 0:
            self.dataset_test.start()
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers,
                          worker_init_fn=worker_init_fn)


if __name__ == '__main__':
    SAMPLE_RATE = 44100
    dataset = "../Trainsets/massive_test2.hdf5"
    module = ArrangementDataModule(dataset, batch_size=2, num_workers=2)
    module.setup("")
    dataiter = iter(module.train_dataloader())
    for item in dataiter:
        for elem in item:
            print(elem.shape)
        # break
    # check = next(dataiter)
    # print(check)
