import json
from functools import partial
from typing import Iterator
from itertools import chain, cycle
import h5py
import torch
import torchaudio
from sortedcontainers import SortedDict
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import T_co
import math


class ArrangementDataset(IterableDataset):
    def __init__(self, filename, sampleRate=44100,timeInSeconds=60.0):
        self.h5file = None
        self.file_number_samples_to_read = None
        self.data = None
        self.filename = filename
        self.sample_rate = sampleRate
        self.timeInSeconds = 60.0

    def start(self):
        self.h5file = h5py.File(self.filename, "r")
        songsGroup = self.h5file["Songs"]
        self.data = list(json.loads(songsGroup.attrs["index"]).values())
        self.file_number_samples_to_read = int(self.timeInSeconds * self.sample_rate)

    def process_file(self, indexItem):
        songGroup = self.h5file[f"/Songs/{indexItem['group']}"]
        tuning = torch.Tensor(songGroup["tuning"][0:])
        tuningAndArrangement = torch.cat((tuning, torch.Tensor([songGroup.attrs["arrangementIndex"]]),
                                          torch.Tensor([float(songGroup.attrs["capo"])])))
        info = torchaudio.info(indexItem["ogg"])
        file_sample_rate = info.sample_rate
        waveform, sample_rate = torchaudio.load(indexItem["ogg"], normalize=True)
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, orig_freq=file_sample_rate, new_freq=self.sample_rate)
            sample_rate = self.sample_rate
        allSections = torch.split(waveform, self.file_number_samples_to_read, dim=1)
        yield allSections[indexInSection], tuningAndArrangement, tokens, self.pad_token

    def get_stream(self, list_data):
        return chain.from_iterable(map(self.process_file, cycle(list_data)))

    def __iter__(self) -> Iterator[T_co]:
        return self.get_stream(self.data)


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.start()
    per_worker = int(math.ceil((len(dataset.data)) / float(worker_info.num_workers)))
    dataset.data = dataset.data[worker_id * per_worker:min(worker_id * (per_worker + 1), len(dataset.data))]


def guitarCollate(batch):
    pad_token = batch[0][3]
    spec_batch = torch.stack([d[0] for d in batch])
    tuning_batch = torch.stack([d[1] for d in batch])
    padded_tokens = pad_sequence([d[2] for d in batch], padding_value=pad_token)
    return spec_batch, tuning_batch, padded_tokens.permute(1, 0)


SAMPLE_RATE = 44100

if __name__ == '__main__':
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=2048,
        hop_length=512,
        n_mels=128
    )
    dataset = ArrangementDataset("../Trainsets/massive.hdf5", sampleRate=SAMPLE_RATE)
    pad_token = dataset.pad_token
    loader = DataLoader(dataset=dataset, batch_size=64,
                        collate_fn=guitarCollate,
                        worker_init_fn=worker_init_fn,
                        num_workers=10)  # , num_workers=4)
    dataiter = iter(loader)
    for item in dataiter:
        print(item)
    # check = next(dataiter)
    # print(check)
