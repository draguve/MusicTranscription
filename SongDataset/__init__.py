import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from pprint import pprint
import json
from sortedcontainers import SortedDict
import torchaudio
from torch.nn.functional import pad
import matplotlib.pyplot as plt


class SongDataset(Dataset):
    def __init__(self, filename, transformation, sampleRate=44100):
        self.filename = filename
        h5file = h5py.File(self.filename, "r")
        songsGroup = h5file["Songs"]
        self.size = songsGroup.attrs["totalSize"]
        self.sortedIndex = SortedDict()
        self.maxTokens = songsGroup.attrs["maxTokens"]
        self.spectrogramSizeInSeconds = songsGroup.attrs["spectrogramSizeInSeconds"]
        self.numberOfTimeTokensPerSecond = songsGroup.attrs["numberOfTimeTokensPerSecond"]
        self.vocabSize = songsGroup.attrs["vocabSize"] + 1
        self.sample_rate = sampleRate
        self.pad_token = songsGroup.attrs["vocabSize"]
        data = json.loads(songsGroup.attrs["index"])
        for key in data.keys():
            self.sortedIndex[int(key)] = data[key]
        self.sortedKeys = self.sortedIndex.keys()
        self.transformation = transformation

    def __getitem__(self, index):
        songIndex = self.sortedKeys[self.sortedIndex.bisect_right(index) - 1]
        song = self.sortedIndex[songIndex]
        sectionIndex = index - song["startIndex"]
        # songGroup = self.h5file[f"/Songs/{song['group']}"]
        h5file = h5py.File(self.filename, "r")
        songsGroup = h5file["Songs"]
        songGroup = songsGroup[song["group"]]
        # print(song)
        file_sample_rate = torchaudio.info(song["ogg"]).sample_rate
        file_start_offset = int(songGroup["startSeconds"][sectionIndex] * file_sample_rate)
        file_number_samples_to_read = int(self.spectrogramSizeInSeconds * file_sample_rate)
        waveform, sample_rate = torchaudio.load(song["ogg"], normalize=True, frame_offset=file_start_offset,
                                                num_frames=file_number_samples_to_read)
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, orig_freq=file_sample_rate, new_freq=self.sample_rate)
        output = self.transformation(waveform)
        padding = torch.full((self.maxTokens - len(songGroup["tokens"][sectionIndex]),), int(self.pad_token))
        tokens = torch.from_numpy(songGroup["tokens"][sectionIndex])
        tokens = torch.cat((tokens, padding), 0)
        print(output.shape)

        # TODO create masks here

        return output, tokens

    def __len__(self):
        return self.size


SAMPLE_RATE = 44100

if __name__ == '__main__':
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=2048,
        hop_length=512,
        n_mels=128
    )
    dataset = SongDataset("../test.hdf5", mel_spectrogram, sampleRate=SAMPLE_RATE)
    # loader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=4)
    # dataiter = iter(loader)
    # check = next(dataiter)
