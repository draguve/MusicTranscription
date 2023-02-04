import json

import h5py
import torch
import torchaudio
from sortedcontainers import SortedDict
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class SongDataset(Dataset):
    def __init__(self, filename, sampleRate=44100):
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

    def __getitem__(self, index):
        songIndex = self.sortedKeys[self.sortedIndex.bisect_right(index) - 1]
        song = self.sortedIndex[songIndex]
        sectionIndex = index - song["startIndex"]
        # songGroup = self.h5file[f"/Songs/{song['group']}"]
        h5file = h5py.File(self.filename, "r")
        songsGroup = h5file["Songs"]
        songGroup = songsGroup[song["group"]]
        # print(song)
        info = torchaudio.info(song["ogg"])
        file_sample_rate = info.sample_rate
        file_start_offset = int(songGroup["startSeconds"][sectionIndex] * file_sample_rate)
        file_number_samples_to_read = int(self.spectrogramSizeInSeconds * file_sample_rate)
        waveform, sample_rate = torchaudio.load(song["ogg"], normalize=True, frame_offset=file_start_offset,
                                                num_frames=file_number_samples_to_read)
        if waveform.size(1) != file_number_samples_to_read:
            # print(song, index, sectionIndex)
            return None, None, None
        if waveform.size(0) != 2:
            return None, None, None
            # raise Exception("Read Less than expected")
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, orig_freq=file_sample_rate, new_freq=self.sample_rate)
        # output = self.transformation(waveform)
        # padding = torch.full((self.maxTokens - len(songGroup["tokens"][sectionIndex]),), int(self.pad_token))
        tokens = torch.from_numpy(songGroup["tokens"][sectionIndex]).type(torch.LongTensor)
        # tokens = torch.cat((tokens, padding), )
        # print(output.shape)
        tuning = torch.Tensor(songGroup["tuning"][0:])
        tuningAndArrangement = torch.cat((tuning, torch.Tensor([songGroup.attrs["arrangementIndex"]]),
                                          torch.Tensor([float(songGroup.attrs["capo"])])))
        return waveform, tuningAndArrangement, tokens

    def __len__(self):
        return self.size


class GuitarCollater(object):
    def __init__(self, pad_token):
        self.pad_toke = pad_token

    def __call__(self, batch):
        batch_filter = list(filter(lambda x: x[0] is not None, batch))
        spec_batch = torch.stack([d[0] for d in batch_filter])
        tuning_batch = torch.stack([d[1] for d in batch_filter])
        padded_tokens = pad_sequence([d[2] for d in batch_filter], padding_value=self.pad_toke)
        return spec_batch, tuning_batch, padded_tokens.permute(1, 0)


SAMPLE_RATE = 44100

if __name__ == '__main__':
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=2048,
        hop_length=512,
        n_mels=128
    )
    dataset = SongDataset("../Trainsets/massive_test.hdf5", sampleRate=SAMPLE_RATE)
    print(dataset[0])
    # loader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=4)
    # dataiter = iter(loader)
    # check = next(dataiter)
    # print(check)
