import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
import torchshape

from SongDataset import SongDataset


class GuitarModel(nn.Module):
    def __init__(self, input_shape):
        super(GuitarModel, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=2,
            out_channels=4,
            kernel_size=(5, 5)
        )
        self.relu = nn.ReLU()
        self.maxPool1 = nn.MaxPool2d(
            kernel_size=(3, 3),
            stride=(3, 3)
        )
        self.conv2 = nn.Conv2d(
            in_channels=4,
            out_channels=10,
            kernel_size=(5, 5)
        )
        self.relu2 = nn.ReLU()
        self.maxPool2 = nn.MaxPool2d(
            kernel_size=(2, 2),
            stride=(2, 2)
        )
        self.conv3 = nn.Conv2d(
            in_channels=10,
            out_channels=25,
            kernel_size=(5, 5)
        )
        self.relu3 = nn.ReLU()
        self.maxPool3 = nn.MaxPool2d(
            kernel_size=(2, 2),
            stride=(2, 2)
        )
        output_shape = torchshape.tensorshape(self.conv1, input_shape)
        output_shape = torchshape.tensorshape(self.maxPool1, output_shape)
        output_shape = torchshape.tensorshape(self.conv2, output_shape)
        output_shape = torchshape.tensorshape(self.maxPool2, output_shape)
        output_shape = torchshape.tensorshape(self.conv3, output_shape)
        output_shape = torchshape.tensorshape(self.maxPool3, output_shape)
        self.flattenedSize = 1
        for i in output_shape:
            self.flattenedSize *= i
        self.fc1 = nn.Linear(in_features=self.flattenedSize + 7, out_features=512)
        self.relu4 = nn.ReLU()

    def forward(self, x, tuningAndArrangement):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxPool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxPool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxPool3(x)
        x = torch.flatten(x)
        x = torch.cat((x, tuningAndArrangement))
        x = self.fc1(x)
        x = self.relu4(x)
        return x


if __name__ == '__main__':
    SAMPLE_RATE = 44100
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=2048,
        hop_length=512,
        n_mels=128
    )
    dataset = SongDataset("../test.hdf5", mel_spectrogram, sampleRate=SAMPLE_RATE)
    model = GuitarModel((1, 2, 128, 87))
    inputSpectrogram = dataset[1][0]
    inputTuningAndArrangement = dataset[1][1]
    print(inputSpectrogram)
    print(inputTuningAndArrangement)
    output = model.forward(inputSpectrogram, inputTuningAndArrangement)
    print(output.shape)

    total_params = sum(
        param.numel() for param in model.parameters()
    )
    print(total_params)
