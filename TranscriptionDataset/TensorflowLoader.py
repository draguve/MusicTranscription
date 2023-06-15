import json
import h5py
import numpy as np
import math
import tensorflow as tf
import tensorflow_io as tfio
import librosa
import matplotlib.pyplot as plt

arrangementsToConvert = ["lead", "lead2", "lead3", "rhythm", "rhythm2", "rhythm3"]
arrangementIndex = {x: index - (len(arrangementsToConvert) / 2) for index, x in enumerate(arrangementsToConvert)}

class MusicMetaSequence(tf.keras.utils.Sequence):

    def __init__(self, dataset_file):
        self.dataset_file = dataset_file
        h5file = h5py.File(dataset_file, "r")
        songsGroup = h5file["Songs"]
        self.data = list(json.loads(songsGroup.attrs["index"]).values())
        self.h5file = h5file

    def __len__(self):
        return math.ceil(len(self.data) / self.batch_size)

    def __getitem__(self, idx):
        songGroup = self.h5file[f"/Songs/{self.data[idx]['group']}"]
        songData,sr = librosa.load(self.data[idx]["ogg"],sr=16000,mono=False)
        mel = librosa.feature.melspectrogram(y=songData, sr=sr, n_mels=128,fmax=8000)
        mel = mel.reshape(256,-1)
        thisArrangement = [int(arrangementIndex[arr]) + 3 for arr in
                           songGroup.attrs["allArrangements"]]
        arrangement = np.zeros(6, np.single)
        arrangement[thisArrangement] = 1
        return mel,arrangement


def main():
    testSequence = MusicMetaSequence("../Trainsets/S_Tier.hdf5")
    S,arrangement = testSequence[0]
    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time',
                                   y_axis='mel', sr=16000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    plt.show()


if __name__ == '__main__':
    main()