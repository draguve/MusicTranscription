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
    def __init__(self, dataset_file,shuffle=True):
        self.shuffle = shuffle
        self.dataset_file = dataset_file
        h5file = h5py.File(dataset_file, "r")
        songsGroup = h5file["Songs"]
        melGroup = h5file["MelSpectrograms"]
        allFiles = list(melGroup.keys())
        allArrangements = list(songsGroup.keys())
        idToArrangement = {}
        self.h5file = h5file
        for key in allArrangements:
            songGroup = self.h5file[f"/Songs/{key}"]
            thisArrangement = [int(arrangementIndex[arr]) + 3 for arr in
                               songGroup.attrs["allArrangements"]]
            arrangement = np.zeros(6, np.single)
            arrangement[thisArrangement] = 1
            idToArrangement[songGroup.attrs["songId"]] = arrangement
        arrangements = None
        for key in idToArrangement.keys():
            if arrangements is None:
                arrangements = idToArrangement[key]
            else:
                arrangements = np.vstack((arrangements,idToArrangement[key]))
        self.arrangements = arrangements
        self.allFiles = list(idToArrangement.keys())
        self.h5file = h5file
        self.indexes = np.arange(len(self.allFiles))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return len(self.indexes)
        # return math.ceil(len(self.data) / self.batch_size)

    def get_expected_output(self):
        return self.arrangements

    def __getitem__(self, idx):
        shuffled_idx = self.indexes[idx]
        melGroup = self.h5file[f"/MelSpectrograms/{self.allFiles[shuffled_idx]}"]
        melSpectrogram = melGroup["mel"]
        arrangement = self.arrangements[shuffled_idx]
        assert melSpectrogram.shape[0] == 256
        arrangement = [
            np.array([arrangement[0]]),
            np.array([arrangement[1]]),
            np.array([arrangement[2]]),
            np.array([arrangement[3]]),
            np.array([arrangement[4]]),
            np.array([arrangement[5]]),
        ]
        return np.expand_dims(np.swapaxes(melSpectrogram[:],0,1),0),arrangement

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

def main():
    testSequence = MusicMetaSequence("../Trainsets/S_Tier_Mel.hdf5")
    S,arrangement = testSequence[0]
    print(arrangement[0].shape)
    # fig, ax = plt.subplots()
    # S_dB = librosa.power_to_db(S, ref=np.max)
    # img = librosa.display.specshow(S_dB, x_axis='time',
    #                                y_axis='mel', sr=16000, ax=ax)
    # fig.colorbar(img, ax=ax, format='%+2.0f dB')
    # ax.set(title='Mel-frequency spectrogram')
    # plt.show()


if __name__ == '__main__':
    main()