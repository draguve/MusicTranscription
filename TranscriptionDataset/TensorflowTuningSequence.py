import h5py
import numpy as np
import tensorflow as tf
import json

arrangementsToConvert = ["lead", "lead2", "lead3", "rhythm", "rhythm2", "rhythm3"]
arrangementIndex = {x: index - (len(arrangementsToConvert) / 2) for index, x in enumerate(arrangementsToConvert)}

class TuningSequence(tf.keras.utils.Sequence):
    def __init__(self, dataset_file,shuffle=True):
        self.shuffle = shuffle
        self.dataset_file = dataset_file
        h5file = h5py.File(dataset_file, "r")
        songsGroup = h5file["Songs"]
        melGroup = h5file["MelSpectrograms"]
        allFiles = list(melGroup.keys())
        allMetas = list(songsGroup.keys())
        keysToSongId = {}
        arrangements = None
        tunings = None
        songs = []

        self.h5file = h5file
        for key in allMetas:
            songGroup = self.h5file[f"/Songs/{key}"]
            thisArrangement = [int(arrangementIndex[songGroup.attrs["arrangement"].lower()]) + 3]
            arrangement = np.zeros(6, np.single)
            arrangement[thisArrangement] = 1
            keysToSongId[key] = songGroup.attrs["songId"]
            songs.append(key)

            if tunings is None:
                tunings = songGroup["tuning"]
            else:
                tunings = np.vstack((tunings,songGroup["tuning"]))
            if arrangements is None:
                arrangements = arrangement
            else:
                arrangements = np.vstack((arrangements,arrangement))

        self.songs = songs
        self.arrangements = arrangements
        self.tunings = tunings
        self.h5file = h5file
        self.indexes = np.arange(len(self.songs))
        self.keysToSongId = keysToSongId
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return len(self.songs)
        # return math.ceil(len(self.data) / self.batch_size)

    def get_expected_output(self):
        return self.arrangements

    def __getitem__(self, idx):
        shuffled_idx = self.indexes[idx]
        song_key = self.songs[shuffled_idx]
        mel_id = self.keysToSongId[song_key]
        melGroup = self.h5file[f"/MelSpectrograms/{mel_id}"]
        melSpectrogram = melGroup["mel"]
        arrangement = self.arrangements[shuffled_idx]
        tuning = self.tunings[shuffled_idx]
        assert melSpectrogram.shape[0] == 256
        tuning = [
            np.array([tuning[0]]),
            np.array([tuning[1]]),
            np.array([tuning[2]]),
            np.array([tuning[3]]),
            np.array([tuning[4]]),
            np.array([tuning[5]]),
        ]
        return [np.expand_dims(np.swapaxes(melSpectrogram[:],0,1),0),arrangement],tuning

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

def main():
    testSequence = TuningSequence("../Trainsets/S_Tier_Mel.hdf5")
    S = testSequence[0]
    print(S)
    # print(arrangement[0].shape)
    # fig, ax = plt.subplots()
    # S_dB = librosa.power_to_db(S, ref=np.max)
    # img = librosa.display.specshow(S_dB, x_axis='time',
    #                                y_axis='mel', sr=16000, ax=ax)
    # fig.colorbar(img, ax=ax, format='%+2.0f dB')
    # ax.set(title='Mel-frequency spectrogram')
    # plt.show()


if __name__ == '__main__':
    main()