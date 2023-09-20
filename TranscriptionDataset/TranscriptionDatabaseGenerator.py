# import Tokenizer
from matplotlib import pyplot as plt
from TUtils import get_all_dlc_files
import librosa
import numpy as np


def main():
    all_dlcs = get_all_dlc_files("../RSFiles/MiniDataset")
    this_dlc = all_dlcs[0]
    SAMPLE_RATE = 16000
    y, sr = librosa.load(this_dlc["ogg"],sr=SAMPLE_RATE,duration=30)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=512)

    print(S.shape)
    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time',
                                   y_axis='mel', sr=sr,  ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    plt.show()


if __name__ == '__main__':
    main()
