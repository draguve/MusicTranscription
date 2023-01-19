from typing import Any

import Tokenizer
from Tokenizer import GuitarTokenizer
import librosa
from TUtils import get_all_dlc_files
from dataclasses import dataclass
from pprint import pprint


@dataclass
class GuitarSectionAllData:
    songSection: Tokenizer.SongSection
    songSectionSpectrogram: Any


# can only load lead or rhythm
def load_song(dlc, seconds=1, samplesPerSecond=1000):
    if "ogg" not in dlc:
        return None
    tokenizer = GuitarTokenizer(seconds, samplesPerSecond)
    y, sr = librosa.load(dlc["ogg"], mono=False)
    # TODO Need to Handle alt leads and rhythm
    output = {}
    output["sr"] = sr
    if "lead" in dlc:
        output["lead"] = []
        songSections = tokenizer.convertSong(dlc["lead"])
        for section in songSections:
            startSample = int(section.startSeconds * sr)
            endSample = int(startSample + sr * seconds)
            sectionAudio = y[:, startSample:endSample]
            sectionSpec = librosa.feature.melspectrogram(y=sectionAudio, sr=sr)
            output["lead"].append(GuitarSectionAllData(section, sectionSpec))
    if "rhythm" in dlc:
        output["rhythm"] = []
        songSections = tokenizer.convertSong(dlc["rhythm"])
        for section in songSections:
            startSample = int(section.startSeconds * sr)
            endSample = int(startSample + sr * seconds)
            sectionAudio = y[:, startSample:endSample]
            sectionSpec = librosa.feature.melspectrogram(y=sectionAudio, sr=sr)
            output["rhythm"].append(GuitarSectionAllData(section, sectionSpec))
    return output


if __name__ == '__main__':
    dlcs = get_all_dlc_files("../Downloads")
    pprint(load_song(dlcs[4]))
