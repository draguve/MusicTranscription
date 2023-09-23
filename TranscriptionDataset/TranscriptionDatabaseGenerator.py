# import Tokenizer
import json

from matplotlib import pyplot as plt
from pathos.multiprocessing import ProcessPool

from TUtils import get_all_dlc_files, get_timestamp, random_string
import librosa
import numpy as np
from tqdm import tqdm
import h5py

from Tokenizer import GuitarTokenizer

DLC_Location = r"E:\Rocksmith - Copy\Downloads"
TRAINSETS_LOCATION = "../Trainsets"
train_set_name = "S_Tier"
MAX_NO_OF_SECONDS = 60
TIMESTEPS_PER_SECOND = 50
MAX_TOKENS_PER_SECTION = 1000
SAMPLE_RATE = 16000
N_FFTS = 2048
HOP_LENGTH = 512
N_MELS = 512
SPECTROGRAM = False


def store_dlc(fileLocations):
    try:
        dlc_id = random_string(10)
        tokenizer = GuitarTokenizer(
            maxNumberOfSeconds=MAX_NO_OF_SECONDS,
            timeStepsPerSecond=TIMESTEPS_PER_SECOND,
            maxNumberOfTokensPerSection=MAX_TOKENS_PER_SECTION,
            sample_rate=SAMPLE_RATE,
            n_ffts=N_FFTS,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
        )
        sections, parsedSong = tokenizer.getTokensAndSpectrogram(fileLocations, spectrogram=SPECTROGRAM)
        if sections is None:
            return None, None, fileLocations
    except Exception:
        return None, None, fileLocations
    return dlc_id, sections, fileLocations


def main():
    could_not_add = []
    all_dlcs = get_all_dlc_files(DLC_Location)
    tokenizer = GuitarTokenizer(
        maxNumberOfSeconds=MAX_NO_OF_SECONDS,
        timeStepsPerSecond=TIMESTEPS_PER_SECOND,
        maxNumberOfTokensPerSection=MAX_TOKENS_PER_SECTION,
        sample_rate=SAMPLE_RATE,
        n_ffts=N_FFTS,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    with h5py.File(
            f'{TRAINSETS_LOCATION}/{train_set_name}_{get_timestamp()}_mTokens{MAX_TOKENS_PER_SECTION}_mNoS{MAX_NO_OF_SECONDS}.hdf5',
            'w') as f:
        songs = f.create_group("Songs")
        processPool = ProcessPool(nodes=8)
        results = processPool.imap(store_dlc, all_dlcs)
        section_index = {}
        song_index = {}
        cfs = np.zeros(tokenizer.numberOfTokens(), dtype=int)
        for result in tqdm(results, desc="Generating Metas", total=len(all_dlcs)):
            dlc_id, sections, fileLocations = result
            if sections is None:
                could_not_add.append(fileLocations["ogg"])
                continue
            for i, section in enumerate(sections):
                thisSection = songs.create_group(f"{dlc_id}_{i}")
                tokens = np.array(section.tokens)
                this_cfs = np.bincount(tokens, minlength=tokenizer.numberOfTokens())
                cfs = cfs + this_cfs
                thisSection.create_dataset("tokens", data=tokens)
                if SPECTROGRAM:
                    spectrogram = np.transpose(section.spectrogram, (1, 0))
                    thisSection.create_dataset("mel", data=spectrogram)
                    thisSection.attrs["melShape"] = section.spectrogram.shape
                    section_index[f"{dlc_id}_{i}"]["melShape"] = section.spectrogram.shape
                thisSection.attrs["startSeconds"] = section.startSeconds
                thisSection.attrs["stopSeconds"] = section.stopSeconds
                thisSection.attrs["numTokens"] = len(section.tokens)
                thisSection.attrs["timeInSeconds"] = section.stopSeconds - section.startSeconds
                section_index[f"{dlc_id}_{i}"] = {}
                section_index[f"{dlc_id}_{i}"]["startSeconds"] = section.startSeconds
                section_index[f"{dlc_id}_{i}"]["stopSeconds"] = section.stopSeconds
                section_index[f"{dlc_id}_{i}"]["numTokens"] = len(section.tokens)
                section_index[f"{dlc_id}_{i}"]["timeInSeconds"] = section.stopSeconds - section.startSeconds
            song_index[f"{dlc_id}"] = json.dumps(fileLocations)
        meta_group = f.create_group("Meta")
        meta_group.create_dataset("cfs", data=cfs)
        meta_group.attrs["section_index"] = json.dumps(section_index)
        meta_group.attrs["song_index"] = json.dumps(song_index)
        meta_group.attrs["maxNumberOfSeconds"] = MAX_NO_OF_SECONDS
        meta_group.attrs["timeStepsPerSecond"] = TIMESTEPS_PER_SECOND
        meta_group.attrs["maxNumberOfTokensPerSection"] = MAX_TOKENS_PER_SECTION
        meta_group.attrs["sample_rate"] = SAMPLE_RATE
        meta_group.attrs["n_ffts"] = N_FFTS
        meta_group.attrs["hop_length"] = HOP_LENGTH
        meta_group.attrs["n_mels"] = N_MELS
        meta_group.attrs["vocab_size"] = tokenizer.numberOfTokens()
        # meta_group.attrs["notAdded"] = could_not_add
        meta_group.attrs["failedAddLen"] = len(could_not_add)


if __name__ == '__main__':
    main()
