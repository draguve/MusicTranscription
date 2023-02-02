import json

import h5py
import numpy as np
import sortedcontainers
from tqdm import tqdm

from SongDataset.ArrangementUtils import arrangementIndex
from TUtils import get_all_dlc_files
from Tokenizer import GuitarTokenizer
from Tokenizer import SongXMLParser
from TUtils import random_string

SpectrogramSizeInSeconds = 1.0
NumberOfTimeTokensPerSecond = 1000
remove_all_silence = True

maxNumberOfTokens = 0

toRemoveForStore = ["notes", "chords", "ebeats", "chordTemplates", "phraseIterations", "sections", "anchors",
                    "handShapes"]


def store_dlc(lastAdded, dlcKey, songGroup, guitarTokenizer, typeOfArrangement, fileLocations):
    global maxNumberOfTokens
    parsedSong = SongXMLParser.parse_xml_file(fileLocations[typeOfArrangement])
    random_id = random_string(10)
    group_name = f"{dlcKey}_{typeOfArrangement}_{random_id}"
    songGroup = songGroup.create_group(group_name)
    songSections = guitarTokenizer.convertSongFromParsedFile(parsedSong)
    if remove_all_silence:
        songSections = list(filter(lambda section: len(section.tokens) > 4, songSections))
    numberOfSection = len(songSections)
    startSections = np.array([section.startSeconds for section in songSections])
    endSections = np.array([section.stopSeconds for section in songSections])
    songGroup.create_dataset("startSeconds", data=startSections)
    songGroup.create_dataset("endSeconds", data=endSections)
    songGroup.create_dataset("tuning", data=[int(offset) for offset in
                                             sortedcontainers.SortedDict(parsedSong["tuning"]).values()])
    songGroup.attrs["arrangementIndex"] = arrangementIndex[typeOfArrangement]
    dt = h5py.vlen_dtype(np.dtype('int32'))
    tokensStore = songGroup.create_dataset("tokens", len(startSections), dtype=dt)
    for i in range(len(startSections)):
        tokensStore[i] = np.array(songSections[i].tokens)
        maxNumberOfTokens = max(maxNumberOfTokens, len(tokensStore[i]))
    dataToStore = parsedSong.copy()
    for key in toRemoveForStore:
        del dataToStore[key]
    sortedDlcs[lastAdded] = {"group": group_name, "startIndex": lastAdded, "len": numberOfSection,
                             "typeOfArrangement": typeOfArrangement,
                             "ogg": fileLocations["ogg"]}
    for item in sortedDlcs[lastAdded].keys():
        songGroup.attrs[item] = sortedDlcs[lastAdded][item]
    for item in dataToStore.keys():
        songGroup.attrs[item] = str(dataToStore[item])
    return numberOfSection


if __name__ == '__main__':
    sortedDlcs = sortedcontainers.SortedDict()
    last_added = 0
    dlcs = get_all_dlc_files(r"Downloads")
    tokenizer = GuitarTokenizer(SpectrogramSizeInSeconds, NumberOfTimeTokensPerSecond)
    # creating a file
    with h5py.File('massive_test.hdf5', 'w') as f:
        songs = f.create_group("Songs")
        pbar = tqdm(total=len(dlcs[0:1000]))
        for dlc in dlcs[0:1000]:
            if "rs2dlc" in dlc:
                with open(dlc["rs2dlc"]) as user_file:
                    parsed_json = json.load(user_file)
                    DLCKey = parsed_json["DLCKey"]
                    for key in dlc:
                        if key in arrangementIndex:
                            last_added += store_dlc(last_added, DLCKey, songs, tokenizer, key, dlc)
            else:
                DLCKey = random_string(20)
                for key in dlc:
                    if key in arrangementIndex:
                        last_added += store_dlc(last_added, DLCKey, songs, tokenizer, key, dlc)
            pbar.update(1)
        songs.attrs["index"] = json.dumps(sortedDlcs)
        songs.attrs["totalSize"] = last_added
        songs.attrs["maxTokens"] = maxNumberOfTokens
        songs.attrs["spectrogramSizeInSeconds"] = SpectrogramSizeInSeconds
        songs.attrs["numberOfTimeTokensPerSecond"] = NumberOfTimeTokensPerSecond
        # This is without a pad token
        songs.attrs["vocabSize"] = tokenizer.numberOfTokens()
        pbar.close()
