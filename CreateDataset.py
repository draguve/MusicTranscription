from TUtils import get_all_dlc_files
from Tokenizer import GuitarTokenizer
import h5py
from Tokenizer import SongXMLParser
from pprint import pprint
import json
import sortedcontainers
import numpy as np
from tqdm import tqdm

SpectrogramSizeInSeconds = 1.0
NumberOfTimeTokensPerSecond = 1000

maxNumberOfTokens = 0

toRemoveForStore = ["notes", "chords", "ebeats", "chordTemplates", "phraseIterations", "sections", "anchors",
                    "handShapes"]


def store_dlc(lastAdded, dlcKey, songGroup, guitarTokenizer, typeOfArrangement, fileLocations):
    global maxNumberOfTokens
    parsedSong = SongXMLParser.parse_xml_file(fileLocations[typeOfArrangement])
    # TODO: removing all now e standard songs for now fix later
    for string in parsedSong["tuning"].keys():
        if parsedSong["tuning"][string] != "0":
            return 0
    group_name = f"{dlcKey}_{typeOfArrangement}"
    songGroup = songGroup.create_group(group_name)
    songSections = guitarTokenizer.convertSongFromParsedFile(parsedSong)
    numberOfSection = len(songSections)
    startSections = np.array([section.startSeconds for section in songSections])
    endSections = np.array([section.stopSeconds for section in songSections])
    songGroup.create_dataset("startSeconds", data=startSections)
    songGroup.create_dataset("endSeconds", data=endSections)
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
    dlcs = get_all_dlc_files("Downloads")
    tokenizer = GuitarTokenizer(SpectrogramSizeInSeconds, NumberOfTimeTokensPerSecond)
    # creating a file
    with h5py.File('test.hdf5', 'w') as f:
        songs = f.create_group("Songs")
        pbar = tqdm(total=len(dlcs))
        for dlc in dlcs:
            with open(dlc["rs2dlc"]) as user_file:
                parsed_json = json.load(user_file)
                DLCKey = parsed_json["DLCKey"]
                if "lead" in dlc:
                    last_added += store_dlc(last_added, DLCKey, songs, tokenizer, "lead", dlc)
                # if "rhythm" in dlc:
                #     last_added += store_dlc(last_added, DLCKey, songs, tokenizer, "rhythm")
            pbar.update(1)
        songs.attrs["index"] = json.dumps(sortedDlcs)
        songs.attrs["totalSize"] = last_added
        songs.attrs["maxTokens"] = maxNumberOfTokens
        songs.attrs["spectrogramSizeInSeconds"] = SpectrogramSizeInSeconds
        songs.attrs["numberOfTimeTokensPerSecond"] = NumberOfTimeTokensPerSecond
        # This is without a pad token
        songs.attrs["vocabSize"] = tokenizer.numberOfTokens()
        pbar.close()
