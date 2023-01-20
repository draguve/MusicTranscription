from TUtils import get_all_dlc_files
from Tokenizer import GuitarTokenizer
import h5py
from Tokenizer import SongXMLParser
from pprint import pprint
import json
import sortedcontainers
import numpy as np
from tqdm import tqdm

SpectogramSizeInSeconds = 1.0
NumberOfTimeTokensPerSecond = 1000


def store_dlc(last_added, DLCKey, songs, tokenizer, type):
    group_name = f"{DLCKey}_{type}"
    songGroup = songs.create_group(group_name)
    parsedSong = SongXMLParser.parse_xml_file(dlc[type])
    songSections = tokenizer.convertSongFromParsedFile(parsedSong)
    numberOfSection = len(songSections)
    startSections = np.array([section.startSeconds for section in songSections])
    endSections = np.array([section.stopSeconds for section in songSections])
    songGroup.create_dataset("startSeconds", data=startSections)
    songGroup.create_dataset("endSeconds", data=endSections)
    dt = h5py.vlen_dtype(np.dtype('int32'))
    tokensStore = songGroup.create_dataset("tokens", len(startSections), dtype=dt)
    for i in range(len(startSections)):
        tokensStore[i] = np.array(songSections[i].tokens)
    sortedDlcs[last_added] = {"group": group_name, "startIndex": last_added, "len": numberOfSection, type: type}
    for item in sortedDlcs[last_added].keys():
        songGroup.attrs[item] = sortedDlcs[last_added][item]
    return numberOfSection


if __name__ == '__main__':
    sortedDlcs = sortedcontainers.SortedDict()
    last_added = 0
    dlcs = get_all_dlc_files("Downloads")
    tokenizer = GuitarTokenizer(SpectogramSizeInSeconds, NumberOfTimeTokensPerSecond)
    # creating a file
    with h5py.File('test.hdf5', 'w') as f:
        songs = f.create_group("Songs")
        pbar = tqdm(total=len(dlcs))
        for dlc in dlcs:
            with open(dlc["rs2dlc"]) as user_file:
                parsed_json = json.load(user_file)
                DLCKey = parsed_json["DLCKey"]
                if "lead" in dlc:
                    last_added += store_dlc(last_added, DLCKey, songs, tokenizer, "lead")
                if "rhythm" in dlc:
                    last_added += store_dlc(last_added, DLCKey, songs, tokenizer, "rhythm")
            pbar.update(1)
        songs.attrs["index"] = json.dumps(sortedDlcs)
        pbar.close()
