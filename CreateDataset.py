import json

import h5py
import joblib
import numpy as np
import sortedcontainers
from tqdm import tqdm

from TUtils.ArrangementUtils import arrangementIndex,arrangementsToConvert
from TUtils import get_all_dlc_files, tqdm_joblib
from Tokenizer import GuitarTokenizer
from Tokenizer import SongXMLParser
from TUtils import random_string
from pathos.multiprocessing import ProcessPool
import librosa


SpectrogramSizeInSeconds = 1.0
NumberOfTimeTokensPerSecond = 1000
remove_all_silence = True
generate_and_store_mel = True


#private
toRemoveForStore = ["notes", "chords", "ebeats", "chordTemplates", "phraseIterations", "sections", "anchors",
                    "handShapes"]

def store_dlc(typeOfArrangement, fileLocations,songId):
    guitarTokenizer = GuitarTokenizer(SpectrogramSizeInSeconds, NumberOfTimeTokensPerSecond)
    try:
        if "rs2dlc" in fileLocations:
            with open(fileLocations["rs2dlc"]) as user_file:
                parsed_json = json.load(user_file)
                dlcKey = parsed_json["DLCKey"]
        else:
            dlcKey = random_string(20)
        parsedSong = SongXMLParser.parse_xml_file(fileLocations[typeOfArrangement])
        random_id = random_string(10)
        group_name = f"{dlcKey}_{typeOfArrangement}_{random_id}"
        # songGroup = songGroup.create_group(group_name)
        songGroup = {}
        songSections = guitarTokenizer.convertSongFromParsedFile(parsedSong)
        if remove_all_silence:
            songSections = list(filter(lambda section: len(section.tokens) > 4, songSections))
        numberOfSection = len(songSections)
        startSections = np.array([section.startSeconds for section in songSections])
        endSections = np.array([section.stopSeconds for section in songSections])
        ebeatsTimings = np.array([float(section["time"]) for section in parsedSong["ebeats"]])
        ebeatsMeasureStarts = np.array([section for section in range(len(parsedSong["ebeats"])) if 'measure' in parsedSong["ebeats"][section]])
        songGroup["startSeconds"] = startSections
        songGroup["endSeconds"] = endSections
        songGroup["tuning"] = [int(offset) for offset in
                               sortedcontainers.SortedDict(parsedSong["tuning"]).values()]
        songGroup["ebeatsTimings"] = ebeatsTimings
        songGroup["ebeatsMeasureStarts"] = ebeatsMeasureStarts
        songGroup["attrs"] = {}
        songGroup["attrs"]["arrangementIndex"] = arrangementIndex[typeOfArrangement]
        songGroup["attrs"]["arrangement"] = typeOfArrangement
        songGroup["attrs"]["allArrangements"] = np.intersect1d(arrangementsToConvert,list(fileLocations.keys())).tolist()
        # dt = h5py.vlen_dtype(np.dtype('int32'))
        # tokensStore = songGroup.create_dataset("tokens", len(startSections), dtype=dt)
        # for i in range(len(startSections)):
        #     tokensStore[i] = np.array(songSections[i].tokens)
        #     maxNumberOfTokens = max(maxNumberOfTokens, len(tokensStore[i]))
        tokensStore = []
        for i in range(len(startSections)):
            tokensStore.append(np.array(songSections[i].tokens))
        songGroup["tokens"] = tokensStore
        dataToStore = parsedSong.copy()
        for key in toRemoveForStore:
            del dataToStore[key]
        info_to_store = {
            "group": group_name,
            # "startIndex": lastAdded,
            "len": numberOfSection,
            "typeOfArrangement": typeOfArrangement,
            "ogg": fileLocations["ogg"],
            "numberOfSections": numberOfSection,
            "songId":songId
        }
        # sortedDlcs[lastAdded] = {"group": group_name,
        #                          "startIndex": lastAdded,
        #                          "len": numberOfSection,
        #                          "typeOfArrangement": typeOfArrangement,
        #                          "ogg": fileLocations["ogg"]}
        for item in info_to_store.keys():
            songGroup["attrs"][item] = info_to_store[item]
        for item in dataToStore.keys():
            songGroup["attrs"][item] = str(dataToStore[item])


        return songGroup
    except Exception as e:
        print(e)
        print(f"could not parse {fileLocations}")
        return None

def generate_mel(id,ogg):
    songData, sr = librosa.load(ogg, sr=16000, mono=False)
    mel = librosa.feature.melspectrogram(y=songData, sr=sr, n_mels=128, fmax=8000)
    assert len(mel.shape) == 3
    mel = mel.reshape(256, mel.shape[-1])
    return id,ogg,mel

if __name__ == '__main__':
    dlcs = get_all_dlc_files(r"Downloads2")
    # creating a file
    with h5py.File('Trainsets/S_Tier4.hdf5', 'w') as f:
        processPool = ProcessPool(nodes = 8)

        all_keys = []
        all_dlc = []
        song_ids = []

        all_dlc_mel_id = []
        all_dlc_ogg = []
        for dlc in dlcs:
            id = random_string()

            if generate_and_store_mel:
                all_dlc_mel_id.append(id)
                all_dlc_ogg.append(dlc["ogg"])

            #for each arrangement in the dlc
            for key in dlc:
                if key in arrangementIndex:
                    all_keys.append(key)
                    all_dlc.append(dlc)
                    song_ids.append(id)

        results = processPool.imap(store_dlc,all_keys,all_dlc,song_ids)

        sortedDlcs = sortedcontainers.SortedDict()
        last_added = 0
        songs = f.create_group("Songs")
        maxNumberOfTokens = 0

        for result in tqdm(results,desc="Generating Metas",total=len(all_dlc)):
            if result is None:
                continue
            songGroup = songs.create_group(result["attrs"]["group"])
            songGroup.create_dataset("startSeconds", data=result["startSeconds"])
            songGroup.create_dataset("endSeconds", data=result["endSeconds"])
            songGroup.create_dataset("tuning", data=result["tuning"])
            songGroup.create_dataset("ebeatsTimings", data=result["ebeatsTimings"])
            songGroup.create_dataset("ebeatsMeasureStarts", data=result["ebeatsMeasureStarts"])
            for item in result["attrs"].keys():
                songGroup.attrs[item] = result["attrs"][item]
            songGroup.attrs["startIndex"] = last_added
            sortedDlcsItem = {
                "group": result["attrs"]["group"],
                "startIndex": last_added,
                "len": result["attrs"]["len"],
                "typeOfArrangement": result["attrs"]["typeOfArrangement"],
                "ogg": result["attrs"]["ogg"],
                "numberOfSections": result["attrs"]["numberOfSections"]
            }
            sortedDlcs[last_added] = sortedDlcsItem
            dt = h5py.vlen_dtype(np.dtype('int32'))
            tokensStore = songGroup.create_dataset("tokens", len(result["tokens"]), dtype=dt)
            for i in range(len(result["tokens"])):
                tokensStore[i] = result["tokens"][i]
                maxNumberOfTokens = max(maxNumberOfTokens, len(tokensStore[i]))
            last_added += result["attrs"]["len"]
        songs.attrs["index"] = json.dumps(sortedDlcs)
        songs.attrs["totalSize"] = last_added
        songs.attrs["maxTokens"] = maxNumberOfTokens
        songs.attrs["spectrogramSizeInSeconds"] = SpectrogramSizeInSeconds
        songs.attrs["numberOfTimeTokensPerSecond"] = NumberOfTimeTokensPerSecond
        # This is without a pad token
        guitarTokenizer = GuitarTokenizer(SpectrogramSizeInSeconds, NumberOfTimeTokensPerSecond)
        songs.attrs["vocabSize"] = guitarTokenizer.numberOfTokens()

        if generate_and_store_mel:
            mels = f.create_group("MelSpectrograms")
            results = processPool.imap(generate_mel, all_dlc_mel_id, all_dlc_ogg)
            for song_id,ogg,mel in tqdm(results, desc="Generating Mels", total=len(all_dlc_mel_id)):
                thisMel = mels.create_group(song_id)
                thisMel.attrs["id"] = song_id
                thisMel.attrs["ogg"] = ogg
                thisMel.create_dataset("mel",data=mel,compression='gzip', compression_opts=9)
                #maybe add the arrangementIndex Here
