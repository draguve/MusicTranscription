import numpy as np

from Tokenizer.EventHandler import Handler, EventVariable, generateTimeRange
from Tokenizer.Encoder import Encoder
from Tokenizer.SongXMLParser import parse_xml_file
from dataclasses import dataclass
import sortedcontainers
from pprint import pprint
from TUtils import get_all_dlc_files
import librosa


def mergeDicts(dict1, dict2):
    """
    Here Dict2 takes precedent over Dict1
    @param dict1:
    @param dict2:
    @return:
    """
    res = {**dict1, **dict2}
    return res


startOfSequenceEventHandler = Handler(
    "sos",
    [
        EventVariable("SOS", 0, 0)
    ]
)


def createStartOfSeqEvent():
    return "sos", [0]


silenceEventHandler = Handler(
    "silence",  # End of Sequence
    [
        EventVariable("silence", 0, 0)
    ]
)

arrangementEventHandler = Handler(
    "arrangement",
    [
        EventVariable("guitarType", 0, 2),
        EventVariable("specialization", 0, 2)
    ]
)

validArrangementList = ["bass", "bass2", "bass3", "lead", "lead2", "lead3", "rhythm", "rhythm2", "rhythm3"]
validArrangementDict = {x: index - (len(validArrangementList) / 2) for index, x in enumerate(validArrangementList)}
arrangementTypeToInt = {
    "bass": 0,
    "lead": 1,
    "rhythm": 2,
}


def createArrangementEvent(arrangementType: str):
    if arrangementType not in validArrangementDict:
        raise Exception("Cannot parse this type of arrangement")
    if arrangementType[-1].isnumeric():
        specialization = int(arrangementType[-1])
    else:
        specialization = 0
    guitarType = ''.join([i for i in arrangementType if not i.isdigit()])

    return "arrangement", [arrangementTypeToInt[guitarType], specialization]


def createSilenceEvent():
    return "silence", [0]


noteStartEventHandler = Handler(
    "noteStart",
    [
        EventVariable("string", 0, 5),
        EventVariable("fret", 0, 25),
        EventVariable("palm_mute", 0, 1),
        EventVariable("hammer_on", 0, 1),
        # EventVariable("pinch", 0, 1),
        # TODO : Vibrato
        # TODO : How to handle link next
        EventVariable("harmonic", 0, 1),  # TODO: Ive joined harmonic and pinch for now
        # EventVariable("fret_hand_mute", 0, 1), # TODO: Ive joined fret hand mute and palm mute for now
        EventVariable("accent", 0, 1),
        # EventRanges.EventRange("hopo", 0, 1), hopo is hammer on pull off
        EventVariable("tap", 0, 1)
    ]
)


def createNoteStartEvent(string: int, fret: int, palm_mute: bool, hammer_on: bool, harmonic: bool, accent: bool,
                         tap: bool, arrangement: str):
    return "noteStart", [string, fret, 1 if palm_mute else 0, 1 if hammer_on else 0, 1 if harmonic else 0,
                         1 if accent else 0, 1 if tap else 0], string, arrangement


noteEndEventHandler = Handler(
    "noteEnd",
    [
        EventVariable("string", 0, 5),
        EventVariable("fret", 0, 25),
        EventVariable("pull_off", 0, 1),
        EventVariable("unpitched_slide", 0, 1)
    ]
)


def createNoteEndEvent(string: int, fret: int, pull_off: bool, unpitched_slide: bool, arrangement: str):
    return "noteEnd", [string, fret, 1 if pull_off else 0, 1 if unpitched_slide else 0], string, arrangement


noteBendEventHandler = Handler(
    "bend",
    [
        EventVariable("string", 0, 5),
        EventVariable("semi-tone", -5, 5, 0.5, False),
        EventVariable("tap", 0, 1)
    ]
)


def createNoteBendEvent(string: int, semi_tone: float, tap: bool, arrangement: str):
    return "bend", [string, semi_tone, 1 if tap else 0], string, arrangement


endOfTieEventHandler = Handler(
    "eot",  # End of Tie Section
    [
        EventVariable("EOT", 0, 0)
    ]
)


def createEndOfTieEvent():
    return "eot", [0]


endOfSequenceEventHandler = Handler(
    "eos",  # End of Sequence
    [
        EventVariable("EOS", 0, 0)
    ]
)


def createEndOfSeqEvent():
    return "eos", [0]


def createTimeEvent(time):
    return "time", [time]


@dataclass
class SongSection:
    startSeconds: float
    stopSeconds: float
    tokens: list[int]
    spectrogram: np.core.multiarray


class GuitarTokenizer:
    def __init__(self,
                 maxNumberOfSeconds=60,
                 timeStepsPerSecond=100,
                 maxNumberOfTokensPerSection=1024,
                 sample_rate=16000,
                 n_ffts=2048,
                 hop_length=512,
                 n_mels=512,
                 maxNumberOfFrames: int = None
                 ):
        if maxNumberOfFrames is not None:
            maxNumberOfSeconds = librosa.frames_to_time(maxNumberOfFrames, sr=sample_rate, hop_length=hop_length,
                                                        n_fft=n_ffts)
        self._numberOfSeconds = maxNumberOfSeconds
        self.maxNumberOfSeconds = maxNumberOfSeconds
        self._timeStepsPerSecond = timeStepsPerSecond
        self.minTimeForNotes = 1 / timeStepsPerSecond
        self.sample_rate = sample_rate
        self.n_ffts = n_ffts
        self.hop_length = hop_length
        self.n_mels = n_mels
        self._timeEventHandler = Handler(
            "time",
            [generateTimeRange(maxNumberOfSeconds, timeStepsPerSecond)]
        )
        self.maxNumberOfTokensPerSection = maxNumberOfTokensPerSection
        self.encoder = Encoder([
            startOfSequenceEventHandler,
            self._timeEventHandler,
            arrangementEventHandler,
            noteStartEventHandler,
            noteEndEventHandler,
            noteBendEventHandler,
            endOfTieEventHandler,
            silenceEventHandler,
            endOfSequenceEventHandler,
        ])
        self.sosToken = self.encoder.encode(*createStartOfSeqEvent())
        self.eosToken = self.encoder.encode(*createEndOfSeqEvent())

    def numberOfTokens(self) -> int:
        return self.encoder.numberOfTokens

    def processAndAddNote(self, sortedEvents, n, arrangement):
        note: dict = n
        noteTime = float(note["time"])
        if not noteTime in sortedEvents:
            sortedEvents[noteTime] = {"start": [], "end": [], "bend": []}

        # create start note
        sortedEvents[noteTime]["start"].append(
            createNoteStartEvent(
                string=int(note["string"]),
                fret=int(note["fret"]),
                palm_mute="palmMute" in note,
                hammer_on="hammerOn" in note or "hopo" in note,
                harmonic="harmonic" in note or "harmonicPinch" in note,
                accent="accent" in note,
                tap="tap" in note,
                arrangement=arrangement
            )
        )

        noteEndTime = noteTime + float(note["sustain"]) if "sustain" in note else noteTime + self.minTimeForNotes
        noteEndFret = int(note["slideTo"]) if "slideTo" in note else int(note["fret"])
        noteEndFret = int(note["slideUnpitchTo"]) if "slideUnpitchTo" in note else noteEndFret

        if not noteEndTime in sortedEvents:
            sortedEvents[noteEndTime] = {"start": [], "end": [], "bend": []}

        sortedEvents[noteEndTime]["end"].append(
            createNoteEndEvent(
                string=int(note["string"]),
                fret=noteEndFret,
                pull_off="pullOff" in note or "hopo" in note,
                unpitched_slide="slideUnpitchTo" in note,
                arrangement=arrangement
            )
        )

        if "bend" in note:
            for bv in note["bendValues"]:
                bendValue: dict = bv
                bendTime = float(bendValue["time"])

                if not bendTime in sortedEvents:
                    sortedEvents[bendTime] = {"start": [], "end": [], "bend": []}

                sortedEvents[bendTime]["bend"].append(
                    createNoteBendEvent(
                        string=int(note["string"]),
                        semi_tone=float(bendValue["step"]) if "step" in bendValue else 0.0,
                        tap="tap" in bendValue,
                        arrangement=arrangement
                    )
                )

    def processAndAddChords(self, sortedEvents, c, chordTemplates, arrangement):
        chord: dict = c
        chordTime = float(chord["time"])
        chordAttributes = chord.copy()
        del chordAttributes["chordNotes"]
        del chordAttributes["chordId"]
        # print(chordAttributes)
        if "chordNotes" in chord and len(chord["chordNotes"]) > 0:
            # do for every chordNote
            for n in chord["chordNotes"]:
                note = mergeDicts(chordAttributes, n)
                self.processAndAddNote(sortedEvents, note, arrangement)
        else:
            # do for every note chordId
            chordId = int(chord["chordId"])
            for n in chordTemplates[chordId]["notes"]:
                note = mergeDicts(chordAttributes, n)
                self.processAndAddNote(sortedEvents, note, arrangement)

    @staticmethod
    def convertPathsToParsedFiles(pathsDict):
        input_paths = {}
        for key in pathsDict.keys():
            if key in validArrangementList:
                input_paths[key] = pathsDict[key]
        loaded_files = {}
        for key, path in input_paths.items():
            loaded_files[key] = parse_xml_file(path)
        return loaded_files

    def getTokensAndSpectrogram(self, filePaths):
        loaded_files = self.convertPathsToParsedFiles(filePaths)
        sortedEvents = sortedcontainers.SortedDict()
        for arrangementKey in loaded_files.keys():
            for chordTemplate in loaded_files[arrangementKey]["chordTemplates"]:
                filtered_dict = {k: v for (k, v) in chordTemplate.items() if "fret" in k}
                chordTemplate["notes"] = [{"string": k.replace("fret", ""), "fret": v} for (k, v) in
                                          filtered_dict.items()]

        # convert chord templates notes to readable notes

        for arrangementKey in loaded_files.keys():
            loadedFile = loaded_files[arrangementKey]
            for n in loadedFile["notes"]:
                self.processAndAddNote(sortedEvents, n, arrangementKey)

            for c in loadedFile["chords"]:
                self.processAndAddChords(sortedEvents, c, loadedFile["chordTemplates"], arrangementKey)

        sections = []
        y, sr = librosa.load(filePaths["ogg"], sr=self.sample_rate, mono=True)
        S = librosa.feature.melspectrogram(y=y,
                                           sr=sr,
                                           n_fft=self.n_ffts,
                                           n_mels=self.n_mels,
                                           hop_length=self.hop_length)
        lastOpenNotesForArrangement = {}
        for arrangementKey in loaded_files.keys():
            lastOpenNotesForArrangement[arrangementKey] = [None] * 6
        arrangementsInSong = list(loaded_files.keys())
        songLength = float(loaded_files[arrangementsInSong[0]]["songLength"])
        sortedEventsAsList = sortedEvents.keys()

        startTime = 0.0
        startSpectrogramIndex = 0
        tokens = [self.encoder.encode(*createStartOfSeqEvent())]
        lastArrangement = None

        for eventTime in sortedEventsAsList:
            # check if i need to make a new section
            if len(tokens) > self.maxNumberOfTokensPerSection or (eventTime - startTime) > self.maxNumberOfSeconds:
                # check for silence
                if len(tokens) == 2:
                    tokens.append(self.encoder.encode(*createSilenceEvent()))
                tokens.append(self.encoder.encode(*createEndOfSeqEvent()))
                numberOfFrames = librosa.core.time_to_frames(eventTime - startTime,
                                                             sr=self.sample_rate,
                                                             hop_length=self.hop_length,
                                                             n_fft=self.n_ffts)
                thisSectionSpectrogram = S[:, startSpectrogramIndex:startSpectrogramIndex + numberOfFrames]
                this_section = SongSection(startTime, eventTime, tokens, thisSectionSpectrogram)
                sections.append(this_section)
                startTime = eventTime
                startSpectrogramIndex = startSpectrogramIndex + numberOfFrames
                tokens = [self.encoder.encode(*createStartOfSeqEvent())]
                # add open notes to the new section
                for arrangementKey in loaded_files.keys():
                    if any(lastOpenNotesForArrangement[arrangementKey]):
                        lastArrangement = arrangementKey
                        tokens.append(self.encoder.encode(*createArrangementEvent(arrangementKey)))
                        for noteData in lastOpenNotesForArrangement[arrangementKey]:
                            if noteData is not None:
                                tokens.append(self.encoder.encode("noteStart", noteData))
                tokens.append(self.encoder.encode(*createEndOfTieEvent()))

            tokens.append(self.encoder.encode(*createTimeEvent(eventTime - startTime)))

            for event in sortedEvents[eventTime]["bend"]:
                if lastArrangement != event[3]:
                    tokens.append(self.encoder.encode(*createArrangementEvent(event[3])))
                    lastArrangement = event[3]
                tokens.append(self.encoder.encode(*event))
            # end notes
            for event in sortedEvents[eventTime]["end"]:
                if lastArrangement != event[3]:
                    tokens.append(self.encoder.encode(*createArrangementEvent(event[3])))
                    lastArrangement = event[3]
                lastOpenNotesForArrangement[event[3]][event[2]] = None
                tokens.append(self.encoder.encode(*event))

            # then write all new notes
            for event in sortedEvents[eventTime]["start"]:
                if lastArrangement != event[3]:
                    tokens.append(self.encoder.encode(*createArrangementEvent(event[3])))
                    lastArrangement = event[3]
                tokens.append(self.encoder.encode(*event))
                lastOpenNotesForArrangement[event[3]][event[2]] = event[1]

        # check and finish last section
        if len(tokens) == 2:
            tokens.append(self.encoder.encode(*createSilenceEvent()))
        tokens.append(self.encoder.encode(*createEndOfSeqEvent()))
        thisSectionSpectrogram = S[:, startSpectrogramIndex:]
        lastSection = SongSection(startTime, songLength, tokens, thisSectionSpectrogram)
        sections.append(lastSection)
        return sections,loaded_files


def test():
    all_dlcs = get_all_dlc_files("../RSFiles/MiniDataset")
    sections = []
    tokenizer = GuitarTokenizer(30, 50)
    sections,parsedSong = tokenizer.getTokensAndSpectrogram(all_dlcs[0])
    for section in sections:
        print(
            f"sTime:{section.startSeconds} eTime:{section.stopSeconds} length:{section.stopSeconds - section.startSeconds} lengthToken:{len(section.tokens)} shapeSpec:{section.spectrogram.shape}")
        for token in section.tokens:
            print(tokenizer.encoder.decode(token))


if __name__ == '__main__':
    test()
