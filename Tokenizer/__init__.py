from Tokenizer.EventHandler import Handler, EventVariable, generateTimeRange
from Tokenizer.Encoder import Encoder
from Tokenizer.SongXMLParser import parse_xml_file
from dataclasses import dataclass
import sortedcontainers
from pprint import pprint
from TUtils import get_all_dlc_files


def mergeDicts(dict1, dict2):
    """
    Here Dict2 takes precedent over Dict1
    @param dict1:
    @param dict2:
    @return:
    """
    res = {**dict1, **dict2}
    return res


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
                         tap: bool):
    return "noteStart", [string, fret, 1 if palm_mute else 0, 1 if hammer_on else 0, 1 if harmonic else 0,
                         1 if accent else 0, 1 if tap else 0], string


noteEndEventHandler = Handler(
    "noteEnd",
    [
        EventVariable("string", 0, 5),
        EventVariable("fret", 0, 25),
        EventVariable("pull_off", 0, 1),
        EventVariable("unpitched_slide", 0, 1)
    ]
)


def createNoteEndEvent(string: int, fret: int, pull_off: bool, unpitched_slide: bool):
    return "noteEnd", [string, fret, 1 if pull_off else 0, 1 if unpitched_slide else 0], string


noteBendEventHandler = Handler(
    "bend",
    [
        EventVariable("string", 0, 5),
        EventVariable("semi-tone", -3, 3, 0.5, False),
        EventVariable("tap", 0, 1)
    ]
)


def createNoteBendEvent(string: int, semi_tone: float, tap: bool, ):
    return "bend", [string, semi_tone, 1 if tap else 0], string


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


class GuitarTokenizer:
    def __init__(self, numberOfSeconds, timeStepsPerSecond):
        self._numberOfSeconds = numberOfSeconds
        self._timeStepsPerSecond = timeStepsPerSecond
        self.minTimeForNotes = 1 / timeStepsPerSecond
        self._timeEventHandler = Handler(
            "time",
            [generateTimeRange(numberOfSeconds, timeStepsPerSecond)]
        )
        self._encoder = Encoder([
            self._timeEventHandler,
            noteStartEventHandler,
            noteEndEventHandler,
            noteBendEventHandler,
            endOfTieEventHandler,
            endOfSequenceEventHandler,
        ])

    def processAndAddNote(self, sortedEvents, n):
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
                tap="tap" in note)
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
                unpitched_slide="slideUnpitchTo" in note
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
                        tap="tap" in bendValue
                    )
                )

    def processAndAddChords(self, sortedEvents, c, chordTemplates):
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
                self.processAndAddNote(sortedEvents, note)
        else:
            # do for every note chordId
            chordId = int(chord["chordId"])
            for n in chordTemplates[chordId]["notes"]:
                note = mergeDicts(chordAttributes, n)
                self.processAndAddNote(sortedEvents, note)

    def convertSongFromPath(self, path):
        loadedFile = parse_xml_file(path)
        return self.convertSongFromParsedFile(loadedFile)

    def convertSongFromParsedFile(self,loadedFile):
        sortedEvents = sortedcontainers.SortedDict()

        # convert chord templates notes to readable notes
        for chordTemplate in loadedFile["chordTemplates"]:
            filtered_dict = {k: v for (k, v) in chordTemplate.items() if "fret" in k}
            chordTemplate["notes"] = [{"string": k.replace("fret", ""), "fret": v} for (k, v) in filtered_dict.items()]

        for n in loadedFile["notes"]:
            self.processAndAddNote(sortedEvents, n)

        for c in loadedFile["chords"]:
            self.processAndAddChords(sortedEvents, c, loadedFile["chordTemplates"])

        sections = []
        queue = list(sortedEvents.keys())
        lastOpenNotes = [None] * 6
        while len(queue) > 0:
            startTime = queue.pop(0)
            stopTime = startTime + self._numberOfSeconds
            thisTimeStep = [startTime]
            while len(queue) > 0 and queue[0] <= stopTime:
                thisTimeStep.append(queue.pop(0))

            tokens = []
            # TODO : check ties are created properly something feels off, THERE IS A BUG HERE FIX IT
            # restart all notes in tie
            for noteData in lastOpenNotes:
                if noteData is not None:
                    tokens.append(self._encoder.encode("noteStart", noteData))
            tokens.append(self._encoder.encode(*createEndOfTieEvent()))

            # write all the tokens in this time step
            while len(thisTimeStep) > 0:
                currentTime = thisTimeStep.pop(0)

                # reemit even in case of 0
                tokens.append(self._encoder.encode(*createTimeEvent(currentTime - startTime)))

                # bend if any first
                for event in sortedEvents[currentTime]["bend"]:
                    tokens.append(self._encoder.encode(*event))
                # end notes
                for event in sortedEvents[currentTime]["end"]:
                    lastOpenNotes[event[2]] = None
                    tokens.append(self._encoder.encode(*event))

                # then write all new notes
                for event in sortedEvents[currentTime]["start"]:
                    tokens.append(self._encoder.encode(*event))
                    lastOpenNotes[event[2]] = event[1]

            tokens.append(self._encoder.encode(*createEndOfSeqEvent()))
            sections.append(SongSection(startTime, stopTime, tokens))
            # print(len(tokens),tokens)
            # for token in tokens:
            #     print(self._encoder.decode(token))
            # print("-----------------------------------------------------------")
        return sections

if __name__ == '__main__':
    all_dlcs = get_all_dlc_files("../Downloads/")
    tokenizer = GuitarTokenizer(1, 1000)
    pprint(tokenizer.convertSongFromPath(all_dlcs[4]["lead"]))
