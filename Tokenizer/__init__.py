from Event import Event, EventVariable, generateTimeRange
from Encoder import Encoder
import bs4
import os
from pathlib import Path
from dataclasses import dataclass
import sortedcontainers


def mergeDicts(dict1, dict2):
    """
    Here Dict2 takes precedent over Dict1
    @param dict1:
    @param dict2:
    @return:
    """
    res = {**dict1, **dict2}
    return res


def atts_plus_bend_values(item: bs4.PageElement):
    all_attrs = item.attrs
    all_attrs["bendValues"] = [bend.attrs for bend in (item.find_all("bendValue"))]
    return all_attrs


def attrs_plus_chord_notes(chord: bs4.PageElement):
    all_attrs = chord.attrs
    all_attrs["chordNotes"] = [atts_plus_bend_values(note) for note in chord.find_all("chordNote")]
    return all_attrs


def parse_vocals(xml_path):
    with open(xml_path, 'r') as f:
        data = f.read()
        bs_data = bs4.BeautifulSoup(data, 'xml')
        xml_data = {"vocal": [item.attrs for item in bs_data.find_all("vocal")]}
        return xml_data


def parse_showlights(xml_path):
    with open(xml_path, 'r') as f:
        data = f.read()
        bs_data = bs4.BeautifulSoup(data, 'xml')
        xml_data = {"showlight": [item.attrs for item in bs_data.find_all("showlight")]}
        return xml_data


def parse_xml_file(xml_path):
    with open(xml_path, 'r') as f:
        data = f.read()
        bs_data = bs4.BeautifulSoup(data, 'xml')
        xml_data = {"title": bs_data.find("title").string, "tuning": bs_data.find("tuning").attrs,
                    "arrangement": bs_data.find("arrangement").string, "offset": bs_data.find("offset").string,
                    "centOffset": bs_data.find("centOffset").string, "songLength": bs_data.find("songLength").string,
                    "startBeat": bs_data.find("startBeat").string, "averageTempo": bs_data.find("averageTempo").string,
                    "capo": bs_data.find("startBeat").string, "artistName": bs_data.find("artistName").string,
                    "albumName": bs_data.find("albumName").string, "albumYear": bs_data.find("albumYear").string,
                    "arrangementProperties": bs_data.find("arrangementProperties").attrs,
                    "notes": [atts_plus_bend_values(item) for item in bs_data.find_all("note")],
                    "chords": [attrs_plus_chord_notes(chord) for chord in bs_data.find_all("chord")],
                    "ebeats": [item.attrs for item in bs_data.find_all("ebeat")],
                    "chordTemplates": [item.attrs for item in bs_data.find_all("chordTemplate")],
                    "phraseIterations": [item.attrs for item in bs_data.find_all("phraseIteration")],
                    "sections": [item.attrs for item in bs_data.find_all("section")],
                    "anchors": [item.attrs for item in bs_data.find_all("anchor")],
                    "handShapes": [item.attrs for item in bs_data.find_all("handShape")]}
        return xml_data


@dataclass
class SongSection:
    startSeconds: float
    durationSeconds: float
    tokens: list[int]


noteStartEventHandler = Event(
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
                         1 if accent else 0, 1 if tap else 0]


noteEndEventHandler = Event(
    "noteEnd",
    [
        EventVariable("string", 0, 5),
        EventVariable("fret", 0, 25),
        EventVariable("pull_off", 0, 1),
        EventVariable("unpitched_slide", 0, 1)
    ]
)


def createNoteEndEvent(string: int, fret: int, pull_off: bool, unpitched_slide: bool):
    return "noteEnd", [string, fret, 1 if pull_off else 0, 1 if unpitched_slide else 0]


noteBendEventHandler = Event(
    "bend",
    [
        EventVariable("string", 0, 5),
        EventVariable("semi-tone", -2.5, 2.5, 0.5, False),
        EventVariable("tap", 0, 1)
    ]
)


def createNoteBendEvent(string: int, semi_tone: float, tap: bool, ):
    return "bend", [string, semi_tone, 1 if tap else 0]


endOfTieEventHandler = Event(
    "eot",  # End of Tie Section
    [
        EventVariable("EOT", 0, 0)
    ]
)


def createEndOfTieEvent():
    return "eot", [0]


endOfSequenceEventHandler = Event(
    "eos",  # End of Sequence
    [
        EventVariable("EOS", 0, 0)
    ]
)


def createEndOfSeqEvent():
    return "eos", [0]


class GuitarTokenizer:
    def __init__(self, numberOfSeconds, timeStepsPerSecond):
        self.minTimeForNotes = 1 / timeStepsPerSecond
        self._encoder = Encoder([
            Event(
                "time",
                [generateTimeRange(numberOfSeconds, timeStepsPerSecond)]
            ),
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
            sortedEvents[noteTime] = []

        # create start note
        sortedEvents[noteTime].append(
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
            sortedEvents[noteEndTime] = []

        sortedEvents[noteEndTime].append(
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
                    sortedEvents[bendTime] = []

                sortedEvents[bendTime].append(
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

    def convertSong(self, path):
        sortedEvents = sortedcontainers.SortedDict()
        loadedFile = parse_xml_file(path)

        # convert chord templates notes to readable notes
        for chordTemplate in loadedFile["chordTemplates"]:
            filtered_dict = {k: v for (k, v) in chordTemplate.items() if "fret" in k}
            chordTemplate["notes"] = [{"string": k.replace("fret", ""), "fret": v} for (k, v) in filtered_dict.items()]

        for n in loadedFile["notes"]:
            self.processAndAddNote(sortedEvents, n)

        for c in loadedFile["chords"]:
            self.processAndAddChords(sortedEvents, c, loadedFile["chordTemplates"])

        # TODO: need to add time events and tie events
        for key in sortedEvents.keys():
            for event in sortedEvents[key]:
                print(key, event, self._encoder.encode(event[0], event[1]))
            # print(key, sortedEvents[key])


def get_all_filenames(directory):
    data = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".ogg") and not filename.endswith("_preview.ogg"):
                dlc = {
                    "ogg": os.path.join(root, filename)
                }
                xml_files = Path(root).glob('*.xml')
                for xml_file in xml_files:
                    xml_filename = Path(xml_file)
                    dlc[xml_filename.stem.replace("arr_", "")] = str(xml_file)
                for rs2_file in (Path(root).glob('*.rs2dlc')):
                    dlc["rs2dlc"] = str(rs2_file)
                data.append(dlc)
    return data


if __name__ == '__main__':
    all_dlcs = get_all_filenames("../Downloads/")
    tokenizer = GuitarTokenizer(1, 1000)
    tokenizer.convertSong(all_dlcs[4]["lead"])
