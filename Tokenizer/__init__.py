from Event import Event, EventVariable, generateTimeRange
from Encoder import Encoder
import bs4
import os
from pathlib import Path
from dataclasses import dataclass


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


class Tokenizer:
    def __init__(self, numberOfSeconds, timeStepsPerSecond):
        self._encoder = Encoder([
            Event(
                "time",
                [generateTimeRange(numberOfSeconds, timeStepsPerSecond)]
            ),
            Event(
                "noteStart",
                [
                    EventVariable("string", 0, 5),
                    EventVariable("fret", 0, 25),
                    EventVariable("palm_mute", 0, 1),
                    EventVariable("hammer_on", 0, 1),
                    # EventRanges.EventRange("hopo", 0, 1), hopo is hammer on pull off
                    EventVariable("tap", 0, 1)
                ]
            ),
            Event(
                "noteEnd",
                [
                    EventVariable("string", 0, 5),
                    EventVariable("fret", 0, 25),
                    EventVariable("pull_off", 0, 1),
                    EventVariable("unpitched_slide", 0, 1)
                ]
            ),
            Event(
                "bend",
                [
                    EventVariable("string", 1, 6),
                    EventVariable("semi-tone", -2.5, 2.5, 0.5, False),
                    EventVariable("tap", 0, 1)
                ]
            ),
            Event(
                "eot",  # End of Tie Section
                [
                    EventVariable("EOS", 0, 0)
                ]
            ),
            Event(
                "eos",  # End of Sequence
                [
                    EventVariable("EOS", 0, 0)
                ]
            ),
        ])


def get_all_filenames():
    data = []
    for root, dirs, files in os.walk("Downloads"):
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
    x = Tokenizer(1,1000)
    print(x._encoder.numberOfTokens)
