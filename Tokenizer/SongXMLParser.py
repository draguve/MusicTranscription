import bs4


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
    with open(xml_path, 'r', errors='replace') as f:
        data = f.read()
        bs_data = bs4.BeautifulSoup(data, 'xml')
        xml_data = {"title": bs_data.find("title").string, "tuning": bs_data.find("tuning").attrs,
                    "arrangement": bs_data.find("arrangement").string, "offset": bs_data.find("offset").string,
                    "centOffset": bs_data.find("centOffset").string, "songLength": bs_data.find("songLength").string,
                    "startBeat": bs_data.find("startBeat").string, "averageTempo": bs_data.find("averageTempo").string,
                    "capo": bs_data.find("capo").string, "artistName": bs_data.find("artistName").string,
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
