import os
from pathlib import Path


def get_all_dlc_files(directory):
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
