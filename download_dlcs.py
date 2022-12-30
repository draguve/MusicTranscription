import json
from urllib.parse import urlparse
from pprint import pprint
import os
from pathlib import Path
import shutil
import gdown

with open('links_all.json', 'r', encoding="utf8") as j:
    contents = json.loads(j.read())

possible_hosts = {}
print(len(contents))
for item in contents:
    if "file_pc_link" in item and item["file_pc_link"] != "":
        download_url = urlparse(item["file_pc_link"])
        if download_url.hostname not in possible_hosts:
            possible_hosts[download_url.hostname] = 0
        possible_hosts[download_url.hostname] += 1
        download_location = f"Downloads/{item['id']}/"
        print(download_location)
        if Path(download_location).exists():
            # dont download when the directory already exists
            dir_check = os.listdir(download_location)
            if len(dir_check) == 0:
                #if the directory is empty delete it 
                print("Empty directory")
                shutil.rmtree(download_location)
            else:
                continue
        # os.makedirs(f"Downloads/{item['id']}")
        try:
            if download_url.hostname == "drive.google.com":
                os.makedirs(download_location)
                downloaded_file = gdown.download(url=item['file_pc_link'], quiet=False, fuzzy=True)
                shutil.move(downloaded_file, download_location)
            else:
                raise Exception(f"could not download from {item['file_pc_link']}")
        except Exception as inst:
            print(inst)
            if Path(download_location).exists():
                shutil.rmtree(download_location)
            # try:
            #     os.makedirs(f"Downloads/{item['id']}")
            # except:
            #     if Path(download_location).exists():
            #         shutil.rmtree(download_location)
        finally:
            # Getting the list of directories
            if Path(download_location).exists():
                dir_check = os.listdir(download_location)
                if len(dir_check) == 0:
                    print("Empty directory")
                    shutil.rmtree(download_location)

pprint(possible_hosts)
