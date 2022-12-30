import json
from urllib.parse import urlparse
from pprint import pprint
import os
from pathlib import Path
import shutil
import gdown
import requests
import traceback
from mediafiredl import MediafireDL

with open('links_all.json', 'r', encoding="utf8") as j:
    contents = json.loads(j.read())

ignore_list = ["www.theriffrepeater.com", "store.steampowered.com", "theriffrepeater.com", "www.ubisoft.com"]

possible_hosts = {}
print(len(contents))
for item in contents:
    if "file_pc_link" in item and item["file_pc_link"] != "":
        download_url = urlparse(item["file_pc_link"])
        if download_url.hostname not in possible_hosts:
            possible_hosts[download_url.hostname] = 0
        possible_hosts[download_url.hostname] += 1
        download_location = f"Downloads/{item['id']}/"
        if Path(download_location).exists():
            # dont download when the directory already exists
            dir_check = os.listdir(download_location)
            if len(dir_check) == 0:
                # if the directory is empty delete it
                print("Empty directory")
                shutil.rmtree(download_location)
            else:
                continue
        # os.makedirs(f"Downloads/{item['id']}")
        if download_url.hostname in ignore_list:
            print(f"Ignoring {item['file_pc_link']}")
            continue
        try:
            match download_url.hostname:
                case "drive.google.com":
                    if "folders" in item["file_pc_link"]:
                        os.makedirs(download_location)
                        gdown.download_folder(item['file_pc_link'],output=download_location, quiet=False)
                    else:
                        os.makedirs(download_location)
                        downloaded_file = gdown.download(url=item['file_pc_link'], quiet=False, fuzzy=True)
                        shutil.move(downloaded_file, download_location)
                case "www.dropbox.com":
                    os.makedirs(download_location)
                    headers = {'user-agent': 'Wget/1.16 (linux-gnu)'}
                    r = requests.get(item['file_pc_link'].replace("dl=0", "dl=1"), stream=True, headers=headers)
                    to_download = download_location + f"{item['title'].replace(' ','')}_{item['album'].replace(' ','')}.psarc"
                    with open(to_download, 'w+b') as f:
                        for chunk in r.iter_content(chunk_size=1024):
                            if chunk:
                                f.write(chunk)
                    #check if size is more than 1MB
                    if os.path.getsize(to_download) < 1000000:
                        print(f'could not download from {item["file_pc_link"]} file size too low {os.path.getsize(to_download)}')
                        shutil.rmtree(download_location)
                    else:
                        print(f"Downloaded from {item['file_pc_link']}")
                case "www.mediafire.com":
                    os.makedirs(download_location)
                    MediafireDL.Download(item['file_pc_link'],download_location[:-1])
                case other:
                    raise Exception(f"No handler")
        except Exception as inst:
            print(f"{inst} could not download {item['file_pc_link']}")
            # traceback.print_exc()
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
