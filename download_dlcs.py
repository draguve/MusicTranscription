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
from joblib import Parallel, delayed
import string
import random
import unicodedata
import re

def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')

def get_random(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def download_item(item):
    download_url = urlparse(item["file_pc_link"])
    ignore_list = ["www.theriffrepeater.com", "store.steampowered.com", "theriffrepeater.com", "www.ubisoft.com"]
    download_location = f"Downloads/{item['id']}/"
    if Path(download_location).exists():
    # dont download when the directory already exists
        dir_check = os.listdir(download_location)
        if len(dir_check) == 0:
            # if the directory is empty delete it
            print("Empty directory")
            shutil.rmtree(download_location)
        else:
            return
    # os.makedirs(f"Downloads/{item['id']}")
    if download_url.hostname in ignore_list:
        print(f"Ignoring {item['file_pc_link']}")
        return
    try:
        match download_url.hostname:
            case "drive.google.com":
                if "folders" in item["file_pc_link"]:
                    os.makedirs(download_location)
                    gdown.download_folder(item['file_pc_link'], output=download_location, quiet=False)
                else:
                    os.makedirs(download_location)
                    downloaded_file = gdown.download(url=item['file_pc_link'], quiet=False, fuzzy=True)
                    shutil.move(downloaded_file, download_location)
            case "www.dropbox.com":
                os.makedirs(download_location)
                headers = {'user-agent': 'Wget/1.16 (linux-gnu)'}
                r = requests.get(item['file_pc_link'].replace("dl=0", "dl=1"), stream=True, headers=headers)
                to_download = download_location + slugify(f"{item['title'].replace(' ', '')}_{item['album'].replace(' ', '')}_{get_random()}.psarc")
                with open(to_download, 'w+b') as f:
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
                # check if size is more than 1MB
                if os.path.getsize(to_download) < 1000000:
                    print(
                        f'could not download from {item["file_pc_link"]} file size too low {os.path.getsize(to_download)}')
                    shutil.rmtree(download_location)
                else:
                    print(f"Downloaded from {item['file_pc_link']}")
            case "www.mediafire.com":
                os.makedirs(download_location)
                if isinstance(MediafireDL.Download(item['file_pc_link'], download_location[:-1],print_error=False), Exception):
                    # fallback download script
                    headers = {'user-agent': 'Wget/1.16 (linux-gnu)'}
                    r = requests.get(item['file_pc_link'], stream=True, headers=headers)
                    filename = slugify(f"{item['title'].replace(' ', '')}_{item['album'].replace(' ', '')}_{get_random()}.psarc")
                    to_download = download_location + filename
                    with open(to_download, 'w+b') as f:
                        for chunk in r.iter_content(chunk_size=1024):
                            if chunk:
                                f.write(chunk)
                    if os.path.getsize(to_download) < 1000000:
                        print(
                            f'could not download from {item["file_pc_link"]} file size too low {os.path.getsize(to_download)}')
                        shutil.rmtree(download_location)
                    else:
                        print(f"Downloaded from {item['file_pc_link']}")
            case other:
                print(f"no handler could not download {item['file_pc_link']}")
                return
                # raise Exception(f"No handler")
    except Exception as inst:
        print(f"{inst} could not download {item['file_pc_link']}")
        traceback.print_exc()
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
                print(f"Empty directory {item['file_pc_link']}")
                shutil.rmtree(download_location)



with open('links_all.json', 'r', encoding="utf8") as j:
    contents = json.loads(j.read())

    possible_hosts = {}
    delayed_funcs = []
    print(len(contents))
    for item in contents:
        if "file_pc_link" in item and item["file_pc_link"] != "":
            download_url = urlparse(item["file_pc_link"])
            if download_url.hostname not in possible_hosts:
                possible_hosts[download_url.hostname] = 0
            possible_hosts[download_url.hostname] += 1
            delayed_funcs.append(delayed(download_item)(item))

    parallel_pool = Parallel(n_jobs=6,verbose=True)
    parallel_pool(delayed_funcs)

    pprint(possible_hosts)
