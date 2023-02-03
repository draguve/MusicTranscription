import os
import subprocess
import magic
from zipfile import ZipFile
import joblib
import contextlib
from pathlib import Path
from tqdm import tqdm
from pprint import pprint
import shutil

from TUtils import tqdm_joblib

Tools = "Tools/"

path_ww2ogg = os.path.abspath(Tools + "ww2ogg.exe")
path_cookbook = os.path.abspath(Tools + "packed_codebooks_aoTuV_603.bin")
path_revorb = os.path.abspath(Tools + "revorb.exe")


def process_file(filename, root, just_filename):
    output_path = Path(filename).with_suffix(".ogg")
    check = subprocess.run(f"{path_ww2ogg} {os.path.abspath(filename)} --pcb {path_cookbook} -o {output_path}", capture_output=True, text=True)
    if check.returncode != 0:
        print(check.stdout)
        print(check.stderr)
        print(f"failed at file {filename}")
        return
    check = subprocess.run([path_revorb, output_path], capture_output=True, text=True)
    if check.returncode != 0:
        print(check.stdout)
        print(check.stderr)
        print(f"failed at file {filename}")
        return
    os.remove(filename)


# iterate over files in
# that directory
total = 0
delayed_funcs = []
for root, dirs, files in os.walk("Downloads"):
    for filename in files:
        if filename.endswith(".wem"):
            just_filename = filename
            filename = os.path.join(root, filename)
            delayed_funcs.append(joblib.delayed(process_file)(filename,root,just_filename))
            total += 1

with tqdm_joblib(tqdm(desc="Cleanup", total=total)) as progress_bar:
    joblib.Parallel(n_jobs=16)(delayed_funcs)
