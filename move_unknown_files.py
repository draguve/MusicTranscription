import os
import magic
from zipfile import ZipFile
import joblib
import contextlib
from tqdm import tqdm
import shutil

from TUtils import tqdm_joblib


def process_file(filename, root,just_filename):
    if not filename.lower().endswith('.psarc'):
        file_magic = magic.from_file(filename)
        if file_magic != "PSA archive data":
            # print(f"Need to rename {filename}")
            shutil.move(filename, f"Misc/{just_filename}"+root.replace('/','').replace('\\',''))

# iterate over files in
# that directory
total = 0
delayed_funcs = []
for root, dirs, files in os.walk("Downloads"):
    for filename in files:
        just_filename = filename
        filename = os.path.join(root, filename)
        delayed_funcs.append(joblib.delayed(process_file)(filename,root,just_filename))
        total += 1

with tqdm_joblib(tqdm(desc="Cleanup", total=total)) as progress_bar:
    joblib.Parallel(n_jobs=16)(delayed_funcs)

