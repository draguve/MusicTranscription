import os
import magic
from zipfile import ZipFile
import joblib
import contextlib
from tqdm import tqdm

from TUtils import tqdm_joblib


def process_file(filename, root):
    if not filename.lower().endswith('.psarc'):
        file_magic = magic.from_file(filename)
        if file_magic == "PSA archive data":
            # print(f"Need to rename {filename}")
            os.rename(filename, filename + ".psarc")
        elif file_magic == "Zip archive data, at least v2.0 to extract":
            try:
                with ZipFile(filename, 'r') as zObject:
                    zObject.extractall(path=root)
                os.remove(filename)
            except Exception as e:
                print(f"Could not process {e} {filename}")
        elif file_magic == "Zip archive data, at least v1.0 to extract":
            try:
                with ZipFile(filename, 'r') as zObject:
                    zObject.extractall(path=root)
                os.remove(filename)
            except Exception as e:
                print(f"Could not process {e} {filename}")
        elif file_magic == "data":
            os.remove(filename)
        elif file_magic == "PDF document, version 1.3":
            os.remove(filename)
        else:
            print(f"unexpected filetype {filename} - {file_magic}")


# iterate over files in
# that directory
total = 0
delayed_funcs = []
for root, dirs, files in os.walk("Downloads"):
    for filename in files:
        filename = os.path.join(root, filename)
        delayed_funcs.append(joblib.delayed(process_file)(filename,root))
        total += 1

with tqdm_joblib(tqdm(desc="Cleanup", total=total)) as progress_bar:
    joblib.Parallel(n_jobs=16)(delayed_funcs)

