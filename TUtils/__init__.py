import contextlib
import os
from pathlib import Path
import string
import random

import joblib


def get_all_dlc_files(directory):
    data = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".ogg") and not filename.endswith("_preview.ogg"):
                dlc = {
                    "ogg": os.path.abspath(os.path.join(root, filename))
                }
                xml_files = Path(root).glob('*.xml')
                for xml_file in xml_files:
                    xml_filename = Path(xml_file)
                    dlc[xml_filename.stem.replace("arr_", "")] = os.path.abspath(str(xml_file))
                for rs2_file in (Path(root).glob('*.rs2dlc')):
                    dlc["rs2dlc"] = os.path.abspath(str(rs2_file))
                data.append(dlc)
    return data


def clamp(my_value, min_value, max_value):
    return max(min(my_value, max_value), min_value)


def random_string(length=15):
    return ''.join(random.choices(string.ascii_letters, k=length))


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()
