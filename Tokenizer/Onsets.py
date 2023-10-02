import time

import Tokenizer
from Tokenizer.loaderH5 import H5GuitarTokenizer
from TranscriptionDataset.TranscriptionDataset import TranscriptionDataset
import numpy as np
from mir_eval.onset import evaluate


def convertTokensToOnsets(tokenizer, tokens):
    decoded_tokens = [tokenizer.decode(token) for token in tokens]
    print(decoded_tokens)
    last_time = -1.0
    onsets = []
    offsets = []
    for token in decoded_tokens:
        if token[0] == "time":
            last_time = token[1][0]
        if token[0] == "noteStart":
            onsets.append(last_time)
        # if token[0] == "noteEnd":
        #     offsets.append(last_time)
    return np.array(onsets)  # , np.array(offsets)


def main():
    datasetLocation = "../Trainsets/S_Tier_1695428558_mTokens1000_mNoS60.hdf5"
    tokenizer = H5GuitarTokenizer(datasetLocation)
    dataset = TranscriptionDataset(datasetLocation, batchSize=1)
    dataset_iter = iter(dataset)
    tokens = next(dataset_iter)[-1].numpy()
    # DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
    start = time.perf_counter()
    test1 = convertTokensToOnsets(tokenizer, tokens)
    end = time.perf_counter()
    print("Elapsed (with compilation) = {}s".format((end - start)))
    print(evaluate(test1, test1[:-1]))


if __name__ == '__main__':
    main()
