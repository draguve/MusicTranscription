import dataclasses
import time

import mir_eval.onset
import torch

import Tokenizer
from Tokenizer.loaderH5 import H5GuitarTokenizer
from TranscriptionDataset.TranscriptionDataset import TranscriptionDataset
import numpy as np
from mir_eval.onset import evaluate


#
# @dataclasses.dataclass
# class Note:
#     arrangement: list[int]
#     startTime: float
#     endTime: float
#     string: int
#     fret: int
#
#
# def getNotesFromTokens(tokenizer, tokens):
#     decoded_tokens = [tokenizer.decode(token) for token in tokens]
#     lastArrangement = None
#     openNotes = {}
#     last_time = -1.0
#     notes = []
#     for token in decoded_tokens:
#         if token[0] == "time":
#             last_time = token[1][0]
#         if token[0] == "arrangement":
#             lastArrangement = token[1]
#         if token[0] == "noteStart":
#             openNotes[lastArrangement] =


def convertTokensToOnsets(tokenizer, tokens):
    decoded_tokens = [tokenizer.decode(token) for token in tokens]
    last_time = -1.0
    onsets = []
    for token in decoded_tokens:
        if token[0] == "time":
            last_time = token[1][0]
        if token[0] == "noteStart":
            onsets.append(last_time)
        if token[0] == "eos":
            break
    return np.array(onsets)  # , np.array(offsets)


def convertTensorToF1(tokenizer, logits, correctTokens, lengths):
    tokens = torch.argmax(torch.nn.functional.softmax(logits), dim=-1).numpy()
    batch, length = tokens.shape
    all_out = []
    for i in range(batch):
        correctOnsets = convertTokensToOnsets(tokenizer, correctTokens[i, 0:lengths[i]])
        estimatedOnsets = convertTokensToOnsets(tokenizer, correctOnsets)
        f1 = mir_eval.onset.f_measure(correctOnsets, estimatedOnsets)
        all_out.append(f1)
    return np.mean(np.array(all_out), axis=0)


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
