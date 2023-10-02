import h5py

from Tokenizer import GuitarTokenizer


class H5GuitarTokenizer(GuitarTokenizer):
    def __init__(self, datasetFile):
        h5file = h5py.File(datasetFile, "r")
        meta = h5file.get("Meta").attrs
        super().__init__(
            maxNumberOfSeconds=meta["maxNumberOfSeconds"],
            timeStepsPerSecond=meta["timeStepsPerSecond"],
            maxNumberOfTokensPerSection=meta["maxNumberOfTokensPerSection"],
            sample_rate=meta["sample_rate"],
            n_ffts=meta["n_ffts"],
            hop_length=meta["hop_length"],
            n_mels=meta["n_mels"],
        )
        assert self.encoder.numberOfTokens == meta["vocab_size"]
