# Copyright 2025 Xiaomi Corporation.
from .librispeech import LibriSpeechDataset
from .aishell1 import AiShell1Dataset
from .seedtts import SeedTTSDataset
from .mmsu import MMSUDataset
from .mmau import MMAUDataset
from .mmar import MMARDataset


def get_dataset(dataset, split, sample_rate=24000):
    if dataset == "librispeech":
        return LibriSpeechDataset(split)
    if dataset == "aishell1":
        return AiShell1Dataset(split)
    if dataset == "seedtts":
        return SeedTTSDataset(split)
    if dataset == "mmsu":
        return MMSUDataset()
    if dataset == "mmau":
        return MMAUDataset()
    if dataset == "mmar":
        return MMARDataset()
