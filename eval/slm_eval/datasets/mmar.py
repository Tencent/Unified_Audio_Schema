# Copyright 2025 Xiaomi Corporation.
import os
from datasets import load_dataset


class MMARDataset:
    def __init__(self):
        data_root = os.getcwd()
        data_dir = os.path.join(data_root, "eval/data/MMAR")
        self.data = load_dataset("BoJack/MMAR", split="test", cache_dir=data_dir)

        def add_root(example):
            example["audio_path"] = os.path.join(
                data_dir, example["audio_path"].lstrip("./")
            )
            return example

        self.data = self.data.map(add_root)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for item in self.data:
            yield item
