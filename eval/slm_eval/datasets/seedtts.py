# Copyright 2025 Xiaomi Corporation.
import os
from pathlib import Path


class SeedTTSDataset:
    def __init__(self, split="en"):
        base_dir = Path(os.getcwd())
        data_path = base_dir / "eval" / "data" / "seedtts" / split
        assert data_path.exists(), f"Data path {data_path} does not exist, split must in ['en', 'zh', 'zh_hardcase']"
        self.data = self.load_data(data_path)
        self.lang = "en" if split == "en" else "zh"

    def load_data(self, data_path):
        data = []
        with open(data_path / "text.txt", "r", encoding="utf-8") as f:
            for id, line in enumerate(f.read().splitlines()):
                data.append({
                    "id": id,
                    "text": line,
                })
        print(f"Loaded {len(data)} items.")
        return data

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)
