# Copyright 2025 Xiaomi Corporation.
import os
from pathlib import Path


class LibriSpeechDataset:
    def __init__(self, split="test_clean"):
        base_dir = Path(os.getcwd())
        data_path = base_dir / "eval" / "data" / "librispeech" / split
        assert data_path.exists(), f"Data path {data_path} does not exist, split must in ['test_clean', 'test_other']"
        self.data = self.load_data(data_path)
        self.lang = "en"

    def load_data(self, data_path):
        wav_dir = data_path / "wav"
        data = []
        with open(data_path / "manifest.txt", "r") as f:
            for line in f.read().splitlines():
                id, text = line.strip().split("\t", maxsplit=1)
                audio_path = wav_dir / f"{id}.wav"
                if audio_path.exists():
                    data.append({
                        "id": id,
                        "audio": str(audio_path),
                        "text": text,
                    })
                else:
                    raise FileNotFoundError(f"File {audio_path} does not exist")
        print(f"Loaded {len(data)} items.")
        return data

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)
