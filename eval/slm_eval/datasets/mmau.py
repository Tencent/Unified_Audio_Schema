# Copyright 2025 Xiaomi Corporation.
import json
import os


class MMAUDataset:
    def __init__(self):
        data_root = os.getcwd()
        cache_dir = os.path.join(data_root, "eval/data/MMAU")
        save_path = os.path.join(cache_dir, "mmau-test-mini.json")
        self.data_root = os.path.dirname(save_path)
        self.data = json.load(open(save_path, "r"))

        self.post_process()

    def __len__(self):
        return len(self.data)

    def post_process(self):
        length = len(self.data)
        for i in range(length):
            self.data[i]["audio_id"] = os.path.join(self.data_root, self.data[i]["audio_id"].lstrip("./"))

    def __iter__(self):
        for item in self.data:
            yield item
