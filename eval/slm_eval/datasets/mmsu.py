# Copyright 2025 Xiaomi Corporation.
from pathlib import Path
import json
import os


class MMSUDataset:
    def __init__(self) -> None:
        base_dir = Path(os.getcwd())
        root: Path = base_dir / "eval/data/MMSU_Bench/MMSU"
        self.input_jsonl: Path = root / "question" / "mmsu.jsonl"
        self.data_root: str = str(root)

        data = []
        with open(self.input_jsonl, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        if 'choice_a' in item:
                            item['choice_a'] = str(item['choice_a'])
                        if 'choice_b' in item:
                            item['choice_b'] = str(item['choice_b'])
                        if 'choice_c' in item:
                            item['choice_c'] = str(item['choice_c'])
                        if 'choice_d' in item:
                            item['choice_d'] = str(item['choice_d'])
                        if 'answer_gt' in item:
                            item['answer_gt'] = str(item['answer_gt'])
                        data.append(item)
                    except json.JSONDecodeError as json_err:
                        print(f"Error parsing line {line_num}: {json_err}")
                        print(f"Line content: {line}")
                        continue

        self.ds = data

    def __len__(self):
        if hasattr(self.ds, '__len__'):
            return len(self.ds)
        return 0

    def __getitem__(self, idx):
        if isinstance(self.ds, list):
            return self.ds[idx]
        else:
            return self.ds[idx]

    def __iter__(self):
        if isinstance(self.ds, list):
            return iter(self.ds)
        else:
            return iter(self.ds)
