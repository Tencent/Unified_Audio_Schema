# Copyright 2025 Xiaomi Corporation.
import json
import os
from pathlib import Path
import random
import re
from typing import List, Dict
import numpy as np
import torch
import tqdm
from ..models.uas_audio import UASAudioModel

THINK_RE = re.compile(r'^\s*(?:<think>.*?</think>\s*)+', flags=re.DOTALL | re.IGNORECASE)

MARKERS = [
    re.compile(r'(?m)^[^\S\r\n]*(?:the\s+)?correct\s+(?:choice|answer)\s+is\s*[:\-]?\s*(.*)$', flags=re.IGNORECASE),
    re.compile(r'(?m)^[^\S\r\n]*answer\s*[:\-]\s*(.*)$', flags=re.IGNORECASE),
]

BOLD = re.compile(r'\*\*(.+?)\*\*')


def _normalize(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    s = s.splitlines()[0].strip()
    s = s.replace('\\"', '"').replace("\\'", "'")
    s = re.sub(r'^\*{1,2}(.+?)\*{1,2}$', r'\1', s)
    s = re.sub(r'^`(.+?)`$', r'\1', s)
    s = re.sub(r'^[\'"](.+?)[\'"]$', r'\1', s)
    s = s.strip(" \t\r\n.:;!?,")
    s = re.sub(r'\s+', ' ', s)
    return s


def extract_answer(text: str, *, split_on_commas: bool = False) -> Dict[str, List[str] or str]:
    if text is None:
        return {"answer": "", "all_candidates": []}

    cleaned = text
    candidates: List[str] = []

    for pat in MARKERS:
        for m in pat.finditer(cleaned):
            tail = _normalize(m.group(1))
            if tail:
                candidates.append(tail)
            else:
                after = cleaned[m.end():]
                mb = BOLD.search(after)
                if mb:
                    b = _normalize(mb.group(1))
                    if b:
                        candidates.append(b)
                        continue
                for line in after.splitlines():
                    ln = _normalize(line)
                    if ln:
                        candidates.append(ln)
                        break

    if not candidates:
        for mb in BOLD.finditer(cleaned):
            b = _normalize(mb.group(1))
            if b:
                candidates.append(b)

    if not candidates:
        for line in cleaned.splitlines():
            ln = _normalize(line)
            if ln:
                candidates.append(ln)
                break

    if not candidates:
        return {"answer": "", "all_candidates": []}

    final = candidates[-1]

    if split_on_commas:
        parts = [p.strip() for p in final.split(',') if p.strip()]
        if parts:
            final = parts[0]

    return {"answer": final, "all_candidates": candidates}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MMAUEvaluator:
    def __init__(self, model, dataset, model_type='base', n_few_shots=0, thinking=False):
        self.model = model
        self.dataset = dataset
        self.model_type = model_type
        self.n_few_shots = n_few_shots
        self.thinking = thinking

    def format_qa(self, question, choices):
        final_output = question + '\n\n' + 'Choice: \n'
        for i in range(len(choices)):
            final_output += f"{choices[i]}\n"
        final_output += f"\n"

        final_output += f'Make a choice from the given {len(choices)} choices.'

        return final_output

    def string_match(self, answer, prediction, choices):
        def tokenize(text):
            return set(re.findall(r'\b\w+\b', text.lower()))

        # 首先尝试字母索引匹配：如果预测是单个字母（A-Z），映射到对应的选项索引
        prediction_normalized = prediction.strip().upper()
        if len(prediction_normalized) == 1 and prediction_normalized.isalpha():
            letter_index = ord(prediction_normalized) - ord('A')
            if 0 <= letter_index < len(choices):
                # 检查该索引对应的选项是否与正确答案匹配
                selected_choice = choices[letter_index]
                selected_tokens = tokenize(selected_choice)
                answer_tokens = tokenize(answer)

                if selected_tokens == answer_tokens:
                    return True

        # 如果字母索引匹配失败，使用原有的 token 匹配逻辑
        prediction_tokens = tokenize(prediction)
        answer_tokens = tokenize(answer)

        if not prediction_tokens:
            return False

        incorrect_tokens = set()
        for choice in choices:
            choice_tokens = tokenize(choice)
            if choice_tokens != answer_tokens:
                incorrect_tokens.update(choice_tokens - answer_tokens)

        cond1 = answer_tokens.issubset(prediction_tokens)
        cond2 = prediction_tokens.isdisjoint(incorrect_tokens)

        return cond1 and cond2

    def format_instruct_qa(self, question, choices, audio):
        # For UASAudioModel, use strict format: question + \n\n + choices (one per line with A. B. C. D. prefix)
        if isinstance(self.model, UASAudioModel):
            final_output = question + '\n\n'
            for i in range(len(choices)):
                choice_letter = chr(ord('A') + i)  # A, B, C, D, ...
                final_output += f"{choice_letter}. {choices[i]}\n"
            # Remove trailing newline
            final_output = final_output.rstrip('\n')
        else:
            final_output = "<sound>" + question + '\n\n' + 'Choice: \n'
            for i in range(len(choices)):
                final_output += f"{choices[i]}\n"
            final_output += f"\n"

            final_output += f'Choose a choices from the given {len(choices)} choices. Do not provide any additional explanations or content. Output must match exactly one of the listed choices.'

        instruction = [
            {
                "from": "human",
                "value": [
                    {
                        "type": "text",
                        "value": final_output
                    },
                    {
                        "type": "sound",
                        "value": audio
                    }
                ]
            }
        ]
        return instruction

    def few_shot_qa(self, question, choices, audio, audio_path, past_examples):
        final_output = []
        for i in range(len(past_examples)):
            final_output.append({
                'question': self.format_qa(past_examples[i]['question'], past_examples[i]['choices']),
                'audio': past_examples[i]['audio_id'],
                'answer': past_examples[i]['answer'],
                'audio_path': past_examples[i]['audio_id']
            })
        final_output.append({
            'question': self.format_qa(question, choices),
            'audio': audio,
            'audio_path': audio_path
        })

        return final_output

    def get_few_shot_exclude(self, dataset, n_few_shots, exclude_index):
        import random

        if not 0 <= exclude_index < len(dataset):
            raise IndexError(f"exclude_index ({exclude_index}) 超出范围 (0~{len(dataset)-1})")

        candidates = dataset[:exclude_index] + dataset[exclude_index+1:]

        if n_few_shots > len(candidates):
            raise ValueError(f"n_few_shots ({n_few_shots}) 不能大于候选样本数 ({len(candidates)})")

        return random.sample(candidates, n_few_shots)

    def evaluate(self, output_dir, rank=0, world_size=1):
        set_seed(42)
        if not self.thinking:
            output_dir = Path(output_dir)
        else:
            output_dir = Path(str(output_dir) + "_thinking")
        output_dir.mkdir(parents=True, exist_ok=True)

        predictions_file_path_rank = output_dir / f"predictions_rank_{rank}.json"

        predictions_this_rank = []
        processed_item_ids_this_rank = set()

        if predictions_file_path_rank.exists():
            try:
                with open(predictions_file_path_rank, "r", encoding="utf-8") as f:
                    predictions_this_rank = json.load(f)
                for pred_item in predictions_this_rank:
                    processed_item_ids_this_rank.add(pred_item["uid"])
                if processed_item_ids_this_rank:
                    print(f"Rank {rank} resumed from checkpoint: {predictions_file_path_rank}. {len(processed_item_ids_this_rank)} items already processed for this rank.")
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load predictions file {predictions_file_path_rank} for rank {rank}: {e}. Starting fresh inference for this rank.")
                predictions_this_rank = []
                processed_item_ids_this_rank = set()

        dataset_shard = list(self.dataset)[rank::world_size]
        items_to_process_this_run = []

        for idx, item in enumerate(dataset_shard):
            item_id = rank + idx * world_size
            if item_id not in processed_item_ids_this_rank:
                item['uid'] = item_id
                items_to_process_this_run.append(item)

        if rank == 0:
            print(f"\n========== Running MMAU Evaluation ==========\n")

        progress_bar = tqdm.tqdm(items_to_process_this_run, desc=f"Running MMAU Evaluation (Rank {rank})", disable=(rank != 0))

        index = 0
        for item in progress_bar:
            print(f"\n----- Sample {item['uid']+1} (Rank {rank}) -----")

            audio = item['audio_id']
            question = item['question']
            audio_path = item['audio_id']
            choices = item['choices']

            if self.model_type == 'base':
                if self.n_few_shots > 0:
                    response = self.model.few_shots_qa(self.few_shot_qa(question, choices, audio, audio_path, self.get_few_shot_exclude(items_to_process_this_run, self.n_few_shots, index)))
                else:
                    response = self.model.qa(audio, self.format_qa(question, choices))
            elif self.model_type == 'instruct':
                response = self.model.instruction_following(self.format_instruct_qa(question, choices, audio), append_generation_prompt=True, thinking=self.thinking)

            result_dict = {k: v for k, v in item.items() if k != 'audio'}
            result_dict['response'] = response
            print("++++++++++++++++++++++++++ Question ++++++++++++++++++++++++++")
            print("Question: ", item['question'])
            print("++++++++++++++++++++++++++ choices ++++++++++++++++++++++++++")
            print("Choices: ", choices)
            print("++++++++++++++++++++++++++ Response ++++++++++++++++++++++++++")
            print("Response: ", response)
            print("-" * 70)

            predictions_this_rank.append(result_dict)

            temp_file_path_rank = output_dir / f"predictions_rank_{rank}.json.tmp"
            with open(temp_file_path_rank, "w", encoding="utf-8") as f_tmp:
                json.dump(predictions_this_rank, f_tmp, indent=4, ensure_ascii=False)
            os.replace(temp_file_path_rank, predictions_file_path_rank)
            index += 1

        print(f"Rank {rank} MMAU evaluation complete. Processed {len(items_to_process_this_run)} new items. Results saved to {predictions_file_path_rank}")

    def calculate_acc(self, all_predictions):
        task_map = {
            'music': 0,
            'sound': 1,
            'speech': 2
        }
        task_score = [0.0, 0.0, 0.0]
        task_number = [0, 0, 0]

        for i in range(len(all_predictions)):
            item = all_predictions[i]
            if self.thinking:
                if '</think>' in item['response']:
                    response = item['response'].split('</think>', 1)[-1].lstrip()
                else:
                    response = item['response'].lstrip()
            else:
                response = item['response'].lstrip()
            response = extract_answer(response)['answer']
            all_predictions[i]['score'] = self.string_match(item['answer'], response, item['choices'])
            task_score[task_map[item['task']]] += self.string_match(item['answer'], response, item['choices'])
            task_number[task_map[item['task']]] += 1

        total_score = sum(task_score) / sum(task_number)
        for i in range(len(task_score)):
            task_score[i] = task_score[i] / task_number[i]

        return all_predictions, task_score, total_score

    def calculate_metrics(self, output_dir, rank=0, world_size=1):
        if rank != 0:
            return None

        if not self.thinking:
            output_dir = Path(output_dir)
        else:
            output_dir = Path(output_dir + "_thinking")
        all_predictions = []

        temp_files = []
        pattern = f"predictions_rank_*.json"
        import glob

        temp_file_pattern = str(output_dir / pattern)
        found_temp_files = glob.glob(temp_file_pattern)

        if not found_temp_files:
            print(f"警告: 在 {output_dir} 中未找到任何临时文件 (模式: {pattern})")
            return None, []

        found_temp_files.sort(key=lambda x: int(Path(x).stem.split('_rank_')[-1]))

        for temp_file_path_str in found_temp_files:
            temp_file_path = Path(temp_file_path_str)
            temp_files.append(temp_file_path)

            rank_num = temp_file_path.stem.split('_rank_')[-1]

            if temp_file_path.exists():
                with open(temp_file_path, "r", encoding="utf-8") as f:
                    all_predictions.extend(json.load(f))
                print(f"Collected rank {rank_num} results: {temp_file_path}")
            else:
                print(f"Warning: File not found: {temp_file_path}")

        predictions_file_path = output_dir / "results.json"
        with open(predictions_file_path, "w", encoding="utf-8") as f:
            json.dump(all_predictions, f, indent=4, ensure_ascii=False)

        print(f"\n========== MMAU Evaluation Summary ==========\n")
        print(f"Total items processed: {len(all_predictions)}")
        print(f"Results saved to: {predictions_file_path}")

        results, score, total_score = self.calculate_acc(all_predictions)

        with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        print(f"\n========== MMAU Final Evaluation Results ==========")
        print(f"Overall: {total_score}\n")
        print(f"Music: {score[0]}")
        print(f"Sound: {score[1]}")
        print(f"Speech: {score[2]}")
        print("=" * 58)

        scores_dict = {
            "overall": total_score,
            "music": score[0],
            "sound": score[1],
            "speech": score[2],
        }

        with open(output_dir / "mmau_scores.json", "w", encoding="utf-8") as f:
            json.dump(scores_dict, f, indent=4)

        return results
