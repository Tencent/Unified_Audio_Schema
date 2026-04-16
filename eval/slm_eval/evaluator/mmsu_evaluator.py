# Copyright 2025 Xiaomi Corporation.
from collections import defaultdict
import json
from pathlib import Path
import random
import re
import tqdm
from ..models.uas_audio import UASAudioModel


PUNCTUATION_REGEX = re.compile(r"[^\w\s]")


def extract_choice_from_response_instruct(response: str) -> str:
    return response.strip().split("\n")[0]


class MMSUEvaluator:
    def __init__(self, model, dataset, model_type, n_few_shots=0, seed=42, thinking=False):
        self.model = model
        self.dataset = dataset
        self.n_few_shots = n_few_shots
        self.dataset_name = "MMSU"
        random.seed(seed)
        self.model_type = model_type
        self.thinking = thinking

    def load_jsonl_data(self, jsonl_path):
        records = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f):
                try:
                    record = json.loads(line.strip())
                    records.append(record)
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error at line {line_number}: {line.strip()}")
                    print(e)
        return records

    def calculate_accuracy_per_task_and_category(self, data):
        assert self.model_type == "instruct", "Model type must be instruct"
        task_category_counts = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))
        task_taskname_counts = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))

        total_correct = 0
        total_count = 0

        all_categories = defaultdict(int)
        all_tasknames = defaultdict(int)

        for record in data:
            main_category = record.get('category', '')
            sub_category = record.get('sub-sub-category', '')
            task_name = record.get('task_name', '')

            all_categories[(main_category, sub_category)] += 1
            all_tasknames[(main_category, task_name)] += 1

            task_category_counts[main_category][sub_category]["total"] += 1
            task_taskname_counts[main_category][task_name]["total"] += 1
            total_count += 1

            response = record.get('response', '')
            if self.thinking:
                if '</think>' in response:
                    response = response.split('</think>', 1)[-1]
            try:
                predict = response.strip().replace('\n', '')
            except Exception as e:
                print(f'Error prediction! {e}')
                continue

            model_predict = None
            if predict != 'None' and predict:
                if predict[0] == 'A' or predict[0] == 'B' or predict[0] == 'C' or predict[0] == 'D':
                    model_predict = predict[0]
                elif len(predict) > 1:
                    if predict[-2] == 'A' or predict[-2] == 'B' or predict[-2] == 'C' or predict[-2] == 'D':
                        model_predict = predict[-2]
                    else:
                        print(f'Wrong format response: {predict}')
                        continue
                else:
                    print(f'Wrong format response: {predict}')
                    continue

            answer_gt = record.get('answer_gt', '')
            choices = {
                'A': record.get('choice_a', ''),
                'B': record.get('choice_b', ''),
                'C': record.get('choice_c', ''),
                'D': record.get('choice_d', '')
            }
            is_correct = False

            if model_predict:
                if model_predict == 'A' and choices['A'] == answer_gt:
                    is_correct = True
                elif model_predict == 'B' and choices['B'] == answer_gt:
                    is_correct = True
                elif model_predict == 'C' and choices['C'] == answer_gt:
                    is_correct = True
                elif model_predict == 'D' and choices['D'] == answer_gt:
                    is_correct = True


            if is_correct:
                task_category_counts[main_category][sub_category]["correct"] += 1
                task_taskname_counts[main_category][task_name]["correct"] += 1
                total_correct += 1

        task_category_accuracy = defaultdict(dict)
        task_taskname_accuracy = defaultdict(dict)
        task_average_accuracy_results = defaultdict(dict)

        for task, categories in task_category_counts.items():
            total_correct_for_task = 0
            total_count_for_task = 0

            for category, counts in categories.items():
                total = counts["total"]
                correct = counts["correct"]

                overall_accuracy = correct / total if total > 0 else 0

                task_category_accuracy[task][category] = {
                    'overall_accuracy': overall_accuracy,
                    'correct': correct,
                    'total': total,
                }

                total_correct_for_task += correct
                total_count_for_task += total

            task_average_accuracy_results[task] = {
                "total_correct": total_correct_for_task,
                "total_count": total_count_for_task,
                "overall_accuracy": total_correct_for_task / total_count_for_task if total_count_for_task > 0 else 0,
            }

        for task, tasknames in task_taskname_counts.items():
            for taskname, counts in tasknames.items():
                total = counts["total"]
                correct = counts["correct"]

                overall_accuracy = correct / total if total > 0 else 0

                task_taskname_accuracy[task][taskname] = {
                    'overall_accuracy': overall_accuracy,
                    'correct': correct,
                    'total': total,
                }

        overall_accuracy = total_correct / total_count if total_count > 0 else 0

        detailed_stats = []

        detailed_stats.append("========== Parsing Statistics ==========")
        detailed_stats.append(f"Total records: {total_count}")
        detailed_stats.append(f"Overall accuracy: {overall_accuracy*100:.1f}% ({total_correct}/{total_count})")

        detailed_stats.append("\n========== By Main Category ==========")
        main_category_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        for task, accuracy_info in task_average_accuracy_results.items():
            main_category_stats[task]["correct"] = accuracy_info["total_correct"]
            main_category_stats[task]["total"] = accuracy_info["total_count"]

        for category in sorted(main_category_stats.keys()):
            stats = main_category_stats[category]
            overall_acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0

            detailed_stats.append(f"Main category: {category}")
            detailed_stats.append(f"  Overall accuracy: {overall_acc*100:.1f}% ({stats['correct']}/{stats['total']})")
            detailed_stats.append("")

        detailed_stats.append("========== By Sub-category ==========")
        for (task, category), count in sorted(all_categories.items()):
            stats = task_category_counts[task][category]
            overall_acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0

            detailed_stats.append(f"Main category: {task}, Sub-category: {category}")
            detailed_stats.append(f"  Overall accuracy: {overall_acc*100:.1f}% ({stats['correct']}/{stats['total']})")
            detailed_stats.append("")

        detailed_stats.append("========== By Task Name ==========")
        for (task, taskname), count in sorted(all_tasknames.items()):
            stats = task_taskname_counts[task][taskname]
            overall_acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0

            detailed_stats.append(f"Main category: {task}, Task name: {taskname}")
            detailed_stats.append(f"  Overall accuracy: {overall_acc*100:.1f}% ({stats['correct']}/{stats['total']})")
            detailed_stats.append("")

        return task_category_accuracy, task_average_accuracy_results, overall_accuracy, total_count, detailed_stats

    def generate_json_report(self, data, task_category_accuracy, task_average_accuracy_results, overall_accuracy, total_count):
        json_report = {
            "overall_statistics": {
                "total_samples": total_count,
                "metrics": {
                    "overall_accuracy": overall_accuracy,
                }
            },
            "category_statistics": {},
            "subcategory_statistics": {},
            "task_statistics": {}
        }

        for task, accuracy_info in task_average_accuracy_results.items():
            json_report["category_statistics"][task] = {
                "total_samples": accuracy_info["total_count"],
                "metrics": {
                    "overall_accuracy": accuracy_info["overall_accuracy"],
                }
            }

        for task, categories in task_category_accuracy.items():
            if task not in json_report["subcategory_statistics"]:
                json_report["subcategory_statistics"][task] = {}

            for category, stats in categories.items():
                json_report["subcategory_statistics"][task][category] = {
                    "total_samples": stats["total"],
                    "metrics": {
                        "overall_accuracy": stats["overall_accuracy"],
                    }
                }

        return json_report

    def calculate_metrics(self, output_dir, rank=None, world_size=None):
        output_dir = Path(output_dir)

        if self.thinking:
            base_filename = f"{self.dataset_name.replace('/', '_')}_{self.n_few_shots}_shots_thinking"
        else:
            base_filename = f"{self.dataset_name.replace('/', '_')}_{self.n_few_shots}_shots"
        jsonl_path = output_dir / f"{base_filename}.jsonl"
        json_report_path = output_dir / f"{base_filename}_evaluation_report.json"

        if not jsonl_path.exists():
            print("Final result file does not exist, starting to merge temporary files...")
            try:
                final_output_file, all_predictions = self.merge_results(output_dir)
                print(f"Merge completed, final file: {final_output_file}")
            except Exception as e:
                print(f"Error occurred while merging results: {e}")
                print(f"Error: Result file does not exist: {jsonl_path}")
                return None

        if not jsonl_path.exists():
            print(f"Error: Result file does not exist: {jsonl_path}")
            return None

        print(f"Loading data from {jsonl_path}...")

        data = self.load_jsonl_data(jsonl_path)

        if not data:
            print("Error: No valid data found")
            return None

        print(f"Loaded {len(data)} records")

        task_category_accuracies, task_average_accuracies, overall_accuracy, total_count, _ = self.calculate_accuracy_per_task_and_category(data)

        json_report = self.generate_json_report(
            data,
            task_category_accuracies,
            task_average_accuracies,
            overall_accuracy,
            total_count,
        )

        with open(json_report_path, 'w', encoding='utf-8') as f:
            json.dump(json_report, f, ensure_ascii=False, indent=2)

        print(f"JSON format report saved to: {json_report_path}")

        return {
            'task_category_accuracies': task_category_accuracies,
            'task_average_accuracies': task_average_accuracies,
            'overall_accuracy': overall_accuracy,
            'total_count': total_count,
            'total_records': len(data),
            'json_report': json_report
        }

    def evaluate(self, output_dir, rank=0, world_size=1):
        assert self.model_type == "instruct", "Model type must be instruct"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.thinking:
            base_filename = f"{self.dataset_name.replace('/', '_')}_{self.n_few_shots}_shots_thinking"
        else:
            base_filename = f"{self.dataset_name.replace('/', '_')}_{self.n_few_shots}_shots"
        temp_predictions_file_path = output_dir / f"{base_filename}_rank_{rank}.jsonl"

        predictions_this_rank = []
        processed_item_ids_this_rank = set()

        if temp_predictions_file_path.exists():
            try:
                with open(temp_predictions_file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            item = json.loads(line)
                            predictions_this_rank.append(item)
                            processed_item_ids_this_rank.add(item["id"])
                if processed_item_ids_this_rank:
                    print(f"Rank {rank} resumed from checkpoint: {temp_predictions_file_path}. Processed {len(processed_item_ids_this_rank)} items.")
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load prediction file {temp_predictions_file_path} for rank {rank}: {e}. Starting fresh inference.")
                predictions_this_rank = []
                processed_item_ids_this_rank = set()

        dataset_list = list(self.dataset)
        dataset_shard = dataset_list[rank::world_size]
        items_to_process_this_run = [item for item in dataset_shard if item["id"] not in processed_item_ids_this_rank]

        if rank == 0:
            print(f"\n========== {self.dataset_name} Evaluation ==========\n")
            print(f"Total samples: {len(dataset_list)}, Number of processes: {world_size}")
            print(f"Each process handles approximately {len(dataset_list) // world_size} samples")

        print(f"Rank {rank} will process {len(items_to_process_this_run)} new samples")

        progress_bar = tqdm.tqdm(items_to_process_this_run, desc=f"Evaluating MMSU (Rank {rank})", disable=(rank != 0))

        with open(temp_predictions_file_path, "a", encoding="utf-8") as f:
            for data in progress_bar:
                audio_path = self.dataset.data_root + "/" + data["audio_path"]
                question = data['question']
                choice_a = data['choice_a']
                choice_b = data['choice_b']
                choice_c = data.get('choice_c', None)
                choice_d = data.get('choice_d', None)

                # For UASAudioModel, use strict format: question + \n\n + choices (one per line with A. B. C. D. prefix)
                if isinstance(self.model, UASAudioModel):
                    choices = f'A. {choice_a}\nB. {choice_b}\nC. {choice_c}\nD. {choice_d}'
                    instruction = f"{question}\n\n{choices}"
                else:
                    question_prompts = 'Choose the most suitable answer from options A, B, C, and D to respond the question in next line, **you should only choose A or B or C or D.** Do not provide any additional explanations or content.'
                    choices = f'A. {choice_a}\nB. {choice_b}\nC. {choice_c}\nD. {choice_d}'
                    instruction = f"{question_prompts}\n\nQuestion: {question}\n\n{choices}"

                pred = self.model.audio_understanding_sft(audio_path, input_text=instruction, thinking=self.thinking)

                result = {
                    "id": data["id"],
                    "audio_path": data["audio_path"],
                    "choice_a": choice_a,
                    "choice_b": choice_b,
                    "choice_c": choice_c,
                    "choice_d": choice_d,
                    "input_text": instruction,
                    "answer_gt": data["answer_gt"],
                    "response": pred,
                    "task_name": data["task_name"],
                    "category": data["category"],
                    "sub-category": data["sub-category"],
                    "sub-sub-category": data["sub-sub-category"],
                    "linguistics_sub_discipline": data["linguistics_sub_discipline"],
                }

                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                f.flush()

                predictions_this_rank.append(result)

        print(f"Rank {rank} MMSU inference complete. Processed {len(items_to_process_this_run)} new items. Results saved to: {temp_predictions_file_path}")

        return predictions_this_rank

    def merge_results(self, output_dir):
        output_dir = Path(output_dir)
        all_predictions = []

        if self.thinking:
            base_filename = f"{self.dataset_name.replace('/', '_')}_{self.n_few_shots}_shots_thinking"
        else:
            base_filename = f"{self.dataset_name.replace('/', '_')}_{self.n_few_shots}_shots"
        final_output_file = output_dir / f"{base_filename}.jsonl"

        print("Merging results from all ranks...")

        temp_files = []
        pattern = f"{base_filename}_rank_*.jsonl"
        import glob

        temp_file_pattern = str(output_dir / pattern)
        found_temp_files = glob.glob(temp_file_pattern)

        if not found_temp_files:
            print(f"Warning: No temporary files found in {output_dir} (pattern: {pattern})")
            return final_output_file, []

        found_temp_files.sort(key=lambda x: int(Path(x).stem.split('_rank_')[-1]))

        for temp_file_path_str in found_temp_files:
            temp_file_path = Path(temp_file_path_str)
            temp_files.append(temp_file_path)

            rank_num = temp_file_path.stem.split('_rank_')[-1]

            if temp_file_path.exists():
                with open(temp_file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            all_predictions.append(json.loads(line))
                print(f"Collected rank {rank_num} results: {temp_file_path}")
            else:
                print(f"Warning: File does not exist: {temp_file_path}")

        with open(final_output_file, "w", encoding="utf-8") as f:
            for prediction in all_predictions:
                f.write(json.dumps(prediction, ensure_ascii=False) + '\n')

        print(f"Merge complete! Total {len(all_predictions)} results saved to: {final_output_file}")

        category_counts = defaultdict(int)
        task_counts = defaultdict(int)

        for pred in all_predictions:
            category_counts[pred["category"]] += 1
            task_counts[pred["task_name"]] += 1

        print(f"\n========== Data Statistics ==========")
        print(f"Total samples: {len(all_predictions)}")
        print(f"Category distribution:")
        for category, count in sorted(category_counts.items()):
            print(f"  {category}: {count}")
        print(f"Task distribution:")
        for task, count in sorted(task_counts.items()):
            print(f"  {task}: {count}")

        return final_output_file, all_predictions
