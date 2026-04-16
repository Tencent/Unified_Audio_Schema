# Copyright 2025 Xiaomi Corporation.
import json
import string
from pathlib import Path
from jiwer import compute_measures
import tqdm
from zhon.hanzi import punctuation


class ASREvaluator:
    def __init__(self, model, dataset, model_type):
        self.model = model
        self.dataset = dataset
        self.punctuation_all = punctuation + string.punctuation
        self.model_type = model_type

    def wer(self, hyp, ref):
        for x in self.punctuation_all:
            if x == "'":
                continue
            ref = ref.replace(x, '')
            hyp = hyp.replace(x, '')
        ref = ref.replace('  ', ' ')
        hyp = hyp.replace('  ', ' ')

        if self.dataset.lang == "zh":
            ref = " ".join([x for x in ref])
            hyp = " ".join([x for x in hyp])
        elif self.dataset.lang == "en":
            ref = ref.lower()
            hyp = hyp.lower()
        else:
            raise NotImplementedError("Language not supported")

        measures = compute_measures(ref, hyp)
        wer = measures["wer"]
        return wer, measures

    def evaluate(self, output_dir, rank=0, world_size=1):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results_per_rank = []

        dataset_shard = list(self.dataset)[rank::world_size]

        if rank == 0:
            print("\n========== ASR Inference ==========\n")

        total_samples = len(dataset_shard)
        progress_bar = tqdm.tqdm(dataset_shard, desc=f"Running ASR Inference (Rank {rank})", disable=(rank != 0))

        for i, data in enumerate(progress_bar):
            audio_path = data["audio"]
            ref = data["text"]
            if self.model_type == "instruct":
                hyp = self.model.asr_sft(audio_path, lang=self.dataset.lang).split("(")[0]
            else:
                hyp = self.model.asr(audio_path).split("(")[0]
            wer, measures = self.wer(hyp, ref)

            if rank == 0:
                print(f"\n----- Sample {i+1}/{total_samples} ID: {data['id']} (Rank {rank}) -----")
                print(f"Reference: {ref}")
                print(f"Hypothesis: {hyp}")
                print(f"WER: {wer:.4f} ({measures['substitutions']} sub, {measures['insertions']} ins, {measures['deletions']} del)")
                print("-" * 70)

            results_per_rank.append({
                "id": data["id"],
                "ref": ref,
                "hyp": hyp,
                "wer": wer,
                "substitutions": measures["substitutions"],
                "insertions": measures["insertions"],
                "deletions": measures["deletions"],
            })

        with open(output_dir / f"asr_results_rank_{rank}.json", "w", encoding="utf-8") as f:
            json.dump(results_per_rank, f, indent=4, ensure_ascii=False)

        print(f"Rank {rank} ASR inference complete. Results saved to {output_dir / f'asr_results_rank_{rank}.json'}")

    def calculate_metrics(self, output_dir, rank=0, world_size=1):
        if rank != 0:
            return None

        output_dir = Path(output_dir)
        all_results = []
        for r in range(world_size):
            rank_file = output_dir / f"asr_results_rank_{r}.json"
            if rank_file.exists():
                with open(rank_file, "r", encoding="utf-8") as f:
                    all_results.extend(json.load(f))
            else:
                print(f"Warning: Results file {rank_file} not found.")

        avg_wer = sum([result["wer"] for result in all_results]) / len(all_results)
        total_subs = sum([result["substitutions"] for result in all_results])
        total_ins = sum([result["insertions"] for result in all_results])
        total_del = sum([result["deletions"] for result in all_results])

        summary = {
            "avg_wer": avg_wer,
            "total_samples": len(all_results),
            "total_substitutions": total_subs,
            "total_insertions": total_ins,
            "total_deletions": total_del,
            "language": getattr(self.dataset, "lang", "unknown"),
        }

        full_results = {
            "summary": summary,
            "results": all_results
        }

        with open(output_dir / "asr_results.json", "w", encoding="utf-8") as f:
            json.dump(full_results, f, indent=4, ensure_ascii=False)

        print("\n========== ASR Evaluation Summary ==========\n")
        print(f"Language: {getattr(self.dataset, 'lang', 'unknown')}")
        print(f"Average WER: {avg_wer:.4f}")
        print(f"Error breakdown: {total_subs} substitutions, {total_ins} insertions, {total_del} deletions")
        print(f"\nResults saved to {output_dir / 'asr_results.json'}")

        return summary
