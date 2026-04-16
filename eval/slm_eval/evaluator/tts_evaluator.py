# Copyright 2025 Xiaomi Corporation.
import json
from pathlib import Path
import string
from funasr import AutoModel
from jiwer import compute_measures
import tqdm
import whisper
import zhconv
from zhon.hanzi import punctuation


class TTSEvaluator:
    def __init__(self, model, dataset, model_type, device):
        self.model = model
        self.dataset = dataset
        self.punctuation_all = punctuation + string.punctuation
        self.device = device
        self.model_type = model_type
        if self.dataset.lang == "en":
            self.asr_model = self.load_en_asr_model()
        elif self.dataset.lang == "zh":
            self.asr_model = self.load_zh_asr_model()
        else:
            raise NotImplementedError("Language not supported")

    def load_en_asr_model(self):
        model = whisper.load_model("large-v3").to(self.device)
        return model

    def load_zh_asr_model(self):
        device_str = str(self.device)
        model = AutoModel(model="paraformer-zh", device=device_str)
        return model

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

    def asr(self, audio_path):
        if self.dataset.lang == "en":
            transcription = self.asr_model.transcribe(audio_path)["text"]
        elif self.dataset.lang == "zh":
            res = self.asr_model.generate(input=audio_path, batch_size_s=300)
            transcription = res[0]["text"]
            transcription = zhconv.convert(transcription, 'zh-cn')
        else:
            raise NotImplementedError("Language not supported")
        return transcription.strip()

    def evaluate(self, output_dir, rank=0, world_size=1):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        wav_dir = output_dir / "wav"
        wav_dir.mkdir(parents=True, exist_ok=True)
        results_per_rank = []

        dataset_shard = list(self.dataset)[rank::world_size]

        if rank == 0:
            print("\n========== TTS Inference ==========\n")

        total_samples = len(dataset_shard)
        progress_bar = tqdm.tqdm(dataset_shard, desc=f"Running TTS Inference (Rank {rank})", disable=(rank != 0))

        for i, data in enumerate(progress_bar):
            ref = data["text"]
            output_audio_path = wav_dir / f"{data['id']}.wav"

            if not output_audio_path.exists():
                if self.model_type == "instruct":
                    self.model.tts_sft(ref, output_path=output_audio_path)
                else:
                    self.model.tts(ref, output_audio_path)
            else:
                print(f"{output_audio_path} exists, skipped.")

            if rank == 0:
                print(f"\n----- Sample {i+1}/{total_samples} ID: {data['id']} (Rank {rank}) -----")
                print(f"Reference: {ref}")
                print(f"Audio: {output_audio_path}")
                print("-" * 70)

            results_per_rank.append({
                "id": data["id"],
                "ref": ref,
                "audio": str(output_audio_path.resolve()),
            })

        progress_bar = tqdm.tqdm(results_per_rank, desc=f"Running ASR (Rank {rank})", disable=(rank != 0))
        for data in progress_bar:
            hyp = self.asr(data["audio"])
            wer, measures = self.wer(hyp, data["ref"])
            data["hyp"] = hyp
            data["wer"] = wer
            data["substitutions"] = measures["substitutions"]
            data["insertions"] = measures["insertions"]
            data["deletions"] = measures["deletions"]

        with open(output_dir / f"tts_results_rank_{rank}.json", "w", encoding="utf-8") as f:
            json.dump(results_per_rank, f, indent=4, ensure_ascii=False)

        print(f"Rank {rank} TTS inference complete. Results saved to {output_dir / f'tts_results_rank_{rank}.json'}")

    def calculate_metrics(self, output_dir, rank=0, world_size=1):
        if rank != 0:
            return None

        output_dir = Path(output_dir)
        all_results = []
        for r in range(world_size):
            rank_file = output_dir / f"tts_results_rank_{r}.json"
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

        with open(output_dir / "tts_results.json", "w", encoding="utf-8") as f:
            json.dump(full_results, f, indent=4, ensure_ascii=False)

        print("\n========== ASR Evaluation Summary ==========\n")
        print(f"Language: {getattr(self.dataset, 'lang', 'unknown')}")
        print(f"Average WER: {avg_wer:.4f}")
        print(f"Error breakdown: {total_subs} substitutions, {total_ins} insertions, {total_del} deletions")

        return summary
