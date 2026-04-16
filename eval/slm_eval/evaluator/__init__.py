# Copyright 2025 Xiaomi Corporation.
from .asr_evaluator import ASREvaluator
from .tts_evaluator import TTSEvaluator
from .mmsu_evaluator import MMSUEvaluator
from .mmau_evaluator import MMAUEvaluator
from .mmar_evaluator import MMAREvaluator


def get_evaluator(task, model, dataset, n_few_shots=0, device=None, model_type='base', thinking=False):
    print(f"task: {task}")
    if task == "asr":
        return ASREvaluator(model, dataset, model_type=model_type)
    if task == "tts":
        return TTSEvaluator(model, dataset, model_type=model_type, device=device)
    if task == "mmsu":
        return MMSUEvaluator(model, dataset, model_type=model_type, n_few_shots=n_few_shots, thinking=thinking)
    if task == "mmau":
        return MMAUEvaluator(model, dataset, model_type=model_type, n_few_shots=n_few_shots, thinking=thinking)
    if task == "mmar":
        return MMAREvaluator(model, dataset, model_type=model_type, n_few_shots=n_few_shots, thinking=thinking)
