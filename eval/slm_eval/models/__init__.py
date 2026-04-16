# Copyright 2025 Xiaomi Corporation.
def get_model(model, model_type=None, model_path=None, tokenizer_path=None, device=None):
    if model == "uas_audio":
        from .uas_audio import UASAudioModel
        return UASAudioModel(model_path, tokenizer_path)
