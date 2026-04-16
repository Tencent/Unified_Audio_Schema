import os
import sys
from typing import List
import librosa
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, WhisperFeatureExtractor
from .StableToken.src.utils.flow_inference import AudioDecoder


def get_feat_extract_output_lengths(input_mel_lengths):
    input_mel_lengths_leave = input_mel_lengths % 100
    feat_lengths = (input_mel_lengths_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_mel_lengths // 100) * 13
    return output_lengths


class UASAudio:
    def __init__(
            self,
            model_path: str,
            audio_decoder_path: str,
            device: str = "cuda",
    ):
        self.device = device
        self.llm_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, fix_mistral_regex=True)
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, dtype=torch.bfloat16
        ).to(device)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)
        self.sample_rate = self.feature_extractor.sampling_rate
        self.eos_token_id = self.llm_tokenizer.eos_token_id
        self.llm.config.eos_token_id = self.llm_tokenizer.eos_token_id
        self.audio_kwargs = {
            "sampling_rate": self.sample_rate,
            "padding": True,
            "truncation": False,
            "return_attention_mask": True,
            "return_tensors": "pt",
            "stride": 2 * self.feature_extractor.hop_length
        }
        self.audio_offset = self.llm_tokenizer.convert_tokens_to_ids("<|audio_0|>")
        sys.path.insert(0, "./src/StableToken/src")
        sys.path.insert(0, "./src/StableToken/third_party/Matcha-TTS")
        self.audio_decoder = AudioDecoder(
            config_path=os.path.join(audio_decoder_path, "config.yaml"),
            flow_ckpt_path=os.path.join(audio_decoder_path, "flow.pt"),
            hift_ckpt_path=os.path.join(audio_decoder_path, "hift.pt"),
            device=device
        )

    def tokens_to_audio(self, token_ids):
        if len(token_ids) == 0:
            return None, None
        with torch.amp.autocast(self.device):
            tokens = torch.tensor(token_ids, dtype=torch.int32, device=self.device).unsqueeze(0)
            audio_array, sampling_rate = self.audio_decoder.offline_inference(tokens)
        return audio_array, sampling_rate

    def extract_mel_features(self, audio_path_list: List[str]):
        """
        从音频文件路径中提取 mel 频谱特征
        """
        audio_arrays = [librosa.load(audio_path, sr=self.sample_rate)[0] for audio_path in audio_path_list]
        features = self.feature_extractor(audio_arrays, **self.audio_kwargs)
        audio_features = features.input_features
        attention_mask = features.attention_mask
        assert audio_features.shape[-1] == attention_mask.shape[-1]
        return audio_features, attention_mask   # [batch, n_mels, time_steps]

    def __call__(self, messages: list, **kwargs):
        messages, mels, mel_masks = self.apply_chat_template(messages)
        # print(f'Messages for UAS-Audio: {messages}')

        # Tokenize prompts
        prompt_ids = []
        for msg in messages:
            if isinstance(msg, str):
                prompt_ids.append(self.llm_tokenizer(text=msg, return_tensors="pt")["input_ids"])
            elif isinstance(msg, list):
                prompt_ids.append(torch.tensor([msg], dtype=torch.int32))
            else:
                raise ValueError(f"Unsupported content type: {type(msg)}")
        prompt_ids = torch.cat(prompt_ids, dim=-1).to(self.device)
        # print(f'prompt_ids for UAS-Audio: {prompt_ids}')
        attention_mask = torch.ones_like(prompt_ids)

        if mels is None:
            mels_cuda = None
            mel_masks_cuda = None
        else:
            # ensure dtype/device match the model to avoid float vs bfloat16 mismatch
            mels_cuda = mels.to(device=self.llm.device, dtype=self.llm.dtype)
            mel_masks_cuda = mel_masks.to(device=self.llm.device).to(torch.bool)
        # print(f'shape of mels_cuda: {mels_cuda.shape}, shape of mel_masks_cuda: {mel_masks_cuda.shape}')
        # print(f'shape of attention_mask: {attention_mask.shape}, shape of prompt_ids: {prompt_ids.shape}')

        generate_inputs = {
            "input_ids": prompt_ids,
            "attention_mask": attention_mask,
            "mels": mels_cuda,
            "mel_masks": mel_masks_cuda,
        }

        generation_config = dict(
            max_new_tokens=2048,
            pad_token_id=self.llm_tokenizer.pad_token_id,
            eos_token_id=self.eos_token_id,
        )
        generation_config.update(kwargs)
        generation_config = GenerationConfig(**generation_config)

        outputs = self.llm.generate(**generate_inputs, generation_config=generation_config)
        output_token_ids = outputs[0, :-1].tolist()
        output_text_tokens = [i for i in output_token_ids if i < self.audio_offset - 4]
        output_audio_tokens = [i - self.audio_offset for i in output_token_ids if i >= self.audio_offset]
        output_text = self.llm_tokenizer.decode(output_text_tokens)
        return output_token_ids, output_text, output_audio_tokens

    def apply_chat_template(self, messages: list):
        results = []
        audio_path_list = []
        placeholders = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role not in ["user", "assistant", "system"]:
                role = "user"
            if isinstance(content, str):
                prompt = f"<|im_start|>{role}\n{content}<|im_end|>"
                results.append(prompt)
            elif isinstance(content, list):
                parts = [f"<|im_start|>{role}\n"]
                # First, append all audio placeholders in original order
                for item in content:
                    if item["type"] == "audio":
                        audio_path = item['audio']
                        idx = len(audio_path_list)
                        audio_path_list.append(audio_path)
                        ph = f"<<AUDIO_{idx}>>"
                        placeholders.append(ph)
                        parts.append(ph)
                # Then, append non-audio content (text and token) in original order
                for item in content:
                    if item["type"] == "text":
                        parts.append(f"{item['text']}")
                parts.append("<|im_end|>")
                results.append("".join(parts))
            elif content is None:
                results.append(f"<|im_start|>{role}\n")
            else:
                raise ValueError(f"Unsupported content type: {type(content)}")

        if len(audio_path_list) == 0:
            return results, None, None

        features, attention_mask = self.extract_mel_features(audio_path_list)
        mel_lengths = attention_mask.sum(dim=-1).long()

        for i, ph in enumerate(placeholders):
            num_tokens = get_feat_extract_output_lengths(int(mel_lengths[i].item()))
            audio_tokens = "<|AUDIO|>" * num_tokens
            token_str = f"<|audio_bos|>{audio_tokens}<|audio_eos|>"
            results = [seg.replace(ph, token_str) for seg in results]

        return results, features, attention_mask
