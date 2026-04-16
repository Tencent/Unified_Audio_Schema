import json
import os
import re
import tempfile
import torch
import torchaudio
from .src_uas_audio.model import UASAudio

audio_sampling_params = {
    "max_new_tokens": 4096,
    "temperature": 0,
    "do_sample": False
}

text_sampling_params = {
    "max_new_tokens": 4096,
    "temperature": 0,
    "do_sample": False
}

DIALOGUE_SYSTEM_PROMPT = "User will provide you with a speech instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 52 audio tokens. "


class UASAudioModel:
    """
    UASAudioModel is a wrapper around the UASAudio model that provides methods for various tasks such as:
      - instruction_following
      - qa
      - few_shots_qa
      - asr / asr_sft
    """

    def __init__(self, model_path: str, tokenizer_path: str):
        # Load UASAudio
        self.model = UASAudio(model_path, audio_decoder_path=os.path.join(tokenizer_path, "decoder"))
        self.audio_0_id = self.model.audio_offset
        print(f'audio_0_id: {self.audio_0_id}')
        self.history = []

    def tokens_to_audio(self, token_ids):
        return self.model.tokens_to_audio(token_ids)


    @torch.no_grad()
    def gen_text(self, prompts):
        messages = []
        for prompt in prompts:
            # prompt is a tuple: (content, mode, type)
            # content is a dict with "speech" or "text" key
            # mode is "speech" or "text"
            # type is "input" or "output"
            role = "user" if prompt[2] == "input" else "assistant"

            if prompt[1] == "speech":
                # For speech input, content should be a list with audio dict
                messages.append({
                    "role": role,
                    "content": [{"type": "audio", "audio": str(prompt[0]["speech"])}]
                })
            else:
                # For text input, content should be a string
                messages.append({
                    "role": role,
                    "content": [{"type": "text", "text": prompt[0]["text"]}]
                })
        # Add assistant response marker so the last user message doesn't get <|im_end|>
        messages.append({"role": "assistant", "content": None})
        # Use local copy to avoid modifying global dict
        local_params = text_sampling_params.copy()
        local_params["do_sample"] = False
        local_params["max_new_tokens"] = 10
        # print(f'messages: {messages}')
        _, text, _ = self.model(messages, **local_params)
        return text

    @torch.no_grad()
    def gen_speech(self, prompts, output_audio_path, system_prompt=DIALOGUE_SYSTEM_PROMPT):
        messages = [{"role": "system", "content": system_prompt}]
        for prompt in prompts:
            # prompt is a tuple: (content, mode, type)
            # content is a dict with "speech" or "text" key
            # mode is "speech" or "text"
            # type is "input" or "output"
            role = "user" if prompt[2] == "input" else "assistant"

            if prompt[1] == "speech":
                # For speech input, content should be a list with audio dict
                messages.append({
                    "role": role,
                    "content": [{"type": "audio", "audio": str(prompt[0]["speech"])}]
                })
            else:
                # For text input, content should be a list with text dict (consistent with speech format)
                messages.append({
                    "role": role,
                    "content": [{"type": "text", "text": prompt[0]["text"]}]
                })

        # Add assistant response marker for speech generation
        messages.append({"role": "assistant", "content": None})

        # Use local copy to avoid modifying global dict
        local_params = audio_sampling_params.copy()
        local_params["max_new_tokens"] = 100
        _, text, audio_tokens = self.model(messages, **local_params)

        if len(audio_tokens) == 0:
            audio = torch.zeros(22050)
            torchaudio.save(output_audio_path, audio, 22050)
        else:
            audio_array, sampling_rate = self.tokens_to_audio(audio_tokens)
            torchaudio.save(output_audio_path, audio_array, sampling_rate)
        return text

    def detect_language(self, text):
        """Detect language of text (Chinese or English)"""
        chinese = re.findall(r'[\u4e00-\u9fff]', text)
        return "zh" if chinese else "en"

    @torch.no_grad()
    def tts_sft(self, text, output_path, instruction=None):
        """Text-to-speech synthesis using UASAudio model"""
        prompt = "convert the following text into audio"
        messages = [{"role": "system", "content": f'{prompt}: {text}'}]
        messages.append({"role": "assistant", "content": None})
        _, text, audio_tokens = self.model(messages, **audio_sampling_params)
        audio_array, sampling_rate = self.tokens_to_audio(audio_tokens)

        if output_path is not None:
            torchaudio.save(output_path, audio_array, sampling_rate)
        return text

    @torch.no_grad()
    def convert_instructions(self, instructions, td):
        messages = []  # No system prompt
        sound = "<sound>"
        audio_wavs_index = 0
        for ins in instructions:
            if ins["from"] == "human":
                # Extract text and audio from value list
                text_value = None
                audio_value = None
                for item in ins["value"]:
                    if item["type"] == "text":
                        text_value = item["value"]
                    elif item["type"] == "sound":
                        audio_value = item["value"]

                content_list = []

                # Handle audio first (audio before text)
                if audio_value is not None:
                    if isinstance(audio_value, str):
                        content_list.append({"type": "audio", "audio": audio_value})
                    else:
                        audio_path = os.path.join(td, str(audio_wavs_index) + '.wav')
                        torchaudio.save(audio_path, audio_value, 24000)
                        audio_wavs_index += 1
                        content_list.append({"type": "audio", "audio": audio_path})

                # Handle text
                if text_value is not None:
                    if sound in text_value:
                        # Split only on first occurrence to handle multiple <sound> tags correctly
                        parts = text_value.split(sound, 1)
                        left_text = parts[0]
                        right_text = parts[1] if len(parts) > 1 else ""
                        if left_text != "":
                            content_list.append({"type": "text", "text": left_text})
                        if right_text != "":
                            content_list.append({"type": "text", "text": right_text})
                    else:
                        # No <sound> tag, use text as is
                        content_list.append({"type": "text", "text": text_value})

                messages.append({"role": "user", "content": content_list})
                messages.append({"role": "assistant", "content": None})
        return messages

    @torch.no_grad()
    def instruction_following(self, instructions, append_generation_prompt=False, thinking=False):
        with tempfile.TemporaryDirectory() as td:
            messages = self.convert_instructions(instructions, td)
            _, text, _ = self.model(messages, **text_sampling_params)
            return text

    @torch.no_grad()
    def audio_understanding_sft(self, audio, input_text, thinking=False):
        messages = []
        # Audio first, then text
        messages.append({"role": "user", "content": [
            {"type": "audio", "audio": audio},
            {"type": "text", "text": input_text}
        ]})
        messages.append({"role": "assistant", "content": None})
        _, text, _ = self.model(messages, **text_sampling_params)
        # Use replace instead of strip to remove substrings, not just characters
        text = text.replace("<think>", "").replace("</think>", "")
        return text.strip()

    @torch.no_grad()
    def asr_sft(self, audio, lang='zh', max_new_tokens=128, num_beams=1):
        """
        ASR inference with configurable decoding strategy.

        Args:
            audio: Audio file path or audio array
            lang: Language code ('zh' or 'en')
            max_new_tokens: Maximum number of tokens to generate
            num_beams: Number of beams for beam search (1 = greedy decoding)
        """
        messages = []
        messages.append({"role": "user", "content": [
            {"type": "audio", "audio": audio},
            {"type": "text", "text": "Transcribe the following audio content into text."}
        ]})
        messages.append({"role": "assistant", "content": None})

        # Use greedy decoding (num_beams=1) or beam search (num_beams>1)
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "repetition_penalty": 1,
            "num_beams": num_beams,
        }

        # Add early_stopping for beam search to improve efficiency
        if num_beams > 1:
            generation_kwargs["early_stopping"] = True

        _, text, _ = self.model(messages, **generation_kwargs)
        return text.strip()

    @torch.no_grad()
    def gen_uas(self, audio):
        messages = []
        messages.append({"role": "user", "content": [
            {"type": "audio", "audio": audio},
            {"type": "text", "text": "Based on the audio input, output a UAD JSON with transcription, voice features, and non-linguistic events."}
        ]})
        messages.append({"role": "assistant", "content": None})
        _, text, _ = self.model(messages, max_new_tokens=2048)
        return text.strip()

    @torch.no_grad()
    def asr_by_uas(self, audio, lang='en'):
        uas_json_str = self.gen_uas(audio).strip("```json").strip("```").strip()
        try:
            uas_data = json.loads(uas_json_str)
            transcription = uas_data.get("transcription", "")
        except json.JSONDecodeError:
            # Try to extract transcription from string using regex
            print(f"Failed to parse UAS JSON, trying regex extraction: {uas_json_str}")
            match = re.search(r'"transcription"\s*:\s*"([^"]*)"', uas_json_str)
            if match:
                transcription = match.group(1)
                print(f"Extracted transcription using regex: {transcription}")
            else:
                transcription = ""
                print("Failed to extract transcription using regex.")

        return transcription
