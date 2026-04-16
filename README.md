<div align="center">

<img src="assets/unified-audio-schema-logo.png" width="650"/>

# Beyond Transcription: Unified Audio Schema for Perception-Aware AudioLLMs

### WeChat AI

<br>

<p>
<a href="https://arxiv.org/abs/2604.12506"><img src="https://img.shields.io/badge/📄%20Paper-arXiv-b31b1b.svg?style=for-the-badge" alt="Paper"></a>
&nbsp;
<a href="LICENSE"><img src="https://img.shields.io/badge/License-License%20Term%20of%20Unified_Audio_Schema-green.svg?style=for-the-badge" alt="License"></a>
&nbsp;
<a href="https://www.huggingface.co/tencent/Unified_Audio_Schema"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow?style=for-the-badge" alt="Model"></a>
</p>

</div>

---

## 📢 News

| Date | News |
|:-----|:-------|
| **2026-04-16** | 🚀 Initial release of our code and model on [GitHub](https://github.com/Tencent/Unified_Audio_Schema) and [HuggingFace](https://huggingface.co/tencent/Unified_Audio_Schema)! |
| **2026-04-07** | 📄 Our paper has been accepted to ACL 2026 Findings! |

---

## 💡 What is Unified Audio Schema?

**Unified Audio Schema** is a novel holistic framework for audio supervision that disentangles and restructures supervision across **transcription**, **paralinguistics**, and **non-linguistic events**.
- 🧩 Using the **Unified Audio Schema** framework, we trained an advanced Audio Large Language Model (AudioLLM) built upon Qwen2.5-7B with an Audio Transformer (AuT) encoder.
- 🏆 Our model achieves **better audio perception and understanding** beyond traditional ASR-centric systems through this unified supervision approach.

---

## 🚀 Quick Start

### Installation

Clone the repository with its submodules and set up the environment:

```bash
# 1. Clone the repository including all submodules
git clone --recursive https://github.com/Tencent/Unified_Audio_Schema.git
cd Unified_Audio_Schema

# If you have already cloned without --recursive, initialize submodules with:
git submodule update --init --recursive

# 2. Create a conda environment
conda create -n unified-audio python=3.12.12 -y
conda activate unified-audio

# 3. Install dependencies
pip install -r requirements.txt
```

### Download Checkpoints

To run our AudioLLM, you need the [model weights](https://huggingface.co/tencent/Unified_Audio_Schema) and the speech tokenizer [StableToken](https://github.com/Tencent/StableToken) used for speech generation tasks.

```bash
# Download our model weights
huggingface-cli download tencent/Unified_Audio_Schema --local-dir checkpoints/Unified_Audio_Schema

# Download StableToken (required for speech generation)
huggingface-cli download tencent/StableToken --local-dir checkpoints/StableToken
```

---

## 💻 Usage

We provide a comprehensive [Jupyter Notebook](example_usage.ipynb) demonstrating the core capabilities of our model, including unified audio understanding and interleaved text/speech generation.

It details the following multi-modal inference scenarios:
1. **Text-input Conversation**: Feed text instructions and receive interleaved text with corresponding speech output.
2. **Speech-input Conversation**: Provide speech audio directly and receive context-aware interleaved text and speech responses.
3. **Automatic Speech Recognition (ASR)**: Transcribe speech audio into high-precision text.
4. **Audio Captioning**: Intuitively describe the semantic context or background events present within an audio file.
5. **Text-to-Speech (TTS)**: Synthesize dynamic natural speech from plain text instructions.

---

## 📊 Evaluation & Performance

Our evaluation pipeline (located in [`eval`](eval) directory) is based on [Mimo-Audio-Eval](https://github.com/XiaomiMiMo/MiMo-Audio-Eval) and adapted for our model, covering Automatic Speech Recognition (ASR), Text-to-Speech (TTS), and audio understanding benchmarks.

### Running Evaluation

You can run the full suite of benchmarks using our script:

```bash
# Download the evaluation datasets
cd eval
python download_data.py

# Run the evaluation script
cd ..
bash scripts/eval.sh
```

This will automatically evaluate across:
- Audio Understanding (MMSU, MMAR, MMAU)
- ASR (AISHELL-1, LibriSpeech-test-clean)
- TTS (SeedTTS-en, SeedTTS-zh)

### Results

Our model achieves state-of-the-art performance across several audio understanding benchmarks, demonstrating significant improvements in audio perception and understanding capabilities compared to baseline models.

| **Model** | MMSU<br>(Percep.) | MMSU<br>(Reason.) | **MMSU<br>(Overall)** | MMAR<br>(Speech) | MMAR<br>(Sound) | MMAR<br>(Music) | **MMAR<br>(Overall)** | MMAU<br>(Speech) | MMAU<br>(Sound) | MMAU<br>(Music) | **MMAU<br>(Overall)** | **Avg.** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [Kimi-Audio](https://github.com/MoonshotAI/Kimi-Audio) | <u>44.8</u> | 75.7 | <u>59.8</u> | 58.5 | 49.7 | 33.0 | 48.0 | 62.2 | 75.7 | 66.8 | 68.2 | 58.7 |
| [Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni) | 42.7 | **77.6** | 58.1 | 59.9 | **58.8** | 40.8 | 56.7 | **70.6** | <u>78.1</u> | 65.9 | <u>71.5</u> | <u>62.1</u> |
| [Step-Audio2](https://github.com/stepfun-ai/Step-Audio2) | 42.9 | 73.2 | 57.6 | <u>61.2</u> | 54.6 | <u>42.2</u> | <u>56.8</u> | <u>68.2</u> | **79.3** | <u>68.4</u> | **72.7** | 61.9 |
| **Ours** | **55.7** | <u>77.4</u> | **66.2** | **66.0** | **58.8** | **45.2** | **60.1** | 67.0 | 70.0 | **71.3** | 69.4 | **65.2** |



Our model also demonstrates leading or competitive performance on ASR and TTS benchmarks, showing its versatility across a wide range of audio tasks (measured in WER, lower is better):

| Model | ASR<br>(LS-clean) | ASR<br>(AISHELL-1) | TTS<br>(SeedTTS-en) | TTS<br>(SeedTTS-zh) |
| :--- | :---: | :---: | :---: | :---: |
| [Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni) | - | - | 2.3 | 1.4 |
| [Step-Audio2](https://github.com/stepfun-ai/Step-Audio2) | 1.9 | 1.0 | 2.1 | 3.2 |
| [MiMo-Audio](https://github.com/XiaomiMiMo/MiMo-Audio) | 3.8 | 1.8 | 5.4 | 2.0 |
| **Ours** | 2.2 | 2.3 | 1.7 | 1.4 |

---

## 🙏 Acknowledgements

- The speech generation stage of our model utilizes discrete tokens from [StableToken](https://github.com/Tencent/StableToken).
- Special thanks to [Mimo-Audio-Eval](https://github.com/XiaomiMiMo/MiMo-Audio-Eval) for the comprehensive evaluation toolkit.

---

## 📜 Citation

If you find Unified Audio Schema or our model useful for your research, please cite:

```bibtex
@misc{zhang2026transcriptionunifiedaudioschema,
  title={Beyond Transcription: Unified Audio Schema for Perception-Aware AudioLLMs}, 
  author={Linhao Zhang and Yuhan Song and Aiwei Liu and Chuhan Wu and Sijun Zhang and Wei Jia and Yuan Liu and Houfeng Wang and Xiao Zhou},
  year={2026},
  eprint={2604.12506},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2604.12506},
}

@inproceedings{song2026stabletoken,
  title={StableToken: A Noise-Robust Semantic Speech Tokenizer for Resilient Speech{LLM}s},
  author={Yuhan Song and Linhao Zhang and Chuhan Wu and Aiwei Liu and Wei Jia and Houfeng Wang and Zhou Xiao},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=17DNmdQ9aU}
}
```

---

## 📄 License

This project is licensed under the [License Term of Unified_Audio_Schema](LICENSE).
