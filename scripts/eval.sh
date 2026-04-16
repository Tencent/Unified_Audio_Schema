#!/bin/bash
set -e # Stop execution on any error

# ==============================================================================
# UAS-Audio Evaluation Script
# ==============================================================================

# Default relative paths for the model and tokenizer (open-source friendly)
DEFAULT_MODEL_PATH="checkpoints/UAS-Audio"
DEFAULT_TOKENIZER_PATH="checkpoints/StableToken"
MODEL_NAME="uas_audio"

# Support overriding default paths via command-line arguments, e.g.:
# bash scripts/eval.sh ./path_to_my_model ./path_to_my_tokenizer
MODEL_PATH=${1:-$DEFAULT_MODEL_PATH}
TOKENIZER_PATH=${2:-$DEFAULT_TOKENIZER_PATH}

# --------- Convert to absolute paths ---------
# The underlying evaluation scripts rely on absolute paths to avoid path issues
ABS_MODEL_PATH=$(realpath "$MODEL_PATH")
ABS_TOKENIZER_PATH=$(realpath "$TOKENIZER_PATH")

# Define the directory where evaluation scripts are located (relative to project root)
SCRIPT_DIR="eval/eval_scripts/sft"

echo "========================================================================"
echo " 🚀 Launching UAS-Audio Evaluation Process..."
echo " Absolute Model Path:     $ABS_MODEL_PATH"
echo " Absolute Tokenizer Path: $ABS_TOKENIZER_PATH"
echo " Model Name:              $MODEL_NAME"
echo "========================================================================"

# Encapsulate the invocation logic for clarity and formatted output
run_eval() {
    local task_name=$1
    local script_name=$2

    echo ""
    echo "--------------------------------------------------"
    echo "▶️  [START] Evaluating on ${task_name}..."
    echo "--------------------------------------------------"

    bash "${SCRIPT_DIR}/${script_name}" \
        "$ABS_MODEL_PATH" \
        "$ABS_TOKENIZER_PATH" \
        "$MODEL_NAME"

    echo "✅  [DONE] Evaluation for ${task_name} completed successfully."
}

# --------- Core Evaluation Task Sequence ---------

run_eval "MMSU"                    "eval_mmsu_sft.sh"
run_eval "MMAR"                    "eval_mmar_sft.sh"
run_eval "MMAU"                    "eval_mmau_sft.sh"
run_eval "AISHELL-1"               "eval_asr_aishell1_test.sh"
run_eval "LibriSpeech-test-clean"  "eval_asr_librispeech_test_clean.sh"
run_eval "SeedTTS-en"              "eval_tts_seedtts_en.sh"
run_eval "SeedTTS-zh"              "eval_tts_seedtts_zh.sh"

# ------------------------------------------------

echo ""
echo "========================================================================"
echo "🎉 All evaluation tasks have been successfully completed!"
echo "========================================================================"
