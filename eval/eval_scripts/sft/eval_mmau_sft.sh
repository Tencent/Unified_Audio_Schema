model_path=$1
tokenizer_path=$2
model_name=$3
task="mmau"
dataset=mmau

output_dir="results/${model_name}/eval.sft.${task}.${dataset}"

NUM_GPUS=8

torchrun --nproc_per_node=$NUM_GPUS eval/main.py \
    --model $model_name \
    --model-path $model_path \
    --model-type instruct \
    --tokenizer-path $tokenizer_path \
    --dataset $dataset \
    --task $task \
    --output-dir $output_dir \
    --distribute --exec-mode infer

torchrun --nproc_per_node=$NUM_GPUS eval/main.py \
    --model $model_name \
    --model-path $model_path \
    --model-type instruct \
    --tokenizer-path $tokenizer_path \
    --dataset $dataset \
    --task $task \
    --output-dir $output_dir \
    --distribute --exec-mode calculate
