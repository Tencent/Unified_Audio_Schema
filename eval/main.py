# Copyright 2025 Xiaomi Corporation.
import argparse
import os
import torch
import torch.distributed as dist
from slm_eval.models import get_model
from slm_eval.datasets import get_dataset
from slm_eval.evaluator import get_evaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Name of the model.")
    parser.add_argument("--model-type", help="Model type (base or instruct).", default='base')
    parser.add_argument("--model-path", help="Path to the model weights (Optional).")
    parser.add_argument("--tokenizer-path", help="Path to the tokenizer (Optional).")
    parser.add_argument("--dataset", required=True, help="Name of the dataset.")
    parser.add_argument("--split", help="Dataset split to evaluate on (e.g., test).")
    parser.add_argument("--task", required=True, help="Specific task identifier.")
    parser.add_argument("--n-few-shots", type=int, default=0, help="Number of examples to use (only for base model).")
    parser.add_argument("--thinking", default=False, help="Whether to use thinking in audio understanding tasks.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save evaluation results.")
    parser.add_argument("--sample-rate", type=int, default=24000, help="Sample rate for resampling (Optional).")
    parser.add_argument("--exec-mode", type=str, default="infer", choices=["infer", "calculate"],
                        help="Execution mode: 'infer' only, 'calculate' metrics only.")
    parser.add_argument("--distribute", action="store_true", help="Enable distributed evaluation.")
    args = parser.parse_args()

    rank = 0
    world_size = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    process_group_initialized = False

    if args.distribute:
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ["LOCAL_RANK"])
            if torch.cuda.is_available():
                device = torch.device(f"cuda:{local_rank}")
                torch.cuda.set_device(device)
                dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
            else:
                device = torch.device("cpu")
                dist.init_process_group(backend='gloo', init_method='env://',
                                    world_size=world_size, rank=rank)
            process_group_initialized = True
        else:
            print("Warning: Distributed mode requested but environment variables for distributed training are not set.")
            print("Running on single GPU/CPU.")
            args.distribute = False

    print(f"args.distribute: {args.distribute}")
    print(f"world_size: {world_size}")

    model_name = args.model
    model_type = args.model_type
    model_path = args.model_path
    tokenizer_path = args.tokenizer_path
    dataset_name = args.dataset
    split = args.split
    task = args.task
    n_few_shots = args.n_few_shots
    output_dir = args.output_dir
    sample_rate = args.sample_rate

    model = get_model(model_name, model_type, model_path, tokenizer_path, device) if args.exec_mode == "infer" else None

    dataset = get_dataset(dataset_name, split, sample_rate)

    evaluator = get_evaluator(task, model, dataset, n_few_shots, device, model_type=model_type, thinking=args.thinking)

    if args.exec_mode == "infer":
        if rank == 0:
            print(f"\n========== Starting Inference ==========\n")
        evaluator.evaluate(output_dir=output_dir, rank=rank, world_size=world_size)

    if args.exec_mode == "calculate":
        if rank == 0:
            print(f"\n========== Starting Metric Calculation ==========\n")
        evaluator.calculate_metrics(output_dir=output_dir, rank=rank, world_size=world_size)


    if process_group_initialized:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
