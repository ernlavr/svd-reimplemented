import argparse
from typing import Dict

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
)

from svdllm.compression import collect_whitening_matrices, compress_model_svdllm


def build_dataloader(
    tokenizer,
    dataset_name: str,
    dataset_config: str,
    split: str,
    num_samples: int,
    max_seq_len: int,
    batch_size: int,
):
    ds = load_dataset(dataset_name, dataset_config, split=split)
    if num_samples is not None and num_samples > 0:
        ds = ds.select(range(min(num_samples, len(ds))))

    text_column = "text"
    if text_column not in ds.column_names:
        # Heuristic for some datasets where field name may differ
        for cand in ["content", "sentence", "document"]:
            if cand in ds.column_names:
                text_column = cand
                break

    def tokenize_fn(examples: Dict[str, str]):
        return tokenizer(
            examples[text_column],
            truncation=True,
            max_length=max_seq_len,
            padding="max_length",
            return_tensors="pt",
        )

    ds = ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names)

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=default_data_collator,
    )
    return dl


def parse_args():
    parser = argparse.ArgumentParser(description="Compress a HF causal LM with SVD-LLM")
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Hugging Face model name or path (e.g., meta-llama/Meta-Llama-3-8B)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="wikitext",
        help="HF dataset name (default: wikitext)",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="wikitext-2-raw-v1",
        help="HF dataset config name (default: wikitext-2-raw-v1)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use for calibration (default: train)",
    )
    parser.add_argument(
        "--num-calib-samples",
        type=int,
        default=256,
        help="Number of calibration sentences to use (default: 256)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=256,
        help="Maximum sequence length for calibration (default: 256)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Calibration batch size (default: 8)",
    )
    parser.add_argument(
        "--compression-ratio",
        type=float,
        default=0.4,
        help="Desired per-layer weight compression ratio R_w in [0, 1). "
        "E.g., 0.4 means 40%% compression (60%% of params kept).",
    )
    parser.add_argument(
        "--min-rank",
        type=int,
        default=4,
        help="Minimum rank for any compressed layer (default: 4)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the compressed model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for calibration and compression",
    )
    parser.add_argument(
        "--max-calib-steps",
        type=int,
        default=None,
        help="Optional max number of calibration batches (if None, use all)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = args.device

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16)

    dataloader = build_dataloader(
        tokenizer=tokenizer,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        num_samples=args.num_calib_samples,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
    )

    # 1) Collect whitening matrices
    whitening = collect_whitening_matrices(
        model=model,
        dataloader=dataloader,
        device=device,
        modules=None,
        max_steps=args.max_calib_steps,
    )

    # 2) Compress the model in-place
    compress_model_svdllm(
        model=model,
        whitening_mats=whitening,
        compression_ratio=args.compression_ratio,
        device=device,
        min_rank=args.min_rank,
    )

    # 3) Save compressed model
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Compressed model saved to {args.output_dir}")


if __name__ == "__main__":
    main()

