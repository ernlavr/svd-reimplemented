from __future__ import annotations

import argparse
import math
import os
from typing import Dict, Optional, Tuple

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
)

from .compression import collect_whitening_matrices, compress_model_svdllm


def build_text_dataloader(
    tokenizer,
    dataset_name: str,
    dataset_config: str,
    split: str,
    num_samples: Optional[int],
    max_seq_len: int,
    batch_size: int,
) -> DataLoader:
    """
    Generic text dataloader used for both calibration and evaluation.
    """
    ds = load_dataset(dataset_name, dataset_config, split=split)
    if num_samples is not None and num_samples > 0:
        ds = ds.select(range(min(num_samples, len(ds))))

    text_column = "text"
    if text_column not in ds.column_names:
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


def evaluate_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset_name: str,
    dataset_config: str,
    split: str,
    num_eval_samples: Optional[int],
    max_seq_len: int,
    batch_size: int,
    device: str,
) -> Tuple[float, float]:
    """
    Run perplexity evaluation for a given model/tokenizer pair.
    """
    print(f"[SVD-LLM] Building evaluation dataloader for split='{split}'.")
    dataloader = build_text_dataloader(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split,
        num_samples=num_eval_samples,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
    )

    print("[SVD-LLM] Starting evaluation.")
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    from tqdm.auto import tqdm

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="[SVD-LLM] Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["input_ids"].clone()
            batch["labels"] = labels
            outputs = model(**batch)
            loss = outputs.loss
            non_pad = (labels != tokenizer.pad_token_id).sum()
            total_loss += loss.item() * non_pad.item()
            total_tokens += non_pad.item()

    avg_nll = total_loss / max(1, total_tokens)
    ppl = math.exp(avg_nll)

    print(f"[SVD-LLM] Average NLL: {avg_nll:.4f}")
    print(f"[SVD-LLM] Perplexity: {ppl:.4f}")
    return avg_nll, ppl


def add_compress_args(parser: argparse.ArgumentParser) -> None:
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


def run_compress_from_args(args: argparse.Namespace) -> None:
    device = args.device
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    print(f"[SVD-LLM] Loading tokenizer for '{args.model_name}'.")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[SVD-LLM] Loading model '{args.model_name}'.")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if "cuda" in str(device) else torch.float32,
        token=hf_token,
    )

    print("[SVD-LLM] Building calibration dataloader.")
    dataloader = build_text_dataloader(
        tokenizer=tokenizer,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        num_samples=args.num_calib_samples,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
    )

    print("[SVD-LLM] Collecting whitening statistics.")
    whitening = collect_whitening_matrices(
        model=model,
        dataloader=dataloader,
        device=device,
        modules=None,
        max_steps=args.max_calib_steps,
    )

    print("[SVD-LLM] Running SVD-LLM compression.")
    compress_model_svdllm(
        model=model,
        whitening_mats=whitening,
        compression_ratio=args.compression_ratio,
        device=device,
        min_rank=args.min_rank,
    )

    print(f"[SVD-LLM] Saving compressed model to '{args.output_dir}'.")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Compressed model saved to {args.output_dir}")


def add_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        required=True,
        help="Model name or local path (original or compressed).",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="wikitext",
        help="HF dataset name (default: wikitext).",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="wikitext-2-raw-v1",
        help="HF dataset config (default: wikitext-2-raw-v1).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split to use for evaluation (default: validation).",
    )
    parser.add_argument(
        "--num-eval-samples",
        type=int,
        default=None,
        help="Optional cap on number of evaluation samples.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=512,
        help="Maximum sequence length for evaluation (default: 512).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Evaluation batch size (default: 8).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on.",
    )


def run_eval_from_args(args: argparse.Namespace) -> None:
    device = args.device
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    print(f"[SVD-LLM] Loading tokenizer for '{args.model_name_or_path}'.")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[SVD-LLM] Loading model '{args.model_name_or_path}'.")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16 if "cuda" in str(device) else torch.float32,
        token=hf_token,
    )
    evaluate_model(
        model=model,
        tokenizer=tokenizer,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        num_eval_samples=args.num_eval_samples,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        device=device,
    )


def add_run_args(parser: argparse.ArgumentParser) -> None:
    """
    Combined args for a full pipeline run: evaluate original, compress, evaluate compressed.
    """
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
        "--calib-split",
        type=str,
        default="train",
        help="Dataset split to use for calibration (default: train)",
    )
    parser.add_argument(
        "--eval-split",
        type=str,
        default="validation",
        help="Dataset split to use for evaluation (default: validation)",
    )
    parser.add_argument(
        "--num-calib-samples",
        type=int,
        default=256,
        help="Number of calibration sentences to use (default: 256)",
    )
    parser.add_argument(
        "--num-eval-samples",
        type=int,
        default=1024,
        help="Number of evaluation sentences to use (default: 1024)",
    )
    parser.add_argument(
        "--calib-max-seq-len",
        type=int,
        default=256,
        help="Maximum sequence length for calibration (default: 256)",
    )
    parser.add_argument(
        "--eval-max-seq-len",
        type=int,
        default=512,
        help="Maximum sequence length for evaluation (default: 512)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for both calibration and evaluation (default: 8)",
    )
    parser.add_argument(
        "--compression-ratio",
        type=float,
        default=0.4,
        help="Desired per-layer weight compression ratio R_w in [0, 1).",
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
        help="Device to use for both calibration and evaluation",
    )
    parser.add_argument(
        "--max-calib-steps",
        type=int,
        default=None,
        help="Optional max number of calibration batches (if None, use all)",
    )


def run_full_from_args(args: argparse.Namespace) -> None:
    """
    Full pipeline: evaluate original model, compress, evaluate compressed model.
    """
    device = args.device
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    print(f"[SVD-LLM] Loading tokenizer for '{args.model_name}'.")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[SVD-LLM] Loading model '{args.model_name}'.")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if "cuda" in str(device) else torch.float32,
        token=hf_token,
    )

    print("[SVD-LLM] Evaluating original model.")
    evaluate_model(
        model=model,
        tokenizer=tokenizer,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.eval_split,
        num_eval_samples=args.num_eval_samples,
        max_seq_len=args.eval_max_seq_len,
        batch_size=args.batch_size,
        device=device,
    )

    print("[SVD-LLM] Building calibration dataloader.")
    calib_loader = build_text_dataloader(
        tokenizer=tokenizer,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.calib_split,
        num_samples=args.num_calib_samples,
        max_seq_len=args.calib_max_seq_len,
        batch_size=args.batch_size,
    )

    print("[SVD-LLM] Collecting whitening statistics.")
    whitening = collect_whitening_matrices(
        model=model,
        dataloader=calib_loader,
        device=device,
        modules=None,
        max_steps=args.max_calib_steps,
    )

    print("[SVD-LLM] Running SVD-LLM compression.")
    compress_model_svdllm(
        model=model,
        whitening_mats=whitening,
        compression_ratio=args.compression_ratio,
        device=device,
        min_rank=args.min_rank,
    )

    print("[SVD-LLM] Evaluating compressed model.")
    evaluate_model(
        model=model,
        tokenizer=tokenizer,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.eval_split,
        num_eval_samples=args.num_eval_samples,
        max_seq_len=args.eval_max_seq_len,
        batch_size=args.batch_size,
        device=device,
    )

    print(f"[SVD-LLM] Saving compressed model to '{args.output_dir}'.")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


def build_main_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SVD-LLM utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    compress_parser = subparsers.add_parser("compress", help="Compress a HF causal LM")
    add_compress_args(compress_parser)

    eval_parser = subparsers.add_parser("eval", help="Evaluate perplexity of a HF causal LM")
    add_eval_args(eval_parser)

    run_parser = subparsers.add_parser(
        "run",
        help="Evaluate original model, compress, then evaluate compressed model",
    )
    add_run_args(run_parser)

    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_main_argparser()
    args = parser.parse_args(argv)

    if args.command == "compress":
        run_compress_from_args(args)
    elif args.command == "eval":
        run_eval_from_args(args)
    elif args.command == "run":
        run_full_from_args(args)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

