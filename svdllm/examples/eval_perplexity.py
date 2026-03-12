import argparse
import math

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
)


def build_eval_dataloader(
    tokenizer,
    dataset_name: str,
    dataset_config: str,
    split: str,
    num_samples: int | None,
    max_seq_len: int,
    batch_size: int,
):
    ds = load_dataset(dataset_name, dataset_config, split=split)
    if num_samples is not None and num_samples > 0:
        ds = ds.select(range(min(num_samples, len(ds))))

    text_column = "text"
    if text_column not in ds.column_names:
        for cand in ["content", "sentence", "document"]:
            if cand in ds.column_names:
                text_column = cand
                break

    def tokenize_fn(examples):
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
    parser = argparse.ArgumentParser(description="Evaluate perplexity of a HF causal LM")
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
    return parser.parse_args()


def main():
    args = parse_args()

    device = args.device

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16 if "cuda" in device else torch.float32,
    )
    model.to(device)
    model.eval()

    dataloader = build_eval_dataloader(
        tokenizer=tokenizer,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        num_samples=args.num_eval_samples,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
    )

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            # Shift labels to compute standard causal LM loss
            labels = batch["input_ids"].clone()
            batch["labels"] = labels
            outputs = model(**batch)
            # outputs.loss is mean over tokens; multiply by number of tokens
            loss = outputs.loss
            # Count non-padding tokens
            non_pad = (labels != tokenizer.pad_token_id).sum()
            total_loss += loss.item() * non_pad.item()
            total_tokens += non_pad.item()

    avg_nll = total_loss / max(1, total_tokens)
    ppl = math.exp(avg_nll)

    print(f"Average NLL: {avg_nll:.4f}")
    print(f"Perplexity: {ppl:.4f}")


if __name__ == "__main__":
    main()

