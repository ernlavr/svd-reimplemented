## SVD-LLM (Truncation-aware SVD Compression)

This repo provides a **model-agnostic PyTorch implementation** of the SVD-LLM compression method described in  
“SVD-LLM: Truncation-aware Singular Value Decomposition for Large Language Model Compression” \([arXiv:2403.07378](https://arxiv.org/pdf/2403.07378)\).

It is designed to work directly with **Hugging Face `transformers`** models, including:

- **LLaMA 1 / LLaMA 2**
- **LLaMA 3**
- **Qwen 2 / Qwen 2.5 / Qwen 3** (any causal LM implemented with `nn.Linear` layers)

There is a unified CLI entrypoint in `svdllm/main.py` with two subcommands:

- `compress`: run SVD‑LLM compression
- `eval`: evaluate perplexity

### Key features

- **Truncation-aware data whitening** per the paper:
  - Accumulates \(XX^\top\) from calibration activations.
  - Computes whitening matrix via Cholesky decomposition.
- **Rank selection by global compression ratio**:
  - Uses the closed-form \(r = (1 - R_w)\frac{dn}{d + n}\) per layer, where:
    - \(R_w\): desired weight compression ratio (e.g. 0.2 for 20% compression),
    - \(d\): out features, \(n\): in features.
- **Low-rank factorization compatible with any `nn.Linear`**:
  - Replaces `nn.Linear` with a `FactoredLinear` that computes \(W' = W_u W_v\) efficiently as:
    - \(M = X W_v^\top\), \(Y = M W_u^\top\).
- Designed to be **model family agnostic**:
  - By default it compresses all `nn.Linear` layers except embeddings and `lm_head`.

### Installation

```bash
pip install -r requirements.txt
```

Optionally, create a `.env` file in the project root with your Hugging Face token:

```bash
echo "HF_TOKEN=hf_xxx_your_token_here" > .env
```

The CLI will automatically load this and pass it to `transformers` when loading models.

### Basic usage (Hugging Face causal LM)

Example: compress a LLaMA‑3 model with 40% weight compression using a small calibration subset of WikiText‑2:

```bash
python -m svdllm.main compress \
  --model-name meta-llama/Meta-Llama-3-8B \
  --dataset-name wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --split train \
  --num-calib-samples 256 \
  --max-seq-len 256 \
  --compression-ratio 0.4 \
  --output-dir ./llama3-8b-svdllm-40
```

To run on **Qwen 3** or **LLaMA 1**, just change `--model-name` to the appropriate Hugging Face ID. The implementation only assumes a standard `AutoModelForCausalLM` with `nn.Linear` layers.

### Perplexity evaluation

You can evaluate **original vs compressed** models on any HF text dataset using:

```bash
python -m svdllm.main eval \
  --model-name-or-path meta-llama/Meta-Llama-3-8B \
  --dataset-name wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --split validation \
  --num-eval-samples 1024 \
  --max-seq-len 512 \
  --batch-size 8
```

Then, for the compressed checkpoint:

```bash
python -m svdllm.main eval \
  --model-name-or-path ./llama3-8b-svdllm-40 \
  --dataset-name wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --split validation \
  --num-eval-samples 1024 \
  --max-seq-len 512 \
  --batch-size 8
```

The script reports the **average negative log-likelihood per token** and the corresponding **perplexity**.

### High-level API

The core API lives in `svdllm/compression.py`:

- **Collect whitening matrices from calibration data**

```python
from svdllm.compression import collect_whitening_matrices

whitening = collect_whitening_matrices(
    model,
    dataloader,
    device="cuda",
    modules=None,          # or a filtered list of modules to compress
    max_steps=100          # optional cap on calibration steps
)
```

- **Compress the model with SVD‑LLM whitening + SVD**

```python
from svdllm.compression import compress_model_svdllm

compress_model_svdllm(
    model,
    whitening_mats=whitening,
    compression_ratio=0.4,  # 40% weight compression
    device="cuda",
    min_rank=4
)
```

This performs:

1. Truncation-aware data whitening with Cholesky decomposition.
2. SVD on \(W S\) for each compressed layer.
3. Rank selection via the global `compression_ratio`.
4. Replacement of the original `nn.Linear` with a `FactoredLinear` using \(W_u\) and \(W_v\).

### Notes

- The implementation currently covers the **whitening + SVD compression (SVD‑LLM (W))** part of the paper.  
  A sequential LoRA-style parameter update API can be added on top of the factored weights if you want to match the full SVD‑LLM pipeline.
- All math and design choices follow the pseudocode and derivations in the SVD‑LLM paper \([arXiv:2403.07378](https://arxiv.org/pdf/2403.07378)\).

