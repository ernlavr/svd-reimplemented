from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from .modules import FactoredLinear


@dataclass
class WhiteningStats:
    """
    Stores XX^T and a small diagonal jitter for a single Linear layer.
    """

    xxT: torch.Tensor  # (in_features, in_features)
    num_tokens: int


def _iter_linear_modules(model: nn.Module) -> Iterable[Tuple[str, nn.Linear]]:
    """
    Yield (name, module) for all nn.Linear layers that look like
    standard dense projections, skipping obvious non-compression targets
    like embeddings and lm_head.
    """
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        # Skip very small or 1-dim layers (e.g., layer norms or heads)
        if module.in_features <= 1 or module.out_features <= 1:
            continue
        # Common LM heads to skip
        lowered = name.lower()
        if "lm_head" in lowered or "embed" in lowered:
            continue
        yield name, module


def collect_whitening_matrices(
    model: nn.Module,
    dataloader: DataLoader,
    device: str | torch.device = "cuda",
    modules: Optional[List[Tuple[str, nn.Linear]]] = None,
    max_steps: Optional[int] = None,
) -> Dict[str, WhiteningStats]:
    """
    Collect truncation-aware whitening statistics (XX^T) for each Linear layer.

    This implements Algorithm 2 (Truncation-Aware Data Whitening) from the paper,
    but stores XX^T instead of the full activations.

    Args:
        model: The full HF model (e.g., AutoModelForCausalLM).
        dataloader: Yields batches of tokenized inputs; must be compatible
            with the model's forward (e.g., return dict with 'input_ids', etc.).
        device: Device to run calibration on.
        modules: Optional explicit list of (name, nn.Linear) modules to compress.
            If None, all compressible nn.Linear modules are used.
        max_steps: Optional maximum number of batches to use for calibration.

    Returns:
        Dict mapping module name -> WhiteningStats with accumulated XX^T.
    """
    model.eval()
    model.to(device)

    if modules is None:
        modules = list(_iter_linear_modules(model))

    whitening: Dict[str, WhiteningStats] = {}

    # Forward hooks to accumulate XX^T for inputs of each Linear
    def make_hook(name: str):
        def hook(module: nn.Linear, input, output):
            # input is a tuple; the first element is the tensor of shape (..., in_features)
            x = input[0]
            # Move to float32 for numerical stability
            x = x.detach()
            if x.dtype != torch.float32:
                x = x.float()
            # Reshape to (num_tokens, in_features)
            x_flat = x.reshape(-1, x.shape[-1])
            # XX^T = sum over samples of x^T x
            xxT_local = x_flat.t().matmul(x_flat)

            if name not in whitening:
                whitening[name] = WhiteningStats(
                    xxT=xxT_local.cpu(), num_tokens=x_flat.shape[0]
                )
            else:
                stats = whitening[name]
                stats.xxT += xxT_local.cpu()
                stats.num_tokens += x_flat.shape[0]

        return hook

    handles: List[torch.utils.hooks.RemovableHandle] = []
    for name, module in modules:
        handles.append(module.register_forward_hook(make_hook(name)))

    step = 0
    with torch.no_grad():
        for batch in dataloader:
            step += 1
            if max_steps is not None and step > max_steps:
                break

            if isinstance(batch, dict):
                batch = {k: v.to(device) for k, v in batch.items()}
                model(**batch)
            else:
                # Assume batch itself is a tensor of input_ids
                model(batch.to(device))

    # Remove hooks
    for h in handles:
        h.remove()

    return whitening


def _compute_whitening_matrix(xxT: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Given XX^T, compute S via Cholesky: XX^T ≈ S S^T.
    """
    # Move to the same device we will use for SVD
    device = xxT.device
    xxT = xxT.to(device)

    # Add small jitter on the diagonal for numerical stability
    diag = torch.diag(xxT)
    jitter = eps * diag.mean().clamp(min=1.0)
    xxT_jittered = xxT + torch.eye(xxT.shape[0], device=device, dtype=xxT.dtype) * jitter

    S = torch.linalg.cholesky(xxT_jittered)
    return S


def _compute_rank_from_ratio(
    out_features: int,
    in_features: int,
    compression_ratio: float,
    min_rank: int = 1,
) -> int:
    """
    Compute target rank r from the desired weight compression ratio R_w
    using the relationship:

        R_w = 1 - (d + n) r / (d n)
        => r = (1 - R_w) * d n / (d + n)

    where d = out_features, n = in_features.
    """
    d = float(out_features)
    n = float(in_features)
    R_w = float(compression_ratio)
    R_w = max(0.0, min(0.9999, R_w))

    r_float = (1.0 - R_w) * (d * n) / (d + n)
    r = int(max(min_rank, min(out_features, in_features, round(r_float))))
    return r


def compress_model_svdllm(
    model: nn.Module,
    whitening_mats: Dict[str, WhiteningStats],
    compression_ratio: float,
    device: str | torch.device = "cuda",
    min_rank: int = 1,
) -> None:
    """
    Apply SVD-LLM compression (whitening + SVD + truncation) in-place to the model.

    Args:
        model: The model to compress (modified in-place).
        whitening_mats: Output of collect_whitening_matrices.
        compression_ratio: Desired per-layer weight compression ratio R_w in [0, 1).
        device: Device for SVD and matrix operations.
        min_rank: Minimum allowed rank for any layer.
    """
    model.to(device)
    model.eval()

    modules = dict(_iter_linear_modules(model))

    for name, module in modules.items():
        if name not in whitening_mats:
            # No calibration data seen for this layer; skip
            continue

        stats = whitening_mats[name]
        in_features = module.in_features
        out_features = module.out_features

        # Compute whitening matrix S from XX^T
        xxT = stats.xxT.to(device)
        S = _compute_whitening_matrix(xxT)

        # Original weight and bias
        W = module.weight.data.to(device)  # (out, in)
        bias = module.bias.data.to(device) if module.bias is not None else None

        # Compute W_tilde = W S
        W_tilde = W.matmul(S)

        # SVD on W_tilde
        # W_tilde = U diag(sigma) Vh
        U, Svals, Vh = torch.linalg.svd(W_tilde, full_matrices=False)

        # Determine target rank r
        rank = _compute_rank_from_ratio(
            out_features=out_features,
            in_features=in_features,
            compression_ratio=compression_ratio,
            min_rank=min_rank,
        )

        if rank <= 0 or rank > Svals.numel():
            # Fallback: keep full rank if something goes wrong
            rank = Svals.numel()

        U_r = U[:, :rank]  # (out, r)
        Sigma_r = Svals[:rank]  # (r,)
        Vh_r = Vh[:rank, :]  # (r, in)

        # Construct low-rank factors following the paper:
        #   W_u' = U * sqrt(Σ_trunc)
        #   W_v' = sqrt(Σ_trunc) * V^T * S^{-1}
        sqrt_sigma = torch.sqrt(Sigma_r).to(device)

        # W_u: (out, r)
        W_u = U_r * sqrt_sigma.unsqueeze(0)

        # Compute S^{-1}. Since S is upper triangular, use solve instead of explicit inverse.
        I = torch.eye(S.shape[0], device=device, dtype=S.dtype)
        S_inv = torch.linalg.solve(S, I)

        # temp = Vh_r @ S^{-1} -> (r, in)
        temp = Vh_r.matmul(S_inv)
        # W_v: (r, in)
        W_v = sqrt_sigma.unsqueeze(1) * temp

        # Replace the module with FactoredLinear in-place on its parent
        factored = FactoredLinear.from_weight(W_u=W_u, W_v=W_v, bias=bias)

        # Find parent module to set attribute
        parent = model
        *parent_path, last_name = name.split(".")
        for p in parent_path:
            parent = getattr(parent, p)

        setattr(parent, last_name, factored.to(device))

