import math
from typing import Optional

import torch
from torch import nn


class FactoredLinear(nn.Module):
    """
    Linear layer represented as W = W_u @ W_v with no bias.

    Original: y = x @ W^T + b, W in R[out, in]
    Factored: W_u in R[out, r], W_v in R[r, in]
              y = (x @ W_v.T) @ W_u.T + b
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: Optional[torch.Tensor] = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        # W_u: (out, r), W_v: (r, in)
        self.weight_u = nn.Parameter(
            torch.empty(out_features, rank, **factory_kwargs)
        )
        self.weight_v = nn.Parameter(
            torch.empty(rank, in_features, **factory_kwargs)
        )

        if bias is None:
            self.bias = None
        else:
            # Register bias as a parameter; clone to avoid sharing storage
            self.bias = nn.Parameter(bias.detach().clone())

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Kaiming-like init on the composed weight
        # Roughly match variance of standard nn.Linear
        bound = 1 / math.sqrt(self.in_features) if self.in_features > 0 else 0.0
        with torch.no_grad():
            self.weight_u.uniform_(-bound, bound)
            self.weight_v.uniform_(-bound, bound)
            if self.bias is not None:
                self.bias.uniform_(-bound, bound)

    @classmethod
    def from_weight(
        cls,
        weight_u: torch.Tensor,
        weight_v: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> "FactoredLinear":
        """
        Construct a FactoredLinear directly from low-rank factors.
        """
        out_features, rank = weight_u.shape
        rank2, in_features = weight_v.shape
        assert (
            rank == rank2
        ), f"Rank mismatch between W_u ({rank}) and W_v ({rank2})"

        module = cls(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            bias=bias,
            device=weight_u.device,
            dtype=weight_u.dtype,
        )

        with torch.no_grad():
            module.weight_u.copy_(weight_u)
            module.weight_v.copy_(weight_v)

        return module

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input: (..., in_features)
        # Step 1: M = X @ W_v^T -> (..., rank)
        m = torch.matmul(input, self.weight_v.t())
        # Step 2: Y = M @ W_u^T -> (..., out_features)
        output = torch.matmul(m, self.weight_u.t())
        if self.bias is not None:
            output = output + self.bias
        return output

