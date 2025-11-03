from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor


def hierarchical_pool(tokens: Tensor, factor: int = 2, method: Literal["mean", "max"] = "mean") -> Tensor:
    """Down-sample variable-length tokens using fixed pooling factor."""

    if factor <= 1:
        return tokens
    seq_len = tokens.size(-2)
    target = (seq_len + factor - 1) // factor * factor
    if target != seq_len:
        pad = torch.zeros(tokens.size(0), target - seq_len, tokens.size(-1), dtype=tokens.dtype, device=tokens.device)
        tokens = torch.cat([tokens, pad], dim=-2)
    tokens = tokens.view(tokens.size(0), -1, factor, tokens.size(-1))
    if method == "max":
        pooled, _ = torch.max(tokens, dim=2)
    else:
        pooled = torch.mean(tokens, dim=2)
    return pooled


__all__ = ["hierarchical_pool"]
