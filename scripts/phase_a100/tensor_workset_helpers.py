"""Compact exact GPU workset overlap helpers for gsplat 1.5.3 audits."""
from __future__ import annotations

import torch


def jaccard_sorted(a: torch.Tensor, b: torch.Tensor) -> float:
    """Exact Jaccard for sorted unique integer tensors, without Python sets."""
    if a.numel() == 0 and b.numel() == 0:
        return 1.0
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    # Both A/B are unique by construction. Membership keys are one per
    # Gaussian-tile pair; sort_unique is retained for defensive correctness.
    a = torch.unique_consecutive(a)
    b = torch.unique_consecutive(b)
    pos = torch.searchsorted(b, a)
    valid = pos < b.numel()
    matches = torch.zeros_like(valid)
    matches[valid] = b[pos[valid]] == a[valid]
    intersection = int(matches.sum().item())
    union = a.numel() + b.numel() - intersection
    return intersection / union if union else 1.0


def compact_for_history(values: torch.Tensor) -> torch.Tensor:
    """Preserve a compact CPU copy only where long-lived history is required."""
    return values.detach().cpu()
