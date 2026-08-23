"""
3DGS-standard loss functions.

Implements the combined L1 + D-SSIM loss used in the original 3DGS paper.
D-SSIM uses a simplified Gaussian-blur SSIM matching graphdeco-inria's implementation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def d_ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> torch.Tensor:
    """Differentiable D-SSIM loss (1 - SSIM).

    Uses zero-padded Gaussian-weighted SSIM matching graphdeco-inria's
    loss_utils.ssim implementation. Inputs must be RGB tensors in [0, 1].
    """
    if pred.ndim == 3:
        pred = pred.unsqueeze(0).permute(0, 3, 1, 2)  # [1, 3, H, W]
        target = target.unsqueeze(0).permute(0, 3, 1, 2)
    elif pred.ndim == 4:
        pred = pred.permute(0, 3, 1, 2)  # [B, 3, H, W]
        target = target.permute(0, 3, 1, 2)

    # Build Gaussian kernel
    coords = torch.arange(window_size, device=pred.device, dtype=pred.dtype) - window_size // 2
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel = kernel_1d[:, None] * kernel_1d[None, :]
    kernel = kernel.expand(pred.shape[1], 1, window_size, window_size).contiguous()

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    def blur(x):
        return F.conv2d(x, kernel, padding=window_size // 2, groups=pred.shape[1])

    mu_pred = blur(pred)
    mu_target = blur(target)
    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = blur(pred ** 2) - mu_pred_sq
    sigma_target_sq = blur(target ** 2) - mu_target_sq
    sigma_pred_target = blur(pred * target) - mu_pred_target

    ssim_map = (
        (2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)
    ) / (
        (mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2)
    )

    return 1.0 - ssim_map.mean()


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lambda_dssim: float = 0.2,
) -> "Dict[str, torch.Tensor]":
    """Compute L1 + λ * D-SSIM as in the 3DGS paper.

    Returns dict with 'loss' (the combined scalar), 'l1', and 'd_ssim' components.
    """
    l1 = F.l1_loss(pred, target)
    dsim = d_ssim_loss(pred, target)
    total = (1.0 - lambda_dssim) * l1 + lambda_dssim * dsim
    return {"loss": total, "l1": l1, "d_ssim": dsim}

