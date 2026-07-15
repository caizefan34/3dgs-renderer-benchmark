"""Quality-gate policy independent of metric implementations."""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class QualityThresholds:
    """Inclusive acceptance thresholds for mean image-quality metrics."""

    min_psnr_db: float
    min_ssim: float
    max_lpips: float


@dataclass(frozen=True)
class QualityGateResult:
    """Quality decision plus stable, machine-readable rejection reasons."""

    passed: bool
    failures: Tuple[str, ...]


def evaluate_quality_gate(
    psnr_db: float,
    ssim: float,
    lpips: float,
    thresholds: QualityThresholds,
) -> QualityGateResult:
    """Apply all three quality thresholds without short-circuiting."""
    failures = []
    if psnr_db < thresholds.min_psnr_db:
        failures.append("psnr_below_minimum")
    if ssim < thresholds.min_ssim:
        failures.append("ssim_below_minimum")
    if lpips > thresholds.max_lpips:
        failures.append("lpips_above_maximum")
    return QualityGateResult(passed=not failures, failures=tuple(failures))
