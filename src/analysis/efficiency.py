"""Experimental quality-adjusted renderer efficiency metric."""
from dataclasses import asdict, dataclass
import math
from typing import Mapping, Optional


QUALITY_KEYS = ("psnr", "ssim", "lpips")


def calculate_efficiency_score(fps: float, psnr: float, target_psnr: float = 32.0) -> float:
    """Return quality-adjusted FPS using PSNR's linear MSE ratio.

    A renderer at ``target_psnr`` keeps its raw FPS. Each 10 dB below the
    target reduces the score by 10x; quality above the target increases it.
    """
    if fps < 0:
        raise ValueError("FPS must be non-negative")
    return float(fps) * 10.0 ** ((float(psnr) - float(target_psnr)) / 10.0)


@dataclass(frozen=True)
class QualityAdjustmentConfig:
    """Penalty coefficients, expressed per native metric unit."""

    psnr_drop_weight: float = 0.25
    ssim_drop_weight: float = 25.0
    lpips_increase_weight: float = 10.0
    minimum_quality_factor: float = 0.0
    formula_id: str = "exponential_penalty_v1"

    def validate(self) -> None:
        weights = (
            self.psnr_drop_weight,
            self.ssim_drop_weight,
            self.lpips_increase_weight,
        )
        if any(value < 0 for value in weights):
            raise ValueError("Quality penalty weights must be non-negative")
        if not 0.0 <= self.minimum_quality_factor <= 1.0:
            raise ValueError("minimum_quality_factor must be between 0 and 1")


@dataclass(frozen=True)
class QualityAdjustedEfficiency:
    formula_id: str
    fps: float
    quality_factor: Optional[float]
    effective_fps: Optional[float]
    penalties: Optional[dict]
    coefficients: dict
    status: str

    def to_dict(self) -> dict:
        return asdict(self)


def calculate_quality_adjusted_efficiency(
    fps: float,
    quality: Mapping[str, Optional[float]],
    reference: Mapping[str, Optional[float]],
    config: QualityAdjustmentConfig = QualityAdjustmentConfig(),
) -> QualityAdjustedEfficiency:
    """Apply one-sided GT-quality penalties to raw FPS.

    Improvements over the reference do not increase the factor above one.
    Missing quality remains unscored instead of being interpreted as lossless.
    """
    if fps < 0:
        raise ValueError("FPS must be non-negative")
    config.validate()
    coefficients = asdict(config)
    coefficients.pop("formula_id")
    if any(quality.get(key) is None or reference.get(key) is None for key in QUALITY_KEYS):
        return QualityAdjustedEfficiency(
            config.formula_id, float(fps), None, None, None, coefficients,
            "not_computed_missing_quality",
        )

    penalties = {
        "psnr_drop": max(0.0, float(reference["psnr"]) - float(quality["psnr"])),
        "ssim_drop": max(0.0, float(reference["ssim"]) - float(quality["ssim"])),
        "lpips_increase": max(0.0, float(quality["lpips"]) - float(reference["lpips"])),
    }
    weighted_penalty = (
        penalties["psnr_drop"] * config.psnr_drop_weight
        + penalties["ssim_drop"] * config.ssim_drop_weight
        + penalties["lpips_increase"] * config.lpips_increase_weight
    )
    quality_factor = max(config.minimum_quality_factor, math.exp(-weighted_penalty))
    return QualityAdjustedEfficiency(
        formula_id=config.formula_id,
        fps=float(fps),
        quality_factor=quality_factor,
        effective_fps=float(fps) * quality_factor,
        penalties={**penalties, "weighted_total": weighted_penalty},
        coefficients=coefficients,
        status="experimental",
    )
