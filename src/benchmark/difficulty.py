"""Versioned Scene Difficulty Score for 3DGS workloads.

The initial formula combines four renderer-independent measurements.  Raw
measurements and normalization scales are retained so a future formula can be
recomputed without changing or invalidating stored benchmark results.
"""
from dataclasses import asdict, dataclass
from math import prod
from typing import Dict


@dataclass(frozen=True)
class DifficultyInputs:
    """Camera-trajectory aggregates used by the difficulty formula."""

    visible_gaussian_count: int
    overlap_ratio: float
    average_tile_density: float
    depth_complexity: float

    def validate(self) -> None:
        if any(value < 0 for value in asdict(self).values()):
            raise ValueError("Scene difficulty inputs must be non-negative")


@dataclass(frozen=True)
class DifficultyConfig:
    """Saturation scales for the normalized 0--10 score."""

    visible_gaussian_scale: float = 1_000_000.0
    overlap_ratio_scale: float = 8.0
    average_tile_density_scale: float = 128.0
    depth_complexity_scale: float = 16.0
    formula_id: str = "geometric_mean_v1"

    def validate(self) -> None:
        scales = (
            self.visible_gaussian_scale,
            self.overlap_ratio_scale,
            self.average_tile_density_scale,
            self.depth_complexity_scale,
        )
        if any(value <= 0 for value in scales):
            raise ValueError("Scene difficulty normalization scales must be positive")


@dataclass(frozen=True)
class DifficultyResult:
    schema_version: int
    formula_id: str
    score: float
    inputs: DifficultyInputs
    normalized_factors: Dict[str, float]
    normalization: Dict[str, float]

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "formula_id": self.formula_id,
            "score": self.score,
            "inputs": asdict(self.inputs),
            "normalized_factors": self.normalized_factors,
            "normalization": self.normalization,
        }


def calculate_difficulty(
    inputs: DifficultyInputs,
    config: DifficultyConfig = DifficultyConfig(),
) -> DifficultyResult:
    """Return a normalized 0--10 geometric-mean workload score."""
    inputs.validate()
    config.validate()
    raw = asdict(inputs)
    normalization = {
        "visible_gaussian_count": config.visible_gaussian_scale,
        "overlap_ratio": config.overlap_ratio_scale,
        "average_tile_density": config.average_tile_density_scale,
        "depth_complexity": config.depth_complexity_scale,
    }
    factors = {
        name: min(float(value) / normalization[name], 1.0)
        for name, value in raw.items()
    }
    score = 10.0 * prod(factors.values()) ** (1.0 / len(factors))
    return DifficultyResult(
        schema_version=1,
        formula_id=config.formula_id,
        score=round(score, 4),
        inputs=inputs,
        normalized_factors={key: round(value, 6) for key, value in factors.items()},
        normalization=normalization,
    )
