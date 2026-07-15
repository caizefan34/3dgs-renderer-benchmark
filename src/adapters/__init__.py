"""Strict public adapter contract and benchmark helpers."""

from .base import RendererAdapter
from .quality import QualityGateResult, QualityThresholds, evaluate_quality_gate

__all__ = [
    "QualityGateResult",
    "QualityThresholds",
    "RendererAdapter",
    "evaluate_quality_gate",
]
