"""Speed-quality analysis for 3DGS renderer results."""

from .efficiency import (
    QualityAdjustmentConfig,
    QualityAdjustedEfficiency,
    calculate_quality_adjusted_efficiency,
)
from .pareto import pareto_analysis
from .recommendations import build_recommendations

__all__ = [
    "QualityAdjustmentConfig",
    "QualityAdjustedEfficiency",
    "calculate_quality_adjusted_efficiency",
    "pareto_analysis",
    "build_recommendations",
]
