"""
Benchmark Framework for 3D Gaussian Splatting Renderers.

This package provides a unified, reproducible benchmarking infrastructure for
comparing CUDA rasterization backends used in 3D Gaussian Splatting (3DGS)
[Kerbl et al., ACM Trans. Graph., 2023]. The framework enforces a standardized
measurement protocol to ensure fair comparison across renderers.

References:
    Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023).
    3D Gaussian Splatting for Real-Time Radiance Field Rendering.
    ACM Transactions on Graphics, 42(4).
"""
from .scene import load_ply, compute_cov3d_from_scales_rot
from .cameras import (
    generate_cameras,
    load_cameras_from_json,
    validate_cameras_facing_point,
    Camera,
)
from .metrics import Timer, RendererMetrics, FrameMetrics
from .results import ResultsManager
from .config import BenchmarkConfig

__all__ = [
    "load_ply", "compute_cov3d_from_scales_rot",
    "generate_cameras", "load_cameras_from_json", "validate_cameras_facing_point", "Camera",
    "Timer", "RendererMetrics", "FrameMetrics",
    "ResultsManager", "BenchmarkConfig",
]
