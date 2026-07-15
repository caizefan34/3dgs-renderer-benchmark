"""
Comprehensive metrics collection for 3DGS renderer benchmarking.

Captures runtime performance (FPS, percentile-based latency distribution,
frame-time jitter), memory consumption (peak and average VRAM), and
scene loading overhead. Implements the standardized measurement protocol
described in the benchmark documentation.

References:
    Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004).
    Image quality assessment: From error visibility to structural similarity.
    IEEE Transactions on Image Processing, 13(4), 600-612.

    Zhang, R., Isola, P., Efros, A. A., Shechtman, E., & Wang, O. (2018).
    The unreasonable effectiveness of deep features as a perceptual metric.
    In Proceedings of the IEEE Conference on CVPR.
"""
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

from benchmark_suite import BENCHMARK_SUITE_VERSION


@dataclass
class FrameMetrics:
    """Per-frame timing and metadata.

    Attributes:
        frame_index: Zero-based index of the measured frame.
        render_time_ms: Wall-clock render time in milliseconds.
        num_points: Number of Gaussians rendered.
        image_width: Output image width in pixels.
        image_height: Output image height in pixels.
    """
    frame_index: int
    render_time_ms: float
    num_points: int
    image_width: int
    image_height: int


@dataclass
class RendererMetrics:
    """Comprehensive benchmark results for a single renderer.

    Collects runtime statistics (mean, median, percentiles), memory usage,
    scene loading overhead, and image quality metrics (PSNR, SSIM, LPIPS).

    The `compute()` method must be called after populating `frame_times_ms`
    to derive all statistical measures from the raw frame time data.
    """
    renderer_name: str
    mean_fps: float = 0.0
    mean_latency_ms: float = 0.0
    median_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    std_latency_ms: float = 0.0
    coefficient_of_variation: float = 0.0
    stability_score: float = 0.0
    p1_latency_ms: float = 0.0
    p5_latency_ms: float = 0.0
    p10_latency_ms: float = 0.0
    p25_latency_ms: float = 0.0
    p75_latency_ms: float = 0.0
    p90_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    p1_fps: float = 0.0
    p5_fps: float = 0.0
    p95_fps: float = 0.0
    p99_fps: float = 0.0
    jitter_ms: float = 0.0
    num_frames: int = 0
    warmup_frames: int = 0
    image_width: int = 1920
    image_height: int = 1080
    num_gaussians: int = 0
    gpu_name: str = ""
    renderer_implementation: str = ""
    renderer_version: str = "unknown"
    renderer_source_url: str = ""
    renderer_commit_hash: Optional[str] = None
    timing_method: str = "CUDA events"
    benchmark_suite_version: str = BENCHMARK_SUITE_VERSION
    benchmark_commit_hash: Optional[str] = None
    driver_version: Optional[str] = None
    cuda_version: Optional[str] = None
    hardware_metadata: Optional[dict] = None
    mean_wall_latency_ms: float = 0.0
    median_wall_latency_ms: float = 0.0
    wall_fps: float = 0.0

    peak_vram_mb: float = 0.0
    avg_vram_mb: float = 0.0

    scene_load_time_ms: float = 0.0
    scene_parse_time_ms: float = 0.0
    file_size_mb: float = 0.0

    quality_status: str = "not_measured"
    quality_reference: Optional[str] = None
    psnr: Optional[float] = None
    ssim: Optional[float] = None
    lpips: Optional[float] = None
    quality_factor: Optional[float] = None
    effective_fps: Optional[float] = None
    quality_adjustment: Optional[dict] = None
    benchmark_type: str = "synthetic_stress"
    difficulty_score: Optional[float] = None
    difficulty_formula: Optional[str] = None
    difficulty_inputs: Optional[dict] = None
    difficulty_normalization: Optional[dict] = None

    frame_times_ms: List[float] = field(default_factory=list)
    wall_frame_times_ms: List[float] = field(default_factory=list)

    def compute(self):
        """Compute derived statistics from raw frame time measurements.

        Calculates mean, median, standard deviation, jitter (coefficient of
        variation), and percentile-based latency distribution (P1 through P99).
        FPS values are computed as 1000 / latency for each percentile.
        """
        if not self.frame_times_ms:
            return
        t = np.array(self.frame_times_ms)
        n = len(t)

        self.min_latency_ms = float(t.min())
        self.max_latency_ms = float(t.max())
        self.mean_latency_ms = float(t.mean())
        self.median_latency_ms = float(np.median(t))
        self.std_latency_ms = float(t.std())
        self.coefficient_of_variation = float(t.std() / t.mean()) if t.mean() > 0 else 0.0
        self.jitter_ms = self.coefficient_of_variation * 100

        self.p1_latency_ms = float(np.percentile(t, 1))
        self.p5_latency_ms = float(np.percentile(t, 5))
        self.p10_latency_ms = float(np.percentile(t, 10))
        self.p25_latency_ms = float(np.percentile(t, 25))
        self.p75_latency_ms = float(np.percentile(t, 75))
        self.p90_latency_ms = float(np.percentile(t, 90))
        self.p95_latency_ms = float(np.percentile(t, 95))
        self.p99_latency_ms = float(np.percentile(t, 99))
        self.stability_score = (
            min(self.median_latency_ms / self.p99_latency_ms, 1.0)
            if self.p99_latency_ms > 0 else 0.0
        )

        self.mean_fps = round(1000.0 / self.mean_latency_ms, 1) if self.mean_latency_ms > 0 else 0.0
        self.p1_fps = round(1000.0 / self.p99_latency_ms, 1) if self.p99_latency_ms > 0 else 0.0
        self.p5_fps = round(1000.0 / self.p95_latency_ms, 1) if self.p95_latency_ms > 0 else 0.0
        self.p95_fps = round(1000.0 / self.p5_latency_ms, 1) if self.p5_latency_ms > 0 else 0.0
        self.p99_fps = round(1000.0 / self.p1_latency_ms, 1) if self.p1_latency_ms > 0 else 0.0
        if self.wall_frame_times_ms:
            wall = np.array(self.wall_frame_times_ms)
            self.mean_wall_latency_ms = float(wall.mean())
            self.median_wall_latency_ms = float(np.median(wall))
            self.wall_fps = (
                round(1000.0 / self.mean_wall_latency_ms, 1)
                if self.mean_wall_latency_ms > 0 else 0.0
            )

    @property
    def jitter_pct(self) -> float:
        """Return frame-time jitter as a coefficient of variation percentage.

        ``jitter_ms`` is retained as the backing field for compatibility with
        existing result objects, even though its value has always been a
        percentage rather than a duration.
        """
        return self.jitter_ms

    def to_dict(self) -> dict:
        """Serialize metrics to a dictionary for JSON export."""
        d = {
            "renderer": self.renderer_name,
            "renderer_implementation": self.renderer_implementation,
            "renderer_version": self.renderer_version,
            "renderer_source_url": self.renderer_source_url,
            "renderer_commit_hash": self.renderer_commit_hash,
            "timing_method": self.timing_method,
            "benchmark_suite_version": self.benchmark_suite_version,
            "benchmark_commit_hash": self.benchmark_commit_hash,
            "driver_version": self.driver_version,
            "cuda_version": self.cuda_version,
            "hardware_metadata": self.hardware_metadata,
            "mean_wall_latency_ms": self.mean_wall_latency_ms,
            "median_wall_latency_ms": self.median_wall_latency_ms,
            "wall_fps": self.wall_fps,
            "mean_fps": self.mean_fps,
            "mean_latency_ms": self.mean_latency_ms,
            "median_latency_ms": self.median_latency_ms,
            "min_latency_ms": self.min_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "std_latency_ms": self.std_latency_ms,
            "coefficient_of_variation": self.coefficient_of_variation,
            "stability_score": self.stability_score,
            "jitter_pct": round(self.jitter_pct, 2),
        }
        for p in [1, 5, 10, 25, 75, 90, 95, 99]:
            d[f"p{p}_latency_ms"] = getattr(self, f"p{p}_latency_ms")
            d[f"p{p}_fps"] = round(1000.0 / getattr(self, f"p{p}_latency_ms"), 1) if getattr(self, f"p{p}_latency_ms") > 0 else 0.0
        d.update({
            "num_frames": self.num_frames, "image_width": self.image_width, "image_height": self.image_height,
            "num_gaussians": self.num_gaussians, "gpu": self.gpu_name,
            "peak_vram_mb": round(self.peak_vram_mb, 1), "avg_vram_mb": round(self.avg_vram_mb, 1),
            "scene_load_time_ms": round(self.scene_load_time_ms, 2),
            "scene_parse_time_ms": round(self.scene_parse_time_ms, 2),
            "file_size_mb": round(self.file_size_mb, 2),
            "quality_status": self.quality_status,
            "quality_reference": self.quality_reference,
            "psnr": round(self.psnr, 4) if self.psnr is not None else None,
            "ssim": round(self.ssim, 6) if self.ssim is not None else None,
            "lpips": round(self.lpips, 6) if self.lpips is not None else None,
            "quality_factor": round(self.quality_factor, 6) if self.quality_factor is not None else None,
            "effective_fps": round(self.effective_fps, 3) if self.effective_fps is not None else None,
            "quality_adjustment": self.quality_adjustment,
            "benchmark_type": self.benchmark_type,
            "difficulty_score": self.difficulty_score,
            "difficulty_formula": self.difficulty_formula,
            "difficulty_inputs": self.difficulty_inputs,
            "difficulty_normalization": self.difficulty_normalization,
            "frame_times_ms": [round(x, 2) for x in self.frame_times_ms],
            "wall_frame_times_ms": [round(x, 2) for x in self.wall_frame_times_ms],
        })
        return d


class Timer:
    """CUDA event-based timer for precise GPU-accelerated measurements.

    Uses CUDA events (torch.cuda.Event) for accurate GPU-side timing
    that accounts for kernel launch overhead and asynchronous execution.
    """
    def __init__(self, device="cuda"):
        self.device = device
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def start(self):
        """Record a start event on the default CUDA stream."""
        self.start_event.record()

    def stop(self, sync=True) -> float:
        """Record an end event and return elapsed time in milliseconds.

        Args:
            sync: If True, synchronize the CPU with the GPU before querying.

        Returns:
            Elapsed time in milliseconds.
        """
        self.end_event.record()
        if sync:
            torch.cuda.synchronize()
        return self.start_event.elapsed_time(self.end_event)
