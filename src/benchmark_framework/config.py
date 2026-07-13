"""
Benchmark configuration.

Defines the standardized measurement protocol for the 3DGS renderer benchmark,
including warmup frames, measurement frames, repeats, and synchronization
parameters. All renderers are evaluated under identical configuration to
ensure reproducibility.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class BenchmarkConfig:
    """Shared benchmark configuration with explicit methodology documentation.

    Attributes:
        renderers: List of renderer identifiers to benchmark.
        benchmark_frames: Number of frames measured per repeat.
        warmup_frames: Number of frames executed before measurement (excluded from results).
        repeats: Number of independent measurement runs.
        clock_lock: Whether to lock GPU clock frequency for stable measurements.
        image_width: Output image width in pixels.
        image_height: Output image height in pixels.
        output_dir: Directory for benchmark result exports.
    """
    renderers: List[str] = field(default_factory=lambda: ["speedy_splat", "diff_gaussian", "gsplat"])
    benchmark_frames: int = 200
    warmup_frames: int = 50
    repeats: int = 5
    clock_lock: bool = True
    image_width: int = 1920
    image_height: int = 1080
    output_dir: str = "results"

    @classmethod
    def create_default(cls):
        """Create a BenchmarkConfig with default parameters."""
        return cls()

    @property
    def methodology(self) -> dict:
        """Return the measurement methodology as a dictionary for reporting."""
        return {
            "warmup": self.warmup_frames,
            "measured": self.benchmark_frames,
            "repeats": self.repeats,
            "clock_lock": self.clock_lock,
            "cuda_sync": "after each frame (before measurement)",
            "timing_method": "time.perf_counter() + torch.cuda.synchronize()",
        }