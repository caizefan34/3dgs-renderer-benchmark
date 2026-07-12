"""
Benchmark configuration.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class BenchmarkConfig:
    """Shared benchmark configuration."""
    renderers: List[str] = field(default_factory=lambda: ["speedy_splat", "diff_gaussian", "gsplat"])
    benchmark_frames: int = 200
    warmup_frames: int = 50
    image_width: int = 1920
    image_height: int = 1080
    output_dir: str = "results"
    
    @classmethod
    def create_default(cls):
        return cls()

