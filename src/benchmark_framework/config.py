"""
Benchmark configuration.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class BenchmarkConfig:
    """Shared benchmark configuration with explicit methodology fields."""
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
        return cls()
    
    @property
    def methodology(self) -> dict:
        return {
            "warmup": self.warmup_frames,
            "measured": self.benchmark_frames,
            "repeats": self.repeats,
            "clock_lock": self.clock_lock,
            "cuda_sync": "after each frame (before measure)",
            "timing_method": "time.perf_counter() + torch.cuda.synchronize()",
        }

