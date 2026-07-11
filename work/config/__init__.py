"""
Benchmark configuration.
"""
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class BenchmarkConfig:
    """Global benchmark configuration."""
    scene_path: str = "scene.ply"  # relative to data/ dir
    
    image_width: int = 1920
    image_height: int = 1080
    num_cameras: int = 50
    
    renderers: List[str] = field(default_factory=lambda: [
        "gsplat",
        "diff_gaussian",
    ])
    
    warmup_frames: int = 5
    benchmark_frames: int = 30
    
    output_dir: str = "outputs"
    log_dir: str = "outputs/logs"
    seed: int = 42
    
    @classmethod
    def create_default(cls):
        return cls()
