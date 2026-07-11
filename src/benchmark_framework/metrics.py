"""
Metrics collection for benchmarking.
"""
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FrameMetrics:
    frame_index: int
    render_time_ms: float
    cuda_time_ms: float
    num_points: int
    image_width: int
    image_height: int


@dataclass
class RendererMetrics:
    renderer_name: str
    mean_fps: float = 0.0
    mean_latency_ms: float = 0.0
    median_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    std_latency_ms: float = 0.0
    num_frames: int = 0
    warmup_frames: int = 0
    image_width: int = 1920
    image_height: int = 1080
    frame_times_ms: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "renderer": self.renderer_name,
            "mean_fps": self.mean_fps,
            "mean_latency_ms": self.mean_latency_ms,
            "median_latency_ms": self.median_latency_ms,
            "min_latency_ms": self.min_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "std_latency_ms": self.std_latency_ms,
            "num_frames": self.num_frames,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "num_gaussians": 400000,
            "gpu": "NVIDIA GeForce RTX 5070 Laptop GPU",
            "frame_times_ms": self.frame_times_ms,
        }


class Timer:
    def __init__(self, device="cuda"):
        self.device = device
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
    
    def start(self):
        self.start_event.record()
    
    def stop(self, sync=True) -> float:
        self.end_event.record()
        if sync:
            torch.cuda.synchronize()
        return self.start_event.elapsed_time(self.end_event)
