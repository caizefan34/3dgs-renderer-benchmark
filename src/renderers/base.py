"""
Renderer adapter base class and interface.
"""
from abc import ABC, abstractmethod
from typing import Optional
import torch


class RendererAdapter(ABC):
    """Abstract base for all renderer adapters."""
    
    name: str = "base"
    
    def __init__(self, device: str = "cuda"):
        self.device = device
    
    @abstractmethod
    def render(self, scene_data: dict, camera: "Camera") -> torch.Tensor:
        """Render one frame. Returns RGB image tensor (H, W, 3) in [0,1]."""
        pass
    
    @abstractmethod
    def prepare_scene(self, scene_data: dict) -> dict:
        """Prepare scene data for this renderer (preprocess, convert formats)."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this renderer is available on the current system."""
        return False
    
    def name(self) -> str:
        return self.name
