"""
Renderer adapter base class and abstract interface.

Defines the contract that all renderer adapters must implement to participate
in the benchmark. Each adapter wraps a specific CUDA rasterization backend
and provides a uniform interface for scene preparation and rendering.

References:
    Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023).
    3D Gaussian Splatting for Real-Time Radiance Field Rendering.
    ACM Transactions on Graphics, 42(4).
"""
from abc import abstractmethod
from pathlib import Path
from typing import Optional
import torch
from importlib import metadata
import importlib
import json
import subprocess

from adapters.base import RendererAdapter as StrictRendererAdapter


class RendererAdapter(StrictRendererAdapter):
    """Abstract base class for all renderer adapters.

    Each concrete adapter wraps a specific 3DGS rasterization backend
    (e.g., diff-gaussian-rasterization, speedy-splat) and provides a
    standardized interface for scene preparation and frame rendering.

    Subclasses must implement:
        - render(): Render a single frame from scene data and camera.
        - prepare_scene(): Preprocess scene data for the renderer.
        - is_available(): Check runtime availability of the backend.
    """

    name: str = "base"
    package_name: Optional[str] = None
    module_name: Optional[str] = None
    implementation: str = ""
    source_url: str = ""

    def __init__(self, device: str = "cuda"):
        super().__init__(device=device)
        self._checkpoint = None

    def load_checkpoint(self, path: Path):
        """Load and prepare the legacy PLY checkpoint format."""
        from benchmark_framework import load_ply

        scene_data = load_ply(str(path), device=self.device)
        self._checkpoint = self.prepare_scene(scene_data)
        return self._checkpoint

    def get_memory_usage_mb(self) -> float:
        if not str(self.device).startswith("cuda") or not torch.cuda.is_available():
            return 0.0
        return float(torch.cuda.memory_allocated(self.device) / (1024 * 1024))

    def get_device_properties(self) -> dict:
        if not str(self.device).startswith("cuda") or not torch.cuda.is_available():
            return {"device": str(self.device), "cuda_available": False}
        properties = torch.cuda.get_device_properties(self.device)
        return {
            "device": str(self.device),
            "cuda_available": True,
            "name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_mb": float(properties.total_memory / (1024 * 1024)),
            "multi_processor_count": properties.multi_processor_count,
        }

    @abstractmethod
    def render(self, scene_data: dict, camera: "Camera") -> torch.Tensor:
        """Render one frame.

        Args:
            scene_data: Dictionary of GPU tensors from load_ply().
            camera: Camera object specifying the viewpoint.

        Returns:
            RGB image tensor of shape (H, W, 3) with values in [0, 1].
        """
        pass

    @abstractmethod
    def prepare_scene(self, scene_data: dict) -> dict:
        """Prepare scene data for this renderer.

        May preprocess, convert formats, or pre-compute data structures
        specific to the renderer backend.

        Args:
            scene_data: Raw scene data from load_ply().

        Returns:
            Processed scene data dictionary.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this renderer backend is available on the current system.

        Returns:
            True if the renderer can be instantiated and used.
        """
        return False

    def name(self) -> str:
        return self.name

    def metadata(self) -> dict:
        version = "unknown"
        commit_hash = None
        module = None
        if self.module_name:
            try:
                module = importlib.import_module(self.module_name)
                version = getattr(module, "__version__", version)
            except (ImportError, OSError):
                pass
        if self.package_name:
            if version == "unknown":
                try:
                    version = metadata.version(self.package_name)
                except metadata.PackageNotFoundError:
                    pass
            try:
                direct_url = metadata.distribution(self.package_name).read_text("direct_url.json")
                if direct_url:
                    commit_hash = json.loads(direct_url).get("vcs_info", {}).get("commit_id")
            except (metadata.PackageNotFoundError, json.JSONDecodeError):
                pass
        if commit_hash is None and module is not None and getattr(module, "__file__", None):
            for parent in Path(module.__file__).resolve().parents:
                if (parent / ".git").exists():
                    try:
                        commit_hash = subprocess.check_output(
                            ["git", "-C", str(parent), "rev-parse", "HEAD"],
                            text=True,
                            stderr=subprocess.DEVNULL,
                        ).strip()
                    except (OSError, subprocess.SubprocessError):
                        pass
                    break
        return {
            "implementation": self.implementation or self.name,
            "version": version,
            "source_url": self.source_url,
            "commit_hash": commit_hash,
        }
