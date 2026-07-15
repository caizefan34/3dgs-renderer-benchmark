"""Strict renderer interface shared by benchmark backends."""

from abc import ABC, abstractmethod
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Mapping, Optional

import torch


class RendererAdapter(ABC):
    """Required interface for quality-gated renderer implementations.

    ``camera_params`` must contain positive integer ``height`` and ``width``
    values. A conforming render returns float32 HWC tensors named ``rgb``,
    ``depth``, and ``alpha``. RGB has three channels; depth and alpha each
    have one channel. Use :meth:`render_checked` at integration boundaries to
    enforce this runtime contract.
    """

    def __init__(self, device: str = "cuda") -> None:
        self.device = device

    @abstractmethod
    def load_checkpoint(self, path: Path) -> Any:
        """Load a renderer checkpoint and return backend-specific state."""

    @abstractmethod
    def render(self, camera_params: Dict) -> Dict[str, torch.Tensor]:
        """Render one frame at the resolution in ``camera_params``."""

    @abstractmethod
    def get_memory_usage_mb(self) -> float:
        """Return memory currently allocated by this process in MiB."""

    @abstractmethod
    def get_device_properties(self) -> Dict:
        """Return serializable device and runtime properties."""

    @staticmethod
    def _resolution(camera_params: Mapping) -> tuple:
        if not isinstance(camera_params, Mapping):
            raise TypeError("camera_params must be a mapping")
        height = camera_params.get("height")
        width = camera_params.get("width")
        if (
            not isinstance(height, int)
            or isinstance(height, bool)
            or height <= 0
            or not isinstance(width, int)
            or isinstance(width, bool)
            or width <= 0
        ):
            raise ValueError("camera_params height and width must be positive integers")
        return height, width

    @classmethod
    def validate_render_output(
        cls,
        output: Mapping[str, torch.Tensor],
        camera_params: Mapping,
    ) -> None:
        """Raise when a renderer output violates the protocol tensor contract."""
        height, width = cls._resolution(camera_params)
        if not isinstance(output, Mapping):
            raise TypeError("render output must be a mapping")

        expected_shapes = {
            "rgb": (height, width, 3),
            "depth": (height, width, 1),
            "alpha": (height, width, 1),
        }
        missing = expected_shapes.keys() - output.keys()
        if missing:
            raise ValueError(f"render output is missing keys: {sorted(missing)}")

        devices = set()
        for name, expected_shape in expected_shapes.items():
            value = output[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if tuple(value.shape) != expected_shape:
                raise ValueError(
                    f"{name} has shape {tuple(value.shape)}; expected {expected_shape}"
                )
            if value.dtype != torch.float32:
                raise TypeError(f"{name} must have dtype torch.float32, got {value.dtype}")
            if not torch.isfinite(value).all().item():
                raise ValueError(f"{name} contains non-finite values")
            devices.add(value.device)
        if len(devices) != 1:
            raise ValueError("rgb, depth, and alpha must be on the same device")

    def render_checked(self, camera_params: Dict) -> Dict[str, torch.Tensor]:
        """Render one frame and validate names, resolution, dtype, and finiteness."""
        self._resolution(camera_params)
        output = self.render(camera_params)
        self.validate_render_output(output, camera_params)
        return output

    def benchmark(
        self,
        camera_params: Dict,
        warmup_iterations: int = 10,
        repeats: int = 100,
        timer: Optional[Callable[[], float]] = None,
    ) -> list:
        """Return one ordered latency sample per measured repeat in milliseconds.

        Warmup renders are excluded. The injectable monotonic timer keeps the
        protocol unit-testable; CUDA work is synchronized around measured calls.
        """
        if warmup_iterations < 0:
            raise ValueError("warmup_iterations must be non-negative")
        if repeats <= 0:
            raise ValueError("repeats must be positive")
        self._resolution(camera_params)
        clock = timer or perf_counter

        with torch.inference_mode():
            for _ in range(warmup_iterations):
                self.render_checked(camera_params)
            self._synchronize_if_cuda()

            samples = []
            for _ in range(repeats):
                self._synchronize_if_cuda()
                start = clock()
                output = self.render(camera_params)
                self._synchronize_if_cuda()
                elapsed_ms = (clock() - start) * 1000.0
                # Contract checks are deliberately outside the timing boundary.
                self.validate_render_output(output, camera_params)
                samples.append(round(elapsed_ms, 6))
        return samples

    def _synchronize_if_cuda(self) -> None:
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
