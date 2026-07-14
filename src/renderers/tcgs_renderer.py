"""Adapter for DeepLink-org/3DGSTensorCore's Speedy-Splat integration."""
import inspect

import torch

from .base import RendererAdapter
from benchmark_framework import Camera


class TCGSRenderer(RendererAdapter):
    """Inference adapter for the isolated TC-GS rasterizer package."""

    name = "tcgs"
    package_name = "diff-gaussian-rasterization"
    module_name = "diff_gaussian_rasterization"
    implementation = "DeepLink-org/3DGSTensorCore (Speedy-Splat integration)"
    source_url = "https://github.com/DeepLink-org/3DGSTensorCore"

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self._available = None
        self._rasterizers = {}
        self._bg = None
        self.last_wrapper_seconds = None

    def is_available(self) -> bool:
        if self._available is None:
            try:
                from diff_gaussian_rasterization import GaussianRasterizer

                self._available = "scores" in inspect.signature(
                    GaussianRasterizer.forward
                ).parameters
            except (ImportError, OSError, ValueError):
                self._available = False
        return self._available

    def prepare_scene(self, scene_data: dict) -> dict:
        means3d = scene_data["xyz"].contiguous()
        return {
            **scene_data,
            "xyz": means3d,
            "opacities_activated": torch.sigmoid(scene_data["opacity"]).contiguous(),
            "scales_activated": torch.exp(scene_data["scales"]).contiguous(),
            "rotations_normalized": torch.nn.functional.normalize(
                scene_data["rotations"], dim=-1
            ).contiguous(),
            "shs": scene_data["shs"].contiguous(),
            "means2d": torch.zeros_like(means3d[:, :2]),
            "scores": torch.ones(means3d.shape[0], device=self.device),
        }

    def render(self, scene_data: dict, camera: Camera) -> torch.Tensor:
        from diff_gaussian_rasterization import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        if self._bg is None:
            self._bg = torch.zeros(3, dtype=torch.float32, device=self.device)
        settings = GaussianRasterizationSettings(
            image_height=camera.image_height,
            image_width=camera.image_width,
            tanfovx=camera.tanfovx,
            tanfovy=camera.tanfovy,
            bg=self._bg,
            scale_modifier=1.0,
            viewmatrix=camera.world_view_transform,
            projmatrix=camera.full_proj_transform,
            sh_degree=scene_data.get("sh_degree", 3),
            campos=camera.camera_center,
            prefiltered=False,
            debug=False,
        )
        rasterizer = self._rasterizers.get(id(camera))
        if rasterizer is None:
            rasterizer = GaussianRasterizer(raster_settings=settings)
            self._rasterizers[id(camera)] = rasterizer

        rendered, _, _, self.last_wrapper_seconds = rasterizer(
            means3D=scene_data["xyz"],
            means2D=scene_data["means2d"],
            opacities=scene_data["opacities_activated"],
            scores=scene_data["scores"],
            shs=scene_data["shs"],
            colors_precomp=None,
            scales=scene_data["scales_activated"],
            rotations=scene_data["rotations_normalized"],
            cov3D_precomp=None,
        )
        return rendered.permute(1, 2, 0).clamp(0, 1)
