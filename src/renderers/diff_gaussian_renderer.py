"""
diff-gaussian-rasterization renderer adapter.

Wraps the original Inria 3DGS CUDA rasterizer, which uses
Thrust radix sort for tile binning. This serves as the baseline renderer
in the benchmark comparison.

Reference:
    https://github.com/graphdeco-inria/diff-gaussian-rasterization
"""
import inspect
import torch
from weakref import WeakKeyDictionary
from .base import RendererAdapter
from benchmark_framework import Camera


class DiffGaussianRenderer(RendererAdapter):
    """Adapter for diff-gaussian-rasterization (graphdeco-inria)."""

    name = "diff_gaussian"
    package_name = "diff-gaussian-rasterization"
    module_name = "diff_gaussian_rasterization"
    implementation = "graphdeco-inria/diff-gaussian-rasterization"
    source_url = "https://github.com/graphdeco-inria/diff-gaussian-rasterization"

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self._available = None
        self._rasterizers = WeakKeyDictionary()
        self._bg = None

    def is_available(self) -> bool:
        if self._available is None:
            try:
                from diff_gaussian_rasterization import (
                    GaussianRasterizationSettings,
                    GaussianRasterizer,
                )
                self._available = True
            except ImportError:
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
        }

    def render(self, scene_data: dict, camera: Camera) -> torch.Tensor:
        from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

        if self._bg is None:
            self._bg = torch.zeros(3, dtype=torch.float32, device=self.device)

        settings_kwargs = dict(
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
        if "antialiasing" in inspect.signature(GaussianRasterizationSettings).parameters:
            settings_kwargs["antialiasing"] = False
        raster_settings = GaussianRasterizationSettings(**settings_kwargs)
        rasterizer = self._rasterizers.get(camera)
        if rasterizer is None:
            rasterizer = GaussianRasterizer(raster_settings=raster_settings)
            self._rasterizers[camera] = rasterizer

        means3d = scene_data["xyz"].contiguous()
        opacities = scene_data["opacities_activated"]
        shs = scene_data["shs"].contiguous()
        scales = scene_data["scales_activated"]
        rotations = scene_data["rotations_normalized"]
        means2d = scene_data["means2d"]

        render_result = rasterizer(
            means3D=means3d, means2D=means2d, opacities=opacities, shs=shs,
            colors_precomp=None, scales=scales, rotations=rotations, cov3D_precomp=None,
        )
        if len(render_result) not in (3, 4):
            raise ValueError(
                f"diff-gaussian rasterizer returned {len(render_result)} values; expected 3 or 4"
            )
        rendered_image = render_result[0]

        return rendered_image.permute(1, 2, 0).clamp(0, 1)
