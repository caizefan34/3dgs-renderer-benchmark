"""Adapters for the real nerfstudio-project/gsplat rasterization API."""
import torch

from .base import RendererAdapter
from benchmark_framework import Camera


class GsplatRenderer(RendererAdapter):
    """Render with gsplat's CUDA rasterizer, not a diff-gaussian fallback."""

    name = "gsplat"
    package_name = "gsplat"
    module_name = "gsplat"
    implementation = "nerfstudio-project/gsplat rasterization (packed)"
    source_url = "https://github.com/nerfstudio-project/gsplat"

    def __init__(self, device: str = "cuda", packed: bool = True):
        super().__init__(device)
        self.packed = packed
        self._available = None

    def is_available(self) -> bool:
        if self._available is None:
            try:
                from gsplat import rasterization  # noqa: F401
                self._available = True
            except (ImportError, OSError):
                self._available = False
        return self._available

    def prepare_scene(self, scene_data: dict) -> dict:
        return {
            **scene_data,
            "quats": torch.nn.functional.normalize(
                scene_data["rotations"], dim=-1
            ).contiguous(),
            "scales_activated": torch.exp(scene_data["scales"]).contiguous(),
            "opacities_activated": torch.sigmoid(scene_data["opacity"]).contiguous(),
            "shs": scene_data["shs"].contiguous(),
        }

    def render(self, scene_data: dict, camera: Camera) -> torch.Tensor:
        from gsplat import rasterization

        rendered, _, _ = rasterization(
            means=scene_data["xyz"],
            quats=scene_data["quats"],
            scales=scene_data["scales_activated"],
            opacities=scene_data["opacities_activated"],
            colors=scene_data["shs"],
            viewmats=camera.viewmatrix.unsqueeze(0),
            Ks=camera.K.unsqueeze(0),
            width=camera.image_width,
            height=camera.image_height,
            sh_degree=scene_data.get("sh_degree", 3),
            packed=self.packed,
            render_mode="RGB",
        )
        return rendered[0].clamp(0, 1)


class GsplatDenseRenderer(GsplatRenderer):
    name = "gsplat_dense"
    implementation = "nerfstudio-project/gsplat rasterization (dense)"

    def __init__(self, device: str = "cuda"):
        super().__init__(device=device, packed=False)


class GsplatHiGSRenderer(RendererAdapter):
    """Stateful inference-only HiGS path included in current gsplat main."""

    name = "gsplat_higs"
    package_name = "gsplat"
    module_name = "gsplat"
    implementation = "nerfstudio-project/gsplat experimental HiGS (tile 8)"
    source_url = "https://github.com/nerfstudio-project/gsplat"

    def __init__(
        self,
        device: str = "cuda",
        tile_size: int = 8,
        sh_compression: str = "none",
    ):
        super().__init__(device)
        self.tile_size = tile_size
        self.sh_compression = sh_compression
        self.implementation = (
            "nerfstudio-project/gsplat experimental HiGS "
            f"(tile {tile_size}, SH {sh_compression})"
        )
        self._available = None
        self._renderer = None

    def is_available(self) -> bool:
        if self._available is None:
            try:
                from gsplat.experimental import (
                    GaussianInferenceRenderer,
                    GaussianInferenceScene,
                )  # noqa: F401
                self._available = True
            except (ImportError, OSError):
                self._available = False
        return self._available

    def prepare_scene(self, scene_data: dict) -> dict:
        from gsplat.experimental import GaussianInferenceRenderer, GaussianInferenceScene

        scene = GaussianInferenceScene.from_gaussian_tensors(
            scene_data["xyz"],
            torch.nn.functional.normalize(scene_data["rotations"], dim=-1),
            torch.exp(scene_data["scales"]),
            torch.sigmoid(scene_data["opacity"]),
            scene_data["shs"],
            sh_degree=scene_data.get("sh_degree", 3),
            sh_compression=self.sh_compression,
            id="benchmark",
        )
        self._renderer = GaussianInferenceRenderer(scene, tile_size=self.tile_size)
        return scene_data

    def render(self, scene_data: dict, camera: Camera) -> torch.Tensor:
        result = self._renderer.render(
            viewmat=camera.viewmatrix,
            K=camera.K,
            width=camera.image_width,
            height=camera.image_height,
        )
        return result.frame[0, ..., :3].clamp(0, 1)


class GsplatHiGSTile16Renderer(GsplatHiGSRenderer):
    name = "gsplat_higs_tile16"

    def __init__(self, device: str = "cuda"):
        super().__init__(device=device, tile_size=16)


class GsplatHiGSSH32Renderer(GsplatHiGSRenderer):
    name = "gsplat_higs_sh32"

    def __init__(self, device: str = "cuda"):
        super().__init__(device=device, sh_compression="32b")


class GsplatHiGSSH16Renderer(GsplatHiGSRenderer):
    name = "gsplat_higs_sh16"

    def __init__(self, device: str = "cuda"):
        super().__init__(device=device, sh_compression="16b")


class GsplatHiGSTile16SH32Renderer(GsplatHiGSRenderer):
    name = "gsplat_higs_tile16_sh32"

    def __init__(self, device: str = "cuda"):
        super().__init__(device=device, tile_size=16, sh_compression="32b")


class GsplatHiGSTile16SH16Renderer(GsplatHiGSRenderer):
    name = "gsplat_higs_tile16_sh16"

    def __init__(self, device: str = "cuda"):
        super().__init__(device=device, tile_size=16, sh_compression="16b")


class GsplatHiGSAutoRenderer(GsplatHiGSRenderer):
    """Scale-aware HiGS configuration calibrated by the local ablation."""

    name = "gsplat_higs_auto"

    @staticmethod
    def select_config(num_gaussians: int):
        if num_gaussians < 300_000:
            return 16, "none"
        return 8, "32b"

    def prepare_scene(self, scene_data: dict) -> dict:
        self.tile_size, self.sh_compression = self.select_config(
            scene_data["num_points"]
        )
        self.implementation = (
            "nerfstudio-project/gsplat experimental HiGS auto "
            f"(tile {self.tile_size}, SH {self.sh_compression})"
        )
        return super().prepare_scene(scene_data)
