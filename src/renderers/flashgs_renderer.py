"""Adapter for the pinned InternLandMark/FlashGS CUDA extension."""
from __future__ import annotations

import os

import torch

from benchmark_framework import Camera, compute_cov3d_from_scales_rot

from .base import RendererAdapter


class FlashGSRenderer(RendererAdapter):
    name = "flashgs"
    package_name = "flash-gaussian-splatting"
    module_name = "flash_gaussian_splatting"
    implementation = "InternLandMark/FlashGS"
    source_url = "https://github.com/InternLandMark/FlashGS"
    source_commit = "cdfc4e4002318423eda356eed02df8e01fa32cb6"

    def __init__(self, device: str = "cuda", max_rendered: int | None = None):
        super().__init__(device)
        self.max_rendered = max_rendered or int(os.environ.get("FLASHGS_MAX_RENDERED", 1 << 27))
        self._available = None
        self._ops = None
        self._buffers = None
        self._frame_buffers = {}

    def is_available(self) -> bool:
        if self._available is None:
            try:
                import flash_gaussian_splatting
                self._ops = flash_gaussian_splatting.ops
                self._available = True
            except (ImportError, OSError):
                self._available = False
        return self._available

    def metadata(self) -> dict:
        result = super().metadata()
        if self.is_available():
            result["commit_hash"] = self.source_commit
        return result

    def prepare_scene(self, scene_data: dict) -> dict:
        if not self.is_available():
            raise RuntimeError("FlashGS CUDA extension is not installed")
        point_count = int(scene_data["xyz"].shape[0])
        sort_bytes = int(self._ops.get_sort_buffer_size(self.max_rendered))
        self._buffers = {
            "keys_unsorted": torch.empty(self.max_rendered, device=self.device, dtype=torch.int64),
            "values_unsorted": torch.empty(self.max_rendered, device=self.device, dtype=torch.int32),
            "keys_sorted": torch.empty(self.max_rendered, device=self.device, dtype=torch.int64),
            "values_sorted": torch.empty(self.max_rendered, device=self.device, dtype=torch.int32),
            "sort_space": torch.empty(sort_bytes, device=self.device, dtype=torch.int8),
            "offset": torch.zeros(1, device=self.device, dtype=torch.int32),
            "points_xy": torch.empty((point_count, 2), device=self.device, dtype=torch.float32),
            "rgb_depth": torch.empty((point_count, 4), device=self.device, dtype=torch.float32),
            "conic_opacity": torch.empty((point_count, 4), device=self.device, dtype=torch.float32),
        }
        return {
            **scene_data,
            "xyz": scene_data["xyz"].contiguous(),
            "shs": scene_data["shs"].contiguous(),
            "opacities_activated": torch.sigmoid(scene_data["opacity"]).contiguous(),
            "cov3d": compute_cov3d_from_scales_rot(
                scene_data["scales"], scene_data["rotations"]
            ).contiguous(),
        }

    def _frame_buffer(self, width: int, height: int) -> dict:
        key = (width, height)
        if key not in self._frame_buffers:
            tile_count = ((width + 15) // 16) * ((height + 15) // 16)
            self._frame_buffers[key] = {
                "ranges": torch.empty((tile_count, 2), device=self.device, dtype=torch.int32),
                "output": torch.empty((height, width, 3), device=self.device, dtype=torch.int8),
                "background": torch.zeros(3, device=self.device, dtype=torch.float32),
            }
        return self._frame_buffers[key]

    def render(self, scene_data: dict, camera: Camera) -> torch.Tensor:
        buffers = self._buffers
        if buffers is None:
            raise RuntimeError("prepare_scene must be called before render")
        width, height = camera.image_width, camera.image_height
        frame = self._frame_buffer(width, height)
        camera_to_world = torch.linalg.inv(camera.viewmatrix)
        buffers["offset"].zero_()
        self._ops.preprocess(
            scene_data["xyz"], scene_data["shs"], scene_data["opacities_activated"],
            scene_data["cov3d"], width, height, 16, 16,
            camera.camera_center, camera_to_world[:3, :3].contiguous(),
            float(camera.K[0, 0]), float(camera.K[1, 1]), 100.0, 0.01,
            buffers["points_xy"], buffers["rgb_depth"], buffers["conic_opacity"],
            buffers["keys_unsorted"], buffers["values_unsorted"], buffers["offset"],
        )
        rendered_count = int(buffers["offset"].item())
        if rendered_count >= self.max_rendered:
            raise RuntimeError(
                f"FlashGS generated {rendered_count} intersections; "
                f"increase FLASHGS_MAX_RENDERED above {self.max_rendered}"
            )
        self._ops.sort_gaussian(
            rendered_count, width, height, 16, 16, buffers["sort_space"],
            buffers["keys_unsorted"], buffers["values_unsorted"],
            buffers["keys_sorted"], buffers["values_sorted"],
        )
        self._ops.render_16x16(
            rendered_count, width, height,
            buffers["points_xy"], buffers["rgb_depth"], buffers["conic_opacity"],
            buffers["keys_sorted"], buffers["values_sorted"], frame["ranges"],
            frame["background"], frame["output"],
        )
        return frame["output"].view(torch.uint8).float().mul_(1.0 / 255.0)
