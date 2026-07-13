"""
diff-gaussian-rasterization (ashawkey fork) renderer adapter.
"""
import torch
from .base import RendererAdapter
from benchmark_framework import Camera


class DiffGaussianRenderer(RendererAdapter):
    """Adapter for diff-gaussian-rasterization (ashawkey fork)."""
    
    name = "diff_gaussian"
    
    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self._available = None
        self._bg = torch.zeros(3, dtype=torch.float32, device=self.device)
        self._rasterizer_cache = {}
    
    def is_available(self) -> bool:
        if self._available is None:
            try:
                import diff_gaussian_rasterization
                self._available = True
            except ImportError:
                self._available = False
        return self._available
    
    def prepare_scene(self, scene_data: dict) -> dict:
        means3d = scene_data["xyz"].contiguous()
        return {
            "num_points": scene_data.get("num_points", means3d.shape[0]),
            "xyz": means3d,
            "opacities": torch.sigmoid(scene_data["opacity"]).contiguous(),
            "shs": scene_data["shs"].contiguous(),
            "scales": scene_data["scales"].contiguous(),
            "rotations": torch.nn.functional.normalize(scene_data["rotations"], dim=-1).contiguous(),
            "means2d": torch.zeros_like(means3d[:, :2]),
        }

    def _get_rasterizer(self, camera: Camera):
        from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
        cam_key = id(camera)
        rasterizer = self._rasterizer_cache.get(cam_key)
        if rasterizer is not None:
            return rasterizer
        raster_settings = GaussianRasterizationSettings(
            image_height=camera.image_height,
            image_width=camera.image_width,
            tanfovx=camera.tanfovx,
            tanfovy=camera.tanfovy,
            bg=self._bg,
            scale_modifier=1.0,
            viewmatrix=camera.world_view_transform,
            projmatrix=camera.full_proj_transform,
            sh_degree=3,
            campos=camera.camera_center,
            prefiltered=False,
            debug=False,
            antialiasing=False,
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        self._rasterizer_cache[cam_key] = rasterizer
        return rasterizer
    
    def render(self, scene_data: dict, camera: Camera) -> torch.Tensor:
        prepared = scene_data if "opacities" in scene_data else self.prepare_scene(scene_data)
        means2d = prepared["means2d"]
        means2d.zero_()
        rasterizer = self._get_rasterizer(camera)

        # Render
        rendered_image, radii, _ = rasterizer(
            means3D=prepared["xyz"],
            means2D=means2d,
            opacities=prepared["opacities"],
            shs=prepared["shs"],
            colors_precomp=None,
            scales=prepared["scales"],
            rotations=prepared["rotations"],
            cov3D_precomp=None,
        )
        
        # rendered_image is (C, H, W) in [0,1]
        return rendered_image.permute(1, 2, 0).clamp(0, 1)
