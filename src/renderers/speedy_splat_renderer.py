"""
speedy-splat renderer adapter.
Uses CUB DeviceRadixSort optimized rasterization backend.
"""
import torch
from .base import RendererAdapter
from benchmark_framework import Camera


class SpeedySplatRenderer(RendererAdapter):
    """Adapter for speedy-splat (j-alex-hanson).
    
    Optimized diff-gaussian-rasterization with CUB DeviceRadixSort.
    """
    
    name = "speedy_splat"
    
    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self._available = None
    
    def is_available(self) -> bool:
        if self._available is None:
            try:
                from speedy_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
                self._available = True
            except ImportError:
                self._available = False
        return self._available
    
    def prepare_scene(self, scene_data: dict) -> dict:
        return scene_data
    
    def render(self, scene_data: dict, camera: Camera) -> torch.Tensor:
        from speedy_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
        
        bg = torch.zeros(3, dtype=torch.float32, device=self.device)
        
        # NOTE: speedy-splat version does NOT have antialiasing field
        raster_settings = GaussianRasterizationSettings(
            image_height=camera.image_height,
            image_width=camera.image_width,
            tanfovx=camera.tanfovx,
            tanfovy=camera.tanfovy,
            bg=bg,
            scale_modifier=1.0,
            viewmatrix=camera.world_view_transform,
            projmatrix=camera.full_proj_transform,
            sh_degree=3,
            campos=camera.camera_center,
            prefiltered=False,
            debug=False,
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        
        means3d = scene_data["xyz"].contiguous()
        opacities = torch.sigmoid(scene_data["opacity"]).contiguous()
        shs = scene_data["shs"].contiguous()
        scales = scene_data["scales"].contiguous()
        rotations = torch.nn.functional.normalize(scene_data["rotations"], dim=-1).contiguous()
        means2d = torch.zeros_like(means3d[:, :2])
        
        # scores: importance sampling weights (all ones = keep all)
        scores = torch.ones(means3d.shape[0], device=self.device)
        
        rendered_image, radii, _ = rasterizer(
            means3D=means3d,
            means2D=means2d,
            opacities=opacities,
            scores=scores,
            shs=shs,
            colors_precomp=None,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None,
        )
        
        return rendered_image.permute(1, 2, 0).clamp(0, 1)
