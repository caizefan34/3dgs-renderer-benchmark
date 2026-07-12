"""
gsplat renderer adapter - uses diff-gaussian-rasterization (ashawkey fork).
Since gsplat's native CUDA kernels are not compiled, this adapter delegates to
the diff-gaussian-rasterization package directly, matching gsplat's parameter
setup style.

Note: On this system, gsplat lacks native CUDA kernels (CUDA 13.0 + MSVC 14.44
compatibility issue). The gsplat Inria wrapper also has compatibility issues
with the ashawkey fork API. So we use diff-gaussian-rasterization directly.
"""
import torch
from .base import RendererAdapter
from benchmark_framework import Camera


class GsplatRenderer(RendererAdapter):
    """Adapter that uses gsplat-style parameter management with diff-gaussian-rasterization backend."""
    
    name = "gsplat"
    
    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self._available = None
        self._rasterizer = None
        self._current_settings = None
    
    def is_available(self) -> bool:
        if self._available is None:
            try:
                from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
                self._available = True
            except ImportError:
                self._available = False
        return self._available
    
    def prepare_scene(self, scene_data: dict) -> dict:
        return scene_data
    
    def _ensure_rasterizer(self, camera: Camera):
        """Create rasterizer if settings changed."""
        from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
        
        W, H = camera.image_width, camera.image_height
        
        settings = GaussianRasterizationSettings(
            image_height=H,
            image_width=W,
            tanfovx=camera.tanfovx,
            tanfovy=camera.tanfovy,
            bg=torch.zeros(3, dtype=torch.float32, device=self.device),
            scale_modifier=1.0,
            viewmatrix=camera.world_view_transform,
            projmatrix=camera.full_proj_transform,
            sh_degree=3,
            campos=camera.camera_center,
            prefiltered=False,
            debug=False,
            antialiasing=False,
        )
        self._rasterizer = GaussianRasterizer(raster_settings=settings)
    
    def render(self, scene_data: dict, camera: Camera) -> torch.Tensor:
        self._ensure_rasterizer(camera)
        
        N = scene_data["num_points"]
        means3d = scene_data["xyz"].contiguous()
        quats = torch.nn.functional.normalize(scene_data["rotations"], dim=-1).contiguous()
        scales = scene_data["scales"].contiguous()
        opacities = torch.sigmoid(scene_data["opacity"]).contiguous()
        shs = scene_data["shs"].contiguous()
        
        means2d = torch.zeros_like(means3d[:, :2])
        
        rendered_image, radii, _ = self._rasterizer(
            means3D=means3d,
            means2D=means2d,
            opacities=opacities,
            shs=shs,
            colors_precomp=None,
            scales=scales,
            rotations=quats,
            cov3D_precomp=None,
        )
        
        return rendered_image.permute(1, 2, 0).clamp(0, 1)
