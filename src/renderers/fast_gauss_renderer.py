"""
fast-gaussian-rasterization renderer adapter.
Uses CUDA-GL interop (geometry shaders). Only works with EGL/display on Linux.
"""
import torch
from .base import RendererAdapter
from benchmark_framework import Camera


class FastGaussRenderer(RendererAdapter):
    """Adapter for fast-gaussian-rasterization (dendenxu).
    
    NOTE: This renderer uses CUDA-GL interop with OpenGL geometry shaders.
    Requires EGL (headless) or display. Currently NOT available on Windows without WSL2.
    """
    
    name = "fast_gauss"
    
    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self._available = None
        self._rasterizer = None
    
    def is_available(self) -> bool:
        if self._available is None:
            try:
                import fast_gauss
                # Check if EGL is available (headless rendering)
                from fast_gauss.gl_utils import eglctx
                self._available = eglctx is not None
                if not self._available:
                    print("  fast_gauss: EGL not available (requires Linux/WSL2 with EGL)")
            except (ImportError, Exception) as e:
                self._available = False
                print(f"  fast_gauss not available: {e}")
        return self._available
    
    def prepare_scene(self, scene_data: dict) -> dict:
        return scene_data
    
    def render(self, scene_data: dict, camera: Camera) -> torch.Tensor:
        from fast_gauss import GaussianRasterizationSettings, GaussianRasterizer
        
        # Create settings if needed
        bg = torch.zeros(3, dtype=torch.float32, device=self.device)
        settings = GaussianRasterizationSettings(
            image_height=camera.image_height,
            image_width=camera.image_width,
            tanfovx=camera.tanfovx,
            tanfovy=camera.tanfovy,
            bg=bg,
            scale_modifier=1.0,
            viewmatrix=camera.viewmatrix,
            projmatrix=camera.projmatrix,
            campos=camera.camera_center,
            sh_degree=3,
        )
        
        if self._rasterizer is None:
            self._rasterizer = GaussianRasterizer(settings)
        
        # Data
        means3d = scene_data["xyz"].contiguous()
        opacities = torch.sigmoid(scene_data["opacity"]).contiguous()
        shs = scene_data["shs"].contiguous()
        scales = scene_data["scales"].contiguous()
        rotations = torch.nn.functional.normalize(scene_data["rotations"], dim=-1).contiguous()
        means2d = torch.zeros_like(means3d[:, :2])
        
        # Render
        rendered = self._rasterizer(
            means3D=means3d,
            means2D=means2d,
            opacities=opacities,
            shs=shs,
            colors_precomp=None,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None,
        )
        
        # Output is (H, W, 4) RGBA
        return rendered[..., :3].clamp(0, 1)
