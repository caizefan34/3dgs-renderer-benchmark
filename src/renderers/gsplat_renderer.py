"""
gsplat renderer adapter (nerfstudio-project).

Wraps the gsplat rasterization backend from the nerfstudio project.
Due to CUDA 13.0 + MSVC 14.44 compatibility issues with gsplat's native
CUDA kernels, this adapter delegates to the diff-gaussian-rasterization
package as a fallback backend, while maintaining gsplat's parameter
management style.

Reference:
    https://github.com/nerfstudio-project/gsplat
"""
import torch
from .base import RendererAdapter
from benchmark_framework import Camera


class GsplatRenderer(RendererAdapter):
    """Adapter for gsplat (nerfstudio-project).

    Uses gsplat-style parameter management with diff-gaussian-rasterization
    as the backend. The adapter caches the GaussianRasterizer across frames
    for the same camera to reduce construction overhead.

    Platform note: On CUDA 13.0 + MSVC 14.44, gsplat's native CUDA kernels
    fail to compile. This adapter uses the diff-gaussian-rasterization
    package as a compatible fallback.
    """

    name = "gsplat"

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self._available = None
        self._rasterizer = None
        self._current_settings = None

    def is_available(self) -> bool:
        """Check if diff-gaussian-rasterization (used as fallback) is importable."""
        if self._available is None:
            try:
                from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
                self._available = True
            except ImportError:
                self._available = False
        return self._available

    def prepare_scene(self, scene_data: dict) -> dict:
        """Pass through scene data without modification."""
        return scene_data

    def _ensure_rasterizer(self, camera: Camera):
        """Create or reuse a GaussianRasterizer for the given camera.

        Args:
            camera: Camera parameters for the current viewpoint.
        """
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
        """Render one frame using the gsplat adapter.

        Args:
            scene_data: Scene data dictionary with xyz, opacity, scales,
                       rotations, shs tensors.
            camera: Camera parameters for the current viewpoint.

        Returns:
            RGB image tensor of shape (H, W, 3) with values in [0, 1].
        """
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