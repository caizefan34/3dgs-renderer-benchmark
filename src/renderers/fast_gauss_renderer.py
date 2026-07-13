"""
fast-gaussian-rasterization renderer adapter (dendenxu fork).

Wraps the fast-gaussian-rasterization CUDA backend, which uses CUDA-GL
interop with OpenGL geometry shaders for rendering. Requires EGL (headless
OpenGL) or a display server, and is therefore primarily available on Linux
or WSL2 environments.

Reference:
    https://github.com/dendenxu/fast-gaussian-rasterization
"""
import torch
from .base import RendererAdapter
from benchmark_framework import Camera


class FastGaussRenderer(RendererAdapter):
    """Adapter for fast-gaussian-rasterization (dendenxu fork).

    Uses CUDA-GL interop with OpenGL geometry shaders for the rasterization
    step. This approach offloads part of the rendering pipeline to the GPU's
    graphics hardware, but requires EGL or a display server.

    Platform dependency: Currently only available on Linux with EGL support
    or WSL2 environments. Not available on native Windows.
    """

    name = "fast_gauss"

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self._available = None
        self._rasterizer = None

    def is_available(self) -> bool:
        """Check if fast_gauss and EGL are available.

        Requires EGL for headless OpenGL rendering. On systems without EGL
        (e.g., native Windows), this renderer will be unavailable.
        """
        if self._available is None:
            try:
                import fast_gauss
                from fast_gauss.gl_utils import eglctx
                self._available = eglctx is not None
                if not self._available:
                    print("  fast_gauss: EGL context not available (requires Linux/WSL2 with EGL)")
            except (ImportError, Exception) as e:
                self._available = False
                print(f"  fast_gauss not available: {e}")
        return self._available

    def prepare_scene(self, scene_data: dict) -> dict:
        """Pass through scene data without modification."""
        return scene_data

    def render(self, scene_data: dict, camera: Camera) -> torch.Tensor:
        """Render one frame using the fast-gaussian-rasterization backend.

        Args:
            scene_data: Scene data dictionary with xyz, opacity, scales,
                       rotations, shs tensors.
            camera: Camera parameters for the current viewpoint.

        Returns:
            RGB image tensor of shape (H, W, 3) with values in [0, 1].
        """
        from fast_gauss import GaussianRasterizationSettings, GaussianRasterizer

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

        # Prepare per-Gaussian data tensors
        means3d = scene_data["xyz"].contiguous()
        opacities = torch.sigmoid(scene_data["opacity"]).contiguous()
        shs = scene_data["shs"].contiguous()
        scales = scene_data["scales"].contiguous()
        rotations = torch.nn.functional.normalize(scene_data["rotations"], dim=-1).contiguous()
        means2d = torch.zeros_like(means3d[:, :2])

        # Render; output is (H, W, 4) RGBA
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

        # Extract RGB channels, discard alpha
        return rendered[..., :3].clamp(0, 1)