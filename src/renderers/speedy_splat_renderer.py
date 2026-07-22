"""
speedy-splat renderer adapter (j-alex-hanson fork).

Wraps the speedy-splat CUDA rasterizer, which replaces Thrust radix sort
with CUB DeviceRadixSort for accelerated tile binning. This is the
highest-performing renderer in Phase 1 comparisons.

Reference:
    https://github.com/j-alex-hanson/speedy-splat
"""
import torch
from .base import RendererAdapter
from benchmark_framework import Camera


class SpeedySplatRenderer(RendererAdapter):
    """Adapter for speedy-splat renderer (j-alex-hanson fork).

    Implements the optimized rasterizer using CUB DeviceRadixSort for the
    tile-based depth ordering step. CUB is NVIDIA's officially maintained
    CUDA C++ core library with warp-level primitives that reduce shared
    memory bank conflicts and instruction-level parallelism overhead.

    Key architectural difference from diff_gaussian:
      - Uses CUB DeviceRadixSort instead of Thrust radix sort.
      - Accepts a `scores` parameter for importance-based sampling.
      - Does not expose the `antialiasing` flag.
    """

    name = "speedy_splat"
    package_name = "speedy-gaussian-rasterization"
    module_name = "speedy_gaussian_rasterization"
    implementation = "j-alex-hanson/speedy-splat"
    source_url = "https://github.com/j-alex-hanson/speedy-splat"

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self._available = None
        self._rasterizers = {}
        self._bg = None

    def is_available(self) -> bool:
        """Check if speedy_gaussian_rasterization is importable."""
        if self._available is None:
            try:
                self._backend()
                self._available = True
            except (ImportError, OSError):
                self._available = False
        return self._available

    @staticmethod
    def _backend():
        import speedy_gaussian_rasterization
        return speedy_gaussian_rasterization

    def prepare_scene(self, scene_data: dict) -> dict:
        """Activate static parameters and allocate fixed buffers once."""
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
        """Render one frame using the speedy-splat rasterization backend.

        Args:
            scene_data: Scene data dictionary with xyz, opacity, scales,
                       rotations, shs tensors.
            camera: Camera parameters for the current viewpoint.

        Returns:
            RGB image tensor of shape (H, W, 3) with values in [0, 1].
        """
        backend = self._backend()
        GaussianRasterizationSettings = backend.GaussianRasterizationSettings
        GaussianRasterizer = backend.GaussianRasterizer

        if self._bg is None:
            self._bg = torch.zeros(3, dtype=torch.float32, device=self.device)

        # NOTE: speedy-splat does not expose the antialiasing parameter
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
        )
        rasterizer = self._rasterizers.get(id(camera))
        if rasterizer is None:
            rasterizer = GaussianRasterizer(raster_settings=raster_settings)
            self._rasterizers[id(camera)] = rasterizer

        means3d = scene_data["xyz"].contiguous()
        opacities = scene_data["opacities_activated"]
        shs = scene_data["shs"].contiguous()
        scales = scene_data["scales_activated"]
        rotations = scene_data["rotations_normalized"]
        means2d = scene_data["means2d"]

        # scores: importance sampling weights (all ones retains all Gaussians)
        scores = scene_data["scores"]

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


class SpeedySplatRawRenderer(SpeedySplatRenderer):
    """Ablation baseline with the original per-frame wrapper allocations."""

    name = "speedy_splat_raw"
    implementation = "j-alex-hanson/speedy-splat (uncached wrapper ablation)"

    def prepare_scene(self, scene_data: dict) -> dict:
        return scene_data

    def render(self, scene_data: dict, camera: Camera) -> torch.Tensor:
        means3d = scene_data["xyz"].contiguous()
        transient = {
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
        self._rasterizers.clear()
        self._bg = None
        return super().render(transient, camera)
