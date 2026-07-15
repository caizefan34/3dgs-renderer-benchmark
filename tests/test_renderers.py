import importlib
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class RendererRegistryTest(unittest.TestCase):
    def test_registry_import_does_not_require_optional_backends(self):
        sys.modules.pop("renderers", None)
        module = importlib.import_module("renderers")
        self.assertIn("gsplat", module.list_renderers())
        self.assertIn("original_3dgs", module.list_renderers())
        self.assertIn("tcgs", module.list_renderers())


class CameraConventionTest(unittest.TestCase):
    def test_generated_camera_faces_scene_center(self):
        from benchmark_framework import generate_cameras, validate_cameras_facing_point

        cameras = generate_cameras(4, image_width=32, image_height=18, device="cpu")
        depths = validate_cameras_facing_point(cameras, torch.zeros(3))
        self.assertTrue(all(depth > 0 for depth in depths))
        for camera in cameras:
            torch.testing.assert_close(
                camera.full_proj_transform,
                camera.world_view_transform @ camera.projmatrix.T,
            )

    def test_loads_original_3dgs_camera_export(self):
        import json
        import tempfile
        from benchmark_framework import load_cameras_from_json

        payload = [{
            "id": 0, "img_name": "frame_001", "width": 8, "height": 4,
            "position": [0.0, 0.0, 2.0],
            "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "fx": 8.0, "fy": 8.0,
            "reference_crop": [0, 1, 8, 3],
        }]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cameras.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            camera = load_cameras_from_json(str(path), device="cpu")[0]

        self.assertEqual(camera.image_name, "frame_001")
        self.assertEqual(camera.reference_crop, (0, 1, 8, 3))
        torch.testing.assert_close(camera.camera_center, torch.tensor([0.0, 0.0, 2.0]))
        torch.testing.assert_close(camera.viewmatrix[:3, 3], torch.tensor([0.0, 0.0, -2.0]))

    def test_external_camera_resolution_is_reported(self):
        from run_benchmark import _uniform_camera_resolution
        from benchmark_framework import generate_cameras

        cameras = generate_cameras(2, image_width=19, image_height=11, device="cpu")
        self.assertEqual(_uniform_camera_resolution(cameras), (19, 11))

    def test_resize_cameras_preserves_fov_and_scales_intrinsics(self):
        from benchmark_framework import generate_cameras, resize_cameras

        camera = generate_cameras(1, image_width=8, image_height=4, device="cpu")[0]
        resized = resize_cameras([camera], 16, 9)[0]

        self.assertEqual((resized.image_width, resized.image_height), (16, 9))
        self.assertEqual(resized.fov_x, camera.fov_x)
        self.assertEqual(resized.fov_y, camera.fov_y)
        torch.testing.assert_close(resized.K[0, 0], camera.K[0, 0] * 2.0)
        torch.testing.assert_close(resized.K[1, 1], camera.K[1, 1] * 2.25)
        torch.testing.assert_close(resized.viewmatrix, camera.viewmatrix)

    def test_named_resolution_presets(self):
        from run_benchmark import _requested_resolution

        for name, expected in {
            "720p": (1280, 720),
            "1080p": (1920, 1080),
            "4k": (3840, 2160),
        }.items():
            args = SimpleNamespace(resolution=name, width=None, height=None)
            self.assertEqual(_requested_resolution(args), expected)


class HiGSAutoConfigTest(unittest.TestCase):
    def test_scale_threshold(self):
        from renderers.gsplat_renderer import GsplatHiGSAutoRenderer

        self.assertEqual(GsplatHiGSAutoRenderer.select_config(200_000), (16, "none"))
        self.assertEqual(GsplatHiGSAutoRenderer.select_config(400_000), (8, "32b"))


class GsplatRendererTest(unittest.TestCase):
    def _camera(self):
        from benchmark_framework import Camera

        eye = torch.eye(4)
        return Camera(
            image_width=8,
            image_height=4,
            fov_x=1.0,
            fov_y=0.6,
            viewmatrix=eye,
            projmatrix=eye,
            camera_center=torch.zeros(3),
            world_view_transform=eye,
            full_proj_transform=eye,
            tanfovx=0.5,
            tanfovy=0.25,
            K=torch.tensor([[8.0, 0.0, 4.0], [0.0, 8.0, 2.0], [0.0, 0.0, 1.0]]),
        )

    def _scene(self):
        return {
            "xyz": torch.zeros(2, 3),
            "rotations": torch.tensor([[2.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
            "scales": torch.zeros(2, 3),
            "opacity": torch.zeros(2),
            "shs": torch.zeros(2, 16, 3),
            "num_points": 2,
        }

    def test_calls_real_gsplat_api(self):
        calls = []

        def rasterization(**kwargs):
            calls.append(kwargs)
            return torch.zeros(1, 4, 8, 3), torch.zeros(1, 4, 8, 1), {}

        fake_gsplat = types.ModuleType("gsplat")
        fake_gsplat.__version__ = "test"
        fake_gsplat.rasterization = rasterization

        with mock.patch.dict(sys.modules, {"gsplat": fake_gsplat}):
            from renderers.gsplat_renderer import GsplatRenderer

            renderer = GsplatRenderer(device="cpu", packed=True)
            prepared = renderer.prepare_scene(self._scene())
            image = renderer.render(prepared, self._camera())

        self.assertEqual(image.shape, (4, 8, 3))
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0]["packed"])
        self.assertEqual(calls[0]["sh_degree"], 3)
        self.assertEqual(tuple(calls[0]["colors"].shape), (2, 16, 3))
        torch.testing.assert_close(calls[0]["scales"], torch.ones(2, 3))
        torch.testing.assert_close(calls[0]["opacities"], torch.full((2,), 0.5))

    def test_dense_sets_packed_false(self):
        calls = []

        def rasterization(**kwargs):
            calls.append(kwargs)
            return torch.zeros(1, 4, 8, 3), None, {}

        fake_gsplat = types.ModuleType("gsplat")
        fake_gsplat.rasterization = rasterization
        with mock.patch.dict(sys.modules, {"gsplat": fake_gsplat}):
            from renderers.gsplat_renderer import GsplatDenseRenderer

            renderer = GsplatDenseRenderer(device="cpu")
            renderer.render(renderer.prepare_scene(self._scene()), self._camera())

        self.assertFalse(calls[0]["packed"])


class Original3DGSRendererTest(unittest.TestCase):
    def test_uses_scene_sh_degree(self):
        settings_seen = []

        class Settings:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                settings_seen.append(kwargs)

        class Rasterizer:
            def __init__(self, raster_settings):
                self.settings = raster_settings

            def __call__(self, **kwargs):
                return (
                    torch.zeros(3, 4, 8), torch.zeros(1),
                    torch.zeros(1, 4, 8), torch.zeros(1, 4, 8),
                )

        fake = types.ModuleType("diff_gaussian_rasterization")
        fake.GaussianRasterizationSettings = Settings
        fake.GaussianRasterizer = Rasterizer
        with mock.patch.dict(sys.modules, {"diff_gaussian_rasterization": fake}):
            from renderers.diff_gaussian_renderer import DiffGaussianRenderer
            from benchmark_framework import generate_cameras

            scene = {
                "xyz": torch.zeros(2, 3), "opacity": torch.zeros(2),
                "scales": torch.zeros(2, 3),
                "rotations": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
                "shs": torch.zeros(2, 4, 3), "sh_degree": 1,
            }
            renderer = DiffGaussianRenderer(device="cpu")
            renderer.render(
                renderer.prepare_scene(scene),
                generate_cameras(1, image_width=8, image_height=4, device="cpu")[0],
            )

        self.assertEqual(settings_seen[0]["sh_degree"], 1)


class TCGSRendererTest(unittest.TestCase):
    def test_uses_scores_and_four_value_result(self):
        settings_seen = []
        calls = []

        class Settings:
            def __init__(self, **kwargs):
                settings_seen.append(kwargs)

        class Rasterizer:
            def __init__(self, raster_settings):
                pass

            def __call__(self, **kwargs):
                calls.append(kwargs)
                return (
                    torch.zeros(3, 4, 8),
                    torch.zeros(2),
                    torch.zeros(1, 4, 8),
                    0.001,
                )

        fake = types.ModuleType("diff_gaussian_rasterization")
        fake.GaussianRasterizationSettings = Settings
        fake.GaussianRasterizer = Rasterizer
        with mock.patch.dict(sys.modules, {"diff_gaussian_rasterization": fake}):
            from benchmark_framework import generate_cameras
            from renderers.tcgs_renderer import TCGSRenderer

            scene = {
                "xyz": torch.zeros(2, 3),
                "opacity": torch.zeros(2),
                "scales": torch.zeros(2, 3),
                "rotations": torch.tensor([[2.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
                "shs": torch.zeros(2, 4, 3),
                "sh_degree": 1,
            }
            renderer = TCGSRenderer(device="cpu")
            image = renderer.render(
                renderer.prepare_scene(scene),
                generate_cameras(1, image_width=8, image_height=4, device="cpu")[0],
            )

        self.assertEqual(image.shape, (4, 8, 3))
        self.assertEqual(settings_seen[0]["sh_degree"], 1)
        torch.testing.assert_close(calls[0]["scores"], torch.ones(2))
        torch.testing.assert_close(calls[0]["opacities"], torch.full((2,), 0.5))
        torch.testing.assert_close(calls[0]["scales"], torch.ones(2, 3))
        torch.testing.assert_close(
            calls[0]["rotations"],
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
        )


if __name__ == "__main__":
    unittest.main()
