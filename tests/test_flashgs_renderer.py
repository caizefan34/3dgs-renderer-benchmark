import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark_framework import Camera
from renderers.flashgs_renderer import FlashGSRenderer


class FakeOps:
    def __init__(self):
        self.rotation = None

    def get_sort_buffer_size(self, count):
        return count

    def preprocess(self, *args):
        self.rotation = args[9]
        args[-1].fill_(3)

    def sort_gaussian(self, *args):
        pass

    def render_16x16(self, *args):
        args[-1].fill_(-1)


class FlashGSRendererTest(unittest.TestCase):
    def test_adapter_maps_camera_and_unsigned_output(self):
        ops = FakeOps()
        module = types.SimpleNamespace(ops=ops)
        camera = Camera(
            image_width=2, image_height=1, fov_x=1.0, fov_y=1.0,
            viewmatrix=torch.eye(4), projmatrix=torch.eye(4),
            camera_center=torch.zeros(3), world_view_transform=torch.eye(4),
            full_proj_transform=torch.eye(4), tanfovx=0.5, tanfovy=0.5,
            K=torch.tensor([[2.0, 0, 1.0], [0, 3.0, 0.5], [0, 0, 1.0]]),
        )
        scene = {
            "xyz": torch.zeros((1, 3)), "shs": torch.zeros((1, 16, 3)),
            "opacity": torch.zeros(1), "scales": torch.zeros((1, 3)),
            "rotations": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        }
        with mock.patch.dict(sys.modules, {"flash_gaussian_splatting": module}):
            renderer = FlashGSRenderer(device="cpu", max_rendered=8)
            prepared = renderer.prepare_scene(scene)
            image = renderer.render(prepared, camera)

        self.assertEqual(tuple(image.shape), (1, 2, 3))
        self.assertTrue(torch.all(image == 1.0))
        self.assertTrue(torch.equal(ops.rotation, torch.eye(3)))


if __name__ == "__main__":
    unittest.main()
