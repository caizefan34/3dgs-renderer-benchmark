import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from renderers.gemm_gs_renderer import GemmGSRenderer
from renderers.local_gs_renderer import LocalGSRenderer


class CandidateRendererAdapterTest(unittest.TestCase):
    def test_local_gs_uses_pinned_drop_in_api_identity(self):
        renderer = LocalGSRenderer(device="cpu")
        self.assertEqual(renderer.module_name, "diff_gaussian_rasterization")
        self.assertIn("Local-GS", renderer.implementation)

    def test_gemm_gs_uses_diff_gaussian_compatible_api(self):
        renderer = GemmGSRenderer(device="cpu")
        self.assertEqual(renderer.module_name, "diff_gaussian_rasterization")
        self.assertIn("GEMM-GS", renderer.implementation)


if __name__ == "__main__":
    unittest.main()
