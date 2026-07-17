"""Load verified cached gsplat binaries while importing Python from pinned sources."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch  # Load PyTorch and CUDA DLLs before cached extension modules.


ROOT = Path(r"C:\Users\36570\Documents\Codex\3dgs-renderer-benchmark")
SOURCE_ROOT = ROOT / "artifacts" / "renderer-sources"
EXTENSION_DIRS = (
    Path(r"C:\Users\36570\3dgs-renderer-benchmark\results\task_b_gsplat_build\gsplat_cuda"),
    Path(r"C:\Users\36570\AppData\Local\torch_extensions\torch_extensions\Cache\py310_cu130\gsplat_scene_cuda"),
    Path(r"C:\Users\36570\AppData\Local\torch_extensions\torch_extensions\Cache\py310_cu130\experimental_gaussian_render_inference_scene_cuda"),
)

for path in (
    SOURCE_ROOT / "gsplat",
    SOURCE_ROOT / "speedy-splat",
    *EXTENSION_DIRS,
):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

sys.modules["gsplat.csrc"] = importlib.import_module("gsplat_cuda")
sys.modules["gsplat.experimental.render.kernels.csrc"] = importlib.import_module(
    "experimental_gaussian_render_inference_scene_cuda"
)
