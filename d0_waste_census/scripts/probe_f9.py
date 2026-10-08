#!/usr/bin/env python3
"""D0 diagnostic (NOT a campaign run): render one synthetic frame through the
patched D0 worktree at sh_degree 0 and sh_degree 3, and report the D0 metadata
semantics of each branch (f9 flag, tiles/depths/means2d presence, sumcheck,
means2d pixel-space evidence).

No campaign RNG is touched (fresh process, torch.manual_seed(0) for the probe
tensors only). GPU: honors CUDA_VISIBLE_DEVICES of the caller.
"""
import importlib.util
import json
import os
import sys

os.environ["HIGS_PX_RUNTIME"] = "2"
os.environ["HIGS_BWD_SCALAR_ADJOINT"] = "scalar_adjoint"
os.environ["HIGS_BWD_H8_MR"] = "1"
os.environ["HIGS_BWD_ABSGRAD"] = "1"
os.environ.pop("HIGS_DISABLE_F9", None)

REPO = "/root/3dgs-renderer-benchmark"
WT = "/root/epic05_d0/c0_worktree_d0"
SCENE_CUDA_DIR = "/root/epic05_build_cache/scene/gsplat_scene_cuda"
C0_SO = ("/root/epic05_build_cache/c0_ext/"
         "experimental_gaussian_render_inference_scene_cuda/"
         "experimental_gaussian_render_inference_scene_cuda.so")
CORE_SO = "/root/epic05_build_cache/core/gsplat_cuda/gsplat_cuda.so"

sys.path.insert(0, WT)
sys.path.insert(0, SCENE_CUDA_DIR)
sys.path.insert(0, os.path.join(REPO, "scripts", "epic05", "phase7"))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "baseline", "reference_v1"))

import torch  # noqa: E402  (must precede the .so loads: libc10)


def load_ext(name, so_path):
    spec = importlib.util.spec_from_file_location(name, so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[name] = mod
    return mod


core = load_ext("gsplat_cuda", CORE_SO)
sys.modules["gsplat.csrc"] = core
exp = load_ext("experimental_gaussian_render_inference_scene_cuda", C0_SO)
sys.modules["gsplat.experimental.render.kernels.csrc"] = exp
sys.modules["experimental_gaussian_render_inference_scene_cuda"] = exp

from gsplat.experimental import rasterize_gaussian_higs_dynamic  # noqa: E402

torch.manual_seed(0)
dev = "cuda"
N = 4096
means = (torch.rand(N, 3, device=dev) - 0.5) * 2.0
quats = torch.nn.functional.normalize(torch.randn(N, 4, device=dev), dim=-1)
scales = torch.full((N, 3), -2.0, device=dev)
opacities = torch.full((N,), 0.5, device=dev)
viewmat = torch.eye(4, device=dev)[None, None]
K = torch.tensor([[243.0, 0.0, 256.0], [0.0, 243.0, 256.0], [0.0, 0.0, 1.0]],
                 device=dev)[None, None]

print(f"torch {torch.__version__} cuda {torch.version.cuda} dev "
      f"{torch.cuda.get_device_name(0)}", flush=True)

for deg, cch in ((0, 1), (2, 9), (3, 16)):
    colors = torch.randn(N, cch, 3, device=dev) * 0.1
    out = rasterize_gaussian_higs_dynamic(
        means, quats, scales, opacities, colors,
        sh_degree=deg,
        viewmats=viewmat, Ks=K, width=512, height=512,
        backward_mode="higs_native", enable_culling=True,
        camera_model="pinhole", render_mode="RGB", eps2d=0.3)
    md = out.get("metadata", {})
    info = out["densification_info"]
    tiles = md.get("d0_tiles_per_gauss")
    m2d = md.get("d0_means2d_px")
    depths = md.get("d0_depths_vis")
    vis = info["visible_gaussian_ids"]
    rep = {
        "sh_degree": deg,
        "colors_shape": list(colors.shape),
        "f9_enabled": md.get("d0_f9_enabled"),
        "n_isects": md.get("n_isects"),
        "n_isects_full": md.get("n_isects_full"),
        "n_visible_meta": md.get("n_visible"),
        "visible_ids_numel": int(vis.numel()),
        "tiles_present": tiles is not None,
        "tiles_numel": None if tiles is None else int(tiles.numel()),
        "tiles_sum": None if tiles is None
        else int(tiles.reshape(-1).sum().item()),
        "m2d_present": m2d is not None,
        "m2d_shape": None if m2d is None else list(m2d.shape),
        "m2d_med_abs": None if m2d is None else
        [float(x) for x in m2d.reshape(-1, 2).abs().median(dim=0).values],
        "depths_present": depths is not None,
        "depths_numel": None if depths is None else int(depths.numel()),
        "frame_shape": list(out["frame"].shape),
    }
    print(json.dumps(rep, indent=2), flush=True)

print("PROBE_DONE")
