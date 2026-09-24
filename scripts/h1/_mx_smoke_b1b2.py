"""H1 smoke test: confirm B1 (rasterization) and B2 (rasterize_gaussian_higs_frozen)
both render end-to-end in ONE env/process from the higs tree, same torch.
Run: conda activate higs-13scene-env; PYTHONNOUSERSITE=1 PYTHONPATH=<higs-tree> CUDA_VISIBLE_DEVICES=0 python this.py
"""
import os, sys, time, inspect
import torch

print("gsplat_path_target=higs-tree")
import gsplat
print("gsplat", os.path.dirname(gsplat.__file__), "ver", getattr(gsplat, "__version__", "?"))
import gsplat.experimental as ex
print("experimental", os.path.dirname(ex.__file__))
print("torch", torch.__version__, "cuda", torch.version.cuda, "cxx11_abi", torch._C._GLIBCXX_USE_CXX11_ABI)
dev = "cuda"
print("gpu", torch.cuda.get_device_name(0))

from gsplat.rendering import rasterization
from gsplat.experimental import rasterize_gaussian_higs_frozen
from gsplat.experimental.render.functional.gaussian_inference import (
    create_higs_renderer, _HIGS_FROZEN_TRACKER,
)

torch.manual_seed(0)
N = 30000
K_SH = 16
width, height = 1024, 768
# --- synthetic Gaussians in front of camera ---
means = (torch.rand(N, 3, device=dev) - 0.5) * 2.0
means[:, 2] = means[:, 2] * 0.5 + 4.0  # depth ~3.5..4.5
quats = torch.zeros(N, 4, device=dev); quats[:, 0] = 1.0  # identity rotation
scales = torch.rand(N, 3, device=dev) * 0.3 + 0.05  # raw scales (NOT exp; gsplat expects raw)
opacities = torch.rand(N, device=dev) * 0.5 + 0.5
colors = torch.rand(N, K_SH, 3, device=dev) * 0.1  # SH coeffs (small)

# camera: look at origin from z=0
vm = torch.eye(4, device=dev).view(1, 1, 4, 4).clone()
K = torch.tensor([[800., 0, (width - 1) / 2], [0, 800., (height - 1) / 2], [0, 0, 1]], device=dev).view(1, 1, 3, 3)

def psnr(a, b):
    mse = (a - b).pow(2).mean().item()
    return 60.0 if mse < 1e-12 else -10.0 * torch.log10(torch.tensor(mse)).item()

# ---- B1: standard rasterization ----
print("\n=== B1 rasterization ===")
t0 = time.time()
with torch.no_grad():
    out = rasterization(
        means=means.unsqueeze(0), quats=quats.unsqueeze(0),
        scales=scales.unsqueeze(0), opacities=opacities.unsqueeze(0), colors=colors,
        viewmats=vm, Ks=K, width=width, height=height,
        sh_degree=3, packed=True, radius_clip=0.0,
    )
torch.cuda.synchronize()
print("B1 elapsed", round((time.time() - t0) * 1000, 2), "ms")
r_b1 = out[0]; a_b1 = out[1]
print("B1 render shape", tuple(r_b1.shape), "alpha shape", tuple(a_b1.shape),
      "finite", bool(torch.isfinite(r_b1).all()), "range", round(r_b1.min().item(), 4), round(r_b1.max().item(), 4))

# ---- B2: HiGS frozen native ----
print("\n=== B2 rasterize_gaussian_higs_frozen ===")
_HIGS_FROZEN_TRACKER.reset()
handle = create_higs_renderer(means, quats, scales, opacities, colors, sh_degree=3)
t0 = time.time()
with torch.no_grad():
    res = rasterize_gaussian_higs_frozen(
        means, quats, scales, opacities, colors,
        backward_mode="higs_native", scene=handle, freeze_topology=True,
        viewmats=vm, Ks=K, width=width, height=height, sh_degree=3,
        use_higs_culling=True, radius_clip=0.0, tile_sampling_ratio=1.0,
    )
torch.cuda.synchronize()
print("B2 elapsed", round((time.time() - t0) * 1000, 2), "ms")
r_b2 = res["frame"]; a_b2 = res["alpha"]
print("B2 render shape", tuple(r_b2.shape), "alpha shape", tuple(a_b2.shape),
      "finite", bool(torch.isfinite(r_b2).all()), "range", round(r_b2.min().item(), 4), round(r_b2.max().item(), 4))
print("B2 metadata keys", list(res.get("metadata", {}).keys())[:20] if isinstance(res.get("metadata"), dict) else type(res.get("metadata")))

# ---- compare ----
print("\n=== compare ===")
r1 = r_b1.reshape(-1, 3); r2 = r_b2.reshape(-1, 3)
if r1.shape == r2.shape:
    print("render_shape_match True")
    print("PSNR(B1,B2)", round(psnr(r_b1.float(), r_b2.float()), 3), "dB")
    print("max_abs", round((r_b1 - r_b2).abs().max().item(), 5), "mean_abs", round((r_b1 - r_b2).abs().mean().item(), 5))
else:
    print("render_shape_match False", tuple(r1.shape), tuple(r2.shape))
handle.release()
print("\n=== SMOKE_DONE ===")
