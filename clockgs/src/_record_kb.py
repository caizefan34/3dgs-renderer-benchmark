import os, json, datetime
# Append a knowledge entry to the D:\学习 knowledge base per AGENTS.md
kb = r"D:\学习"
entry_dir = os.path.join(kb, "3dgs-renderer-benchmark")
os.makedirs(entry_dir, exist_ok=True)
entry = """# 3dgs-renderer-benchmark — ClockGS lane infra

## 2026-10-03 — ClockGS C0 harness infra (knowledge)

- ENV: Windows, RTX 5070 Laptop 8GB, torch 2.13.0+cu130, CUDA 13.0, gsplat 1.4.0 (CUDA backend NOT built — _C missing, JIT fallback fails: no MSVC cl on PATH), Python 3.13.
- CANONICAL RASTERIZER THAT WORKS: Inria diff_gaussian_rasterization (prebuilt _C.cp313-win_amd64.pyd). forward returns 4-tuple (color, radii, depth, alpha). GaussianRasterizationSettings has NO antialiasing field (older build). Use means3D/means2D/shs/opacities/scales/rotations kwargs.
- BROKEN DEPS in .c0-official-reference: simple_knn (DLL load failed) and fused_ssim (module missing). opencv-python-headless + plyfile installed; scipy 1.18 for KNN.
- FIXES: (1) shim simple_knn._C.distCUDA2 via scipy cKDTree k=4 mean squared dist to 3 NN; (2) pure-torch ssim fallback in utils.loss_utils already exists; (3) local render() in clockgs/src/core.py matches installed rasterizer.
- CLOCK MECHANISM: PyTorch Adam skips params whose .grad is None. Setting geometry (_xyz,_scaling,_rotation).grad=None on skipped iters freezes geometry optimizer state exactly (no momentum drift). Full backward still computed every iter (C0 is a dynamics oracle; skip-cost is C2/C3).
- DENSIFY SCHEDULE: densify_until_iter=15000 (topology stable after 15K). C0 clocks geometry over refinement 15K->30K so geometry is provably still moving (avoids mature-checkpoint trap / LAW-D0 / LAW-MATURE-HELDOUT).
- THROUGHPUT: room images_8 (389x259) ~68 it/s, 12-13 ms/iter.
"""
with open(os.path.join(entry_dir, "CLOCKGS_INFRA.md"), "a", encoding="utf-8") as f:
    f.write(entry)
print("recorded knowledge to", os.path.join(entry_dir, "CLOCKGS_INFRA.md"))
