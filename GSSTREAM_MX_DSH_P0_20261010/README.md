# GSSTREAM-P0 — CPU-only falsification kit (2026-10-10)

**Status: HYPOTHESIS ONLY. NOT ORIGINALITY ESTABLISHED. NOT GPU SPEEDUP PROVEN.**

**Objective.** Test whether a depth-stable, scanline-encoded exact support stream could eliminate *expensive per-pixel Gaussian presence checks*, rather than just skip exp/blend after the check. The key research gap is constructing/reusing a cheap stream for both forward and backward, with no explosion in build traffic, global bytes, sorting or gradient access.

**Closest prior art (PRIOR ART COLLISION HIGH).**
- Speedy-Splat / AccuTile CVPR 2025: exact ellipse-to-tile culling, including row/column ellipse intersection calculations: https://speedysplat.github.io/
- FlashGS CVPR 2025: precise Gaussian intersection/culling and pipelining: https://openaccess.thecvf.com/content/CVPR2025/html/Feng_FlashGS_Efficient_3D_Gaussian_Splatting_for_Large-scale_and_High-resolution_Rendering_CVPR_2025_paper.html
- gsplat `rasterize_to_indices_in_range`: per-pixel (Gaussian ID, pixel ID) index generation **already exists**. https://docs.gsplat.studio/versions/1.5.3/apis/rasterization.html
- gsplat main Aug 2026 already has active-tile forward/backward with sparse IDs: https://github.com/nerfstudio-project/gsplat
- 3DGART 2026: tile/primitive-mapped backward on **ray-traced Gaussian rendering**, not the same raster path: https://arxiv.org/abs/2608.17298

Neither using ellipse formulas, creating sparse pixel-hit indices, nor the mere concept of scanline support is claimable as novel. Only a DIFFERENT correct low-overhead **GPU algorithm for building/consuming compressed ordered streams** might become publishable; that step does not yet exist.

## Run locally or on MX (CPU ONLY)

```bash
python3 test_properties.py
python3 cpu_oracle.py --size 40 --num 75 --output out/synthetic_seed42
python3 cpu_oracle.py --size 40 --num 75 --seed 123 --output out/synthetic_seed123
# only if a *preexisting* CPU-exported array exists (never invoke CUDA to create one in P0):
python3 cpu_oracle.py --npz /path/to/existing_projection.npz --output out/frozen_scene
```

Dependencies: Python 3.9+ and NumPy. No torch, CUDA, compiled extensions, or network calls. All code runs synchronously and terminates. Default synthetic smoke is **not a performance measurement**.

Projection NPZ schema: `means2d (N,2), conics (N,3)` with `[a,b,c]` (precision matrix), `opacities (N,), colors (N,3), depths (N,), height, width`. All fp64-compatible; projected screen coordinates must match raster kernel pixel convention (this test uses integer pixel centers). Must record how native alpha cutoff, conic, depth sorting, and early-stop semantics were obtained. If absent, mark real-data gate BLOCKED; do not fake input.

See `DSH_TASK.md` for frozen source, preregistered gates, outputs and stop rules.
