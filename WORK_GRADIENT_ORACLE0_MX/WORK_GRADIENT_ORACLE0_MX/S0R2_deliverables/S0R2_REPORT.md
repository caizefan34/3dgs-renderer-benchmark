# S0R2 Report 鈥?WORK-GRADIENT-ORACLE0-MX

## Objective
Resolve `BLOCKED_BINARY_MISMATCH`; restore an execution environment matching the HiGS sparse-tile training interface (`gsplat.experimental.rasterize_gaussian_higs_dynamic` with native `tile_mask` sparse execution). **S0R2 scope: software-stack recovery + smoke test only.** No S1A/S1B formal experiments, no 3K/30K training.

## Headline result
**The HiGS sparse-tile kernel is now operational and ABI-matched.** The prior blocker (gsplat 1.5.3 in `nerficg` lacking `experimental.HiGS`) is resolved by switching to the source-matched `higs-13scene-env` + source commit `77ab983`, reusing the source-built `gsplat_scene_cuda.so`. Real sparse execution is proven: selecting 25% of tiles produces a **75.00% reduction in CUDA intersections** (95808 鈫?23952 synthetic; 180628 鈫?46429 real), confirming native `isect_tiles(..., tile_mask=...)` work, **not** dense-render-then-loss-mask.

## Environment resolution
| Item | Value |
|---|---|
| Python env | `/mnt/storage_pool/liaoyuanjun/higs-13scene-env` (3.10.21) |
| PyTorch | 2.9.1+cu128 |
| CUDA runtime / nvcc | 12.8 |
| GPU | A100-PCIE-40GB (sm_80) |
| gsplat source | commit `77ab983ffe43420b2131669cb35776b883ca4c3c` (detached) |
| gsplat_scene_cuda.so | `.../cuda-build/gsplat-higs-mx-py310/gsplat_scene_cuda/gsplat_scene_cuda.so` |
| .so SHA256 | `e4b0bc5bf358ec43875a3b7e0d84591763968021b7c7eabcc6d60572490627c8` |
| isolation | `PYTHONNOUSERSITE=1` (drops stale `~/.local` overlay); build-dir on `sys.path` to reuse prebuilt ext; `TMPDIR`/`TORCH_EXTENSIONS_DIR` on storage pool |

**Key insight:** a stale `~/.local/lib/python3.10/site-packages/gsplat` (older gsplat 1.5.3 missing `_maybe_evaluate_sh` and `tile_mask` `isect_tiles`) was shadowing the source. `PYTHONNOUSERSITE=1` makes the entire `gsplat` package resolve consistently to the source tree, and the prebuilt `gsplat_scene_cuda.so` loads into the exact env it was built in 鈥?no rebuild needed (root partition is 100% full).

## Gate results
- **A** `gsplat.experimental` import 鈥?**PASS**
- **B** `rasterize_gaussian_higs_dynamic` exists with `tile_mask` / `tile_sampling_ratio` 鈥?**PASS**
- **C** module/CUDA/experimental identity traceable, all 鈫?commit `77ab983` 鈥?**PASS**
- **D** full/sparse fwd+bwd smoke: finite frames, correct grad shapes, all param groups get finite nonzero gradients 鈥?**PASS** (real bicycle ckpt: means/quats/scales/opac/colors all backprop)
- **E** tile-mask change 鈫?expected isect reduction (74鈥?5%) 鈫?genuine native sparse execution 鈥?**PASS**
- **F** HT full-vs-sampled gradient-consistency 鈥?**DEFERRED to S1A** (out of S0R2 scope; requires the S1A sampling harness + real scene)
- **G** 6 FasterGSFused checkpoints 鈥?**loadable & differentiable**, but oracle-comparability **NOT established** (see caveats)

## Caveats (must carry into S1A)
1. **PCA color space.** `training_config.yaml` has `APPLY_PCA=true`. Checkpoint SH/DC may be in a PCA-rotated basis, not raw RGB. Checkpoints are structurally loadable and differentiable through the sparse kernel, but are **not automatically a math-comparable Oracle state** until the PCA inverse / FasterGS color convention is resolved.
2. **Garden camera data incomplete.** `garden` has no `eval_cameras.json`; its `sparse/0` COLMAP files are stubs (127/133 bytes). `bicycle` has full `eval_cameras.json` + images and was used to demonstrate the identical code path. Gate D nominally specifies garden 鈥?a **data-availability gap**, not a binary mismatch.
3. **Oracle appearance split** (`RENDER_DIFFUSE/SPECULAR=true`) may require convention audit before pixel-level Oracle use.

## Verdict
**READY_FOR_S1A** (sparse-tile HiGS interface restored, ABI-matched, native sparse execution verified with real checkpoints). Oracle-state mathematical comparability and the HT gradient-consistency test (Gate F) remain S1A responsibilities, with the PCA/camera caveats above.

## No-destructive compliance
No files deleted/overwritten/moved; no checkpoints/datasets/runs modified; no other processes killed; `nerficg` env and dirty worktrees untouched; no software installed to the full root partition; isolated `TMPDIR`/extension cache on storage pool only.
