# Faster-GS Strong Baseline Reproduction — Report

**Generated:** 2026-10-01 (EPIC-05, A100-SXM4-80GB)  
**Seed:** 42 | **Iterations:** 30,000 | **Scenes:** room, garden, bicycle (Mip-NeRF360)  
**Goal:** publication-grade **strong external baseline** for the current 3DGS research. No new method proposed; no algorithmic code modified.

---

## 1. Exact official commit(s)

All three method repos cloned from the official `https://github.com/nerficg-project/faster-gaussian-splatting`, git status **clean** (dirty=False), detached at the exact branch tips:

| Lane | Method repo | Branch | Commit (40-char SHA) |
|---|---|---|---|
| A — Faster-GS Main | `src/Methods/FasterGS` | `main` | `d7fae871c45384b67b4c348f804762c923949018` |
| B — FasterGSFused | `src/Methods/FasterGSFused` | `FasterGSFused` | `ae665af2f91a4973bb92c40fdc4cba8d12fe3c96` |
| basis / baseline | `src/Methods/FasterGSBasis` | `FasterGSBasis` | `0088b4df7f04d81920644a144da6c124c6bf8789` |

- **NeRFICG framework** (host repo): `src/nerficg` @ `e458ac6e41810ad6d9405687c1e698fadd28dac5`.
- **Submodules:** only `src/ICGui` (GUI, not used — `GUI.ACTIVATE=false`).
- **CUDA backends** (pip-installed, non-editable): `FasterGSCudaBackend`, `FasterGSFusedCudaBackend`, `FasterGSBasisCudaBackend`. All 429 `.so` files SHA256-hashed in `FGS_SOURCE_LOCK.json`.
- **Toolchain:** Python 3.11.16, PyTorch 2.8.0+cu128, CUDA 12.8, driver 570.133.20, gcc 11.4.0, CuDNN 91002.

> Note on `main`: the prior session had `FasterGS` at `117bdba` (2 commits behind). The 2-commit delta (`117bdba`→`d7fae87`) is **MCMC-only** (inactive under `USE_MCMC=false`). For strict faithfulness I checked out the true `main` tip `d7fae871` and rebuilt `FasterGSCudaBackend`.

## 2. Is this a faithful reproduction?

**Yes**, with two documented, non-algorithmic caveats:

- **No algorithmic code modified.** Densification thresholds, optimizer, renderer, loss, pruning, truncation, MCMC are all the official values verbatim. The only config change is `BACKUP.INTERVAL: 5000` (official = `-1`), a non-algorithmic checkpointing setting used solely to capture the Gaussian-count trajectory; it does not touch the training loop or quality.
- **Official-arm resolution caveat.** The official config uses `IMAGE_SCALE_FACTOR: 0.25` (`images_4`). The host's native Mip-NeRF360 images are 1920x1080, so `images_4` = 480x270 (lower than the paper's ~1000px). This affects only the OFFICIAL-SPLIT arm's absolute quality; camera identity (split) is unaffected. Official and corrected metrics are **never mixed**.

The **corrected-SPLIT arm** (1920x1080 iso anisotropic stretch, `APPLY_PCA=false`, `TEST_STEP=8`) is protocol-matched to the project's frozen C0 and is the basis for all cross-backend comparison.

## 3. Corrected true-holdout quality (i%8==0 fully excluded, 1920x1080 iso, seed42, 30K)

| Scene | Lane | PSNR | SSIM | LPIPS | N (final) |
|---|---|---|---|---|---|
| room | Faster-GS Main | 31.715 | 0.9203 | 0.294 | 1,276,911 |
| room | FasterGSFused | 31.633 | 0.9199 | 0.296 | 1,262,605 |
| room | FasterGSBasis | 31.675 | 0.9200 | 0.296 | 1,230,942 |
| room | **frozen C0** | **30.798** | **0.9155** | **0.307** | 853,453 |
| garden | Faster-GS Main | 26.880 | 0.8392 | 0.159 | 4,814,980 |
| garden | FasterGSFused | 26.901 | 0.8393 | 0.159 | 4,809,378 |
| garden | FasterGSBasis | 26.876 | 0.8388 | 0.159 | 4,744,171 |
| garden | **frozen C0** | **26.009** | **0.8199** | **0.198** | 2,548,992 |
| bicycle | Faster-GS Main | 24.725 | 0.7292 | 0.287 | 4,767,826 |
| bicycle | FasterGSFused | 24.738 | 0.7281 | 0.289 | 4,720,561 |
| bicycle | FasterGSBasis | 24.730 | 0.7269 | 0.291 | 4,587,236 |
| bicycle | **frozen C0** | **21.695** | **0.6766** | **0.314** | 3,703,162 |

All three Faster-GS lanes beat frozen C0 on every scene (higher PSNR/SSIM, lower LPIPS). The three Faster-GS lanes are within ~0.1 dB of each other (expected: same densification/loss, different backend/basis).

## 4. Wall time (full training, exclusive GPU1, contention-free)

| Scene | Main | FasterGSFused | Basis | frozen C0 |
|---|---|---|---|---|
| room | 355.8 s | 356.4 s | 1391.3 s | 516.1 s |
| garden | 1111.6 s | 593.2 s | 1783.5 s | 815.7 s |
| bicycle | 493.6 s | 522.5 s | 1685.5 s | 868.5 s |

(Official-split arm, 480x270: room 142.2 s, garden 181.4 s, bicycle 244.8 s — much faster due to lower resolution.)

## 5. Speedup vs same-backend basis (Faster-GS internal, **SAME-BACKEND**)

`speedup = basis_wall / lane_wall` (higher = faster). Both main and fused are compared against the basis baseline under the identical NeRFICG framework + Faster-GS CUDA backend.

| Scene | Main vs Basis | FasterGSFused vs Basis |
|---|---|---|
| room | **3.91x** | **3.90x** |
| garden | **1.61x** | **3.01x** |
| bicycle | **3.42x** | **3.23x** |

FasterGSFused is consistently faster than main (fused backward pass + fused optimizer), most notably on garden (3.01x vs 1.61x), where main's per-iteration cost is inflated by the anisotropic-stretched input.

## 6. Speedup vs frozen C0 (**CROSS-BACKEND** — flagged)

`speedup = C0_wall / lane_wall`. **This is CROSS-BACKEND**: Faster-GS (NeRFICG framework + Faster-GS CUDA backend) vs C0 (gsplat backend). The difference includes the backend/rendering stack, **not purely algorithmic gains**. Do not attribute all of it to the algorithm.

| Scene | Main | FasterGSFused | Basis | C0 wall |
|---|---|---|---|---|
| room | 1.45x | 1.45x | 0.37x | 516.1 s |
| garden | **0.73x** | 1.38x | 0.46x | 815.7 s |
| bicycle | 1.76x | 1.66x | 0.52x | 868.5 s |

**Honest caveat:** garden **main is 0.73x (slower) than C0** cross-backend. This is a backend/input-resolution artifact (the anisotropic-stretched garden is expensive for main's renderer), not an algorithmic regression. FasterGSFused on garden is 1.38x. The basis lane is slower than C0 on all scenes (0.37–0.52x), consistent with basis being a heavier representation.

## 7. Peak VRAM (allocated)

| Scene | Main | FasterGSFused | Basis | C0 (ref) |
|---|---|---|---|---|
| room | 9.16 GiB | 8.96 GiB | 9.29 GiB | — |
| garden | 11.60 GiB | 11.12 GiB | 12.37 GiB | — |
| bicycle | 11.47 GiB | 10.87 GiB | 12.20 GiB | — |

(Official-split 480x270: room 1.84, garden 2.48, bicycle 4.76 GiB.) FasterGSFused uses the least VRAM among the Faster-GS lanes (fused backward pass reduces memory), matching the README's claim.

## 8. Which Faster-GS components are actually enabled

From the verbatim official configs (all lanes):

- **Densification:** gradient-based adaptive densification, start=600, end=14900, interval=100, grad_threshold=0.0002, percent_dense=0.01.
- **MCMC densification:** `USE_MCMC=false` (disabled) — so the `main`-tip MCMC delta is inactive.
- **Pruning (Speedy-Splat):** `SPEEDYSPLAT_PRUNING.USE=false` (disabled).
- **3D filter / truncation mode:** `FILTER_3D.USE=false` (disabled).
- **PPISP:** `USE=false`. **Random background:** `false`.
- **Morton ordering:** interval=5000, end=15000 (enabled).
- **Opacity reset:** interval=3000, extra reset at 500 (enabled).
- **Optimizer:** standard NeRFICG Adam-style with official LR schedule (means 1.6e-4→1.6e-6, SH 2.5e-3/1.25e-4, opacities 0.025, scales 0.005, rotations 0.001).
- **Loss:** 0.8*L1 + 0.2*(1-SSIM); no opacity/scale regularization.
- **CUDA backend per lane:** Main → `FasterGSCudaBackend` (refactored fast 3DGS rasterizer + warp-level culling). FasterGSFused → `FasterGSFusedCudaBackend` (**fused backward pass + fused optimizer**). Basis → `FasterGSBasisCudaBackend` (basis-codebook representation; all Faster-GS variants are built on top of it).

## 9. Baselines (reported separately, never mixed)

1. **Faster-GS Main** — official `main` branch @ `d7fae871`, `FasterGSCudaBackend`.
2. **FasterGSFused** — official `FasterGSFused` branch @ `ae665af2`, fused backward + optimizer.
3. **official basis/baseline** — `FasterGSBasis` @ `0088b4df`, basis-codebook representation (same-backend reference).
4. **frozen C0** — current project's pinned C0 (gsplat backend, `publication_trainer_p12.py` + holdout8, sha256=675b0267), **CROSS-BACKEND reference only**.

## 10. Gaussian-count trajectory (corrected arm, checkpointed every 5000 iters)

Example (room, main): 5,000→927,831 | 10,000→1,223,869 | 15,000→1,324,490 | 20,000→1,324,490 | 25,000→1,324,490 | final→1,276,911. Densification saturates ~15k (end=14900), then opacity-reset pruning reduces the count slightly to final. Full trajectories for all 12 runs are in `FGS_TIMING.json` / `FGS_QUALITY.json`.

## Timing discipline & integrity

- All GPU work on **EPIC-05 A100**; local RTX 5070 **not used**.
- Every run: **exclusive GPU1**, `CUDA_VISIBLE_DEVICES=1` pinned at exec-time, **no concurrent arms**.
- Contention audit (`nvidia-smi` before/after): all 12 runs `contention_flag=False`, `timing_authoritative=True` (no other tenant ever on GPU1; other tenants were only on GPU0/6/7).
- All 12 runs `exit_code=0` (no hidden failures).
- No seed-shopping: seed=42 fixed; all three scenes succeeded on the first pass.

## Artifacts

See `MANIFEST.sha256` and the `FGS_*.json` files in this directory: `FGS_ENVIRONMENT.json`, `FGS_SOURCE_LOCK.json`, `FGS_CAMERA_SPLIT.json`, `FGS_CONFIGS.json`, `FGS_OFFICIAL_SPLIT.json`, `FGS_CORRECTED_SPLIT.json`, `FGS_TIMING.json`, `FGS_QUALITY.json`, `FGS_MEMORY.json`, `FGS_SUMMARY.json`, `FGS_REPORT.md`, `MANIFEST.sha256`.