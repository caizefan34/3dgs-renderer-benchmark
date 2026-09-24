# Candidate C Source Audit Package — PACKAGE_INFO.md

**Package name**: `candidate_c_source_audit_20260914.zip`
**Date**: 2026-09-14
**Topic**: Loss-Aware Pre-Backward Gradient Certificate — external source audit
**Purpose**: Provide external mathematical/source auditors the exact source currently used in the R2 / R2.1 evaluation, and the profile-only instrumentation against which the certificate bounding assumptions can be checked.

---

## 1. Repository Provenance

| Property | Value |
|----------|-------|
| Repository root | `C:\Users\36570\3dgs-renderer-benchmark` |
| Git branch | `master` |
| Git HEAD commit | `84f29bb` — "MD-V2: Stage pre-freeze …" |
| Git tag | `research/mechanism-discovery-v2` |
| Git status | **Clean** (0 modified tracked files); ~871 untracked working files in reports/scripts/results — none affect tracked-source integrity |
| GPU (matching) | NVIDIA A100-PCIE-40GB (remote `mx` server, CUDA 13.1, PyTorch 2.7.1+cu118, CUDA 11.8) |

The canonical baseline source is the **unmodified gsplat 1.5.3** for the locked baseline, and the local pip-installed **gsplat 1.4.0** matching this repository's `baseline/reference_v1/` (annotation labels identify which).

---

## 2. Source Sets Captured

| Source set | Purpose in R2/R2.1 | Where in this package |
|---|---|---|
| **Canonical baseline trainer** | Training loop, loss, densification, optimizer | `canonical_source/baseline/` (from `baseline/reference_v1/`) |
| **gsplat Python Layer** | Autograd boundary for rasterizer | `canonical_source/gsplat_python/` (from pip `gsplat-1.4.0`) |
| **Rasterization Forward Kernel** | Projection + tile accumulation | `canonical_source/raster_forward/` (gsplat-1.4.0 CUDA `rasterize_to_pixels_fwd.cu`) |
| **Rasterization Backward Kernel** | Fused backward pass — all four gradient outputs | `canonical_source/raster_backward/` (gsplat-1.4.0 CUDA `rasterize_to_pixels_bwd.cu`) |
| **SH Computation** | SH forward/backward | `canonical_source/sh/` (gsplat-1.4.0 `spherical_harmonics`) |
| **Projection** | 3D→2D projection forward/backward | `canonical_source/projection/` (gsplat-1.4.0 `fully_fused_projection_*`) |
| **R2 instrumented runner** | R2 measurements | `r2_20260914/` (copy of gsplat-instrumented runner + scripts) — instrumentation preserves canonical semantics |
| **R2.1 profile-only patch** | R2.1 measurements | `r2_1_profile_only/` (patch script + build script + this PATCH_NOTES.md) |
| **C51 historical reference** | Round-2 mechanism evidence for sparse backward gating | `historical_c51_reference/` (minimal proof-of-fail C51 kernel + patch notes) |
| **Audit notes** | Maps matching informational docs | `audit_notes/` |

---

## 3. Version Pin

- **PyTorch** 2.7.1+cu118 (A100 remote `mx`), no local GPU — local repo uses **PyTorch 2.13.0+cu130** with CUDA 13.1 as system capability.
- **gsplat**: locked baseline gsplat **1.5.3**; local pip install gsplat **1.4.0** matches the references in this repo's R2/R1/AUDIT docs. The shared raster kernels are identical between 1.4.0 and 1.5.3 except for minor API signatures — the backward kernel source captured here is the exact file used by the R1/R2 measurements.
- **CUDA** 13.1 (local), 11.8 (remote). SM 80 (A100) / SM 120 (local device = RTX 5070, GB207).

---

## 4. Contents Map

```
candidate_c_source_audit/
├── PACKAGE_INFO.md
├── LOSS_PATH_NOTES.md
├── FORWARD_STATE_INVENTORY.md
├── TRAVERSAL_STATE_NOTES.md
├── RASTER_EQUATION_MAP.md
├── NUMERICAL_BOUNDS.md
├── TENSOR_SHAPES.md
├── canonical_source/
│   ├── baseline/          (reference_v1 trainer, gaussian model, config)
│   ├── gsplat_python/     (wrapper, rendering, __init__)
│   ├── raster_forward/    (rasterize_to_fwd kernel + tile kernel)
│   ├── raster_backward/   (rasterize_to_bwd kernel — R2/R2.1 exact version)
│   ├── sh/                (harmonic kernels)
│   └── projection/        (projection kernels fwd/bwd)
├── r2_20260914/           (R2 instrumented scripts + analysis)
│   ├── backward_timing.py
│   ├── corrected_concentration_runner.py
│   ├── r2_analysis.py
│   └── run_all.sh
├── r2_1_profile_only/
│   ├── r21_patch.py       (string-replacement patch for gsplat-1.5.3)
│   ├── r21_build.py       (gcc-10 wrapper compile + system link)
│   └── PATCH_NOTES.md
├── historical_c51_reference/
│   ├── IntersectTile.c1.cu       (minimal sparse-gating kernel)
│   ├── patch_notes_c51.md
│   ├── patch_cuda.py             (C51 CUDA patching)
│   ├── canonical_training.py
│   ├── measure_recall.py
│   ├── analyzed_results.py
│   ├── simulated_sparse_backward.py
│   └── (kernel patches from round-2 evidence)
├── reports/
│   ├── phase-r2-attribute-decoupled-backward-gate.md
│   └── r2-gradient-dependency-graph.md
└── results/
    └── r2/  (empty - result JSONs were not committed to repo)
```

---

## 5. Repository Integrity Checks

- All copies under `canonical_source/` are byte-for-byte copies from the pinned versions listed above (except where a suffix `_original`/comments indicate the R2.1 profile-only variant in `r2_1_profile_only/`).
- The examined `/tmp/gsplat_baseline/gsplat-1.5.3/gsplat/` tree matches the A100 remote; local installations fine-grained via `pip list` / installed-files metadata are listed in this package.
- The unmodified baseline kernel is included untouched; the R2.1 patch **documented but not applied** in this package (the canonical baseline remains used for training). Applying the profile-only instrumentation is done at build time in `r2_1_profile_only/` — see PATCH_NOTES.md section 6 for details.

---

## 6. Continuation / Citation Notes

- This package is one of the four planned audit packages (A / B / C / D). Package C is solely source-level: it contains no training artifacts, no results, no new methodology, and no summary of previous phases.
- The R1/R2 result JSONs referenced from `results/reference_v1/` are reserved for package B (labeled with explicit "Source of truth" relationships here so auditors can cross-check without embedding content).
- The certificate bounding assumptions are captured in NUMERICAL_BOUNDS.md; no inferences or measurements about those bounds are made in this package (that is the task of audit phase).
