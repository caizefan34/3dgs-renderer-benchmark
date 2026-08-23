# Phase 7 Training Readiness Audit

**Date:** 2026-09-02
**Project:** 3DGS Renderer Benchmark — Phase 7: Full Training Evidence Chain
**Auditor:** Automated pipeline audit

---

## 1. Executive Summary

This repository (`3dgs-renderer-benchmark`) is fundamentally a **renderer benchmark**, not a 3DGS training framework. The existing "training validation" scripts (Phase 4, Phase 5, Phase 2) implement **simplified training loops** that do **not** constitute valid full-training evidence:

| Shortcoming | Impact |
|:------------|:-------|
| Random GT (no real images) | Training trajectory irrelevant to real 3DGS |
| No densification | Gaussian count fixed; training cannot converge |
| No pruning | Dead Gaussians accumulate indefinitely |
| No SH degree scheduling | Color representation never fully unlocked |
| Single fixed camera | No multi-view supervision |
| L2 loss only (no L1 + D-SSIM) | Not 3DGS-standard loss; quality metrics incomparable |

**Assessment:** Full 3DGS training with real GT = **MISSING**. A proper training pipeline must be built from scratch.

---

## 2. Repository Architecture Overview

```
3dgs-renderer-benchmark/
├── src/
│   ├── renderers/         ← Renderer adapters (inference only)
│   ├── benchmark_framework/ ← PLY loader, camera loader, metrics
│   ├── datasets/          ← Dataset manifest validation
│   ├── adapters/          ← Strict adapter interface (abstract)
│   └── benchmark/         ← Benchmark difficulty
├── scripts/
│   └── epic05/
│       ├── phase2_experiments.py     ← Synthetic scene training validation
│       ├── phase4_training_validation.py  ← Random-GT training (1000 steps)
│       ├── phase5_training_validation.py  ← Extended random-GT training (5000 steps)
│       └── _check_training_infra.py  ← Training infra check
├── data/
│   ├── official/
│   │   └── mipnerf360/    ← Trained PLY + cameras (room, garden, bicycle)
│   └── datasets/
│       └── mipnerf360/    ← Real GT images (room=311, bicycle=194, garden=250)
├── configs/epic05/
│   ├── optimization_matrix.json    ← Performance optimization config
│   └── official_validation_matrix.json ← Real-scene quality config
└── results/epic05/
    ├── official/quality/  ← GT quality results (tile16 == tile32 confirmed)
    ├── gradient/          ← Gradient correctness results
    └── final_validation/training/ ← Old simplified training results (50K synth)
```

---

## 3. Component-by-Component Audit

### 3.1 Dataset Loader

| Property | Value |
|:---------|:------|
| **Exists?** | PARTIAL |
| **Real GT?** | YES — Mip-NeRF 360 images on disk (data/datasets/mipnerf360) |
| **Differentiable?** | N/A (data loading is not a differentiable operation) |
| **Used in training?** | NO — existing scripts use random GT or renderer-generated GT |
| **Camera-to-image mapping?** | VERIFIED — room (311/311), bicycle (194/194), garden (250/250) all match |
| **Status** | **BLOCKED** (loader exists for PLY/camera JSON; GT loader exists in validate_quality.py but NOT integrated into any training script) |

### 3.2 Camera System

| Property | Value |
|:---------|:------|
| **Exists?** | YES |
| **Real GT?** | YES — official cameras.json with image_name mapping |
| **Differentiable?** | No (poses are fixed inputs) |
| **Used in training?** | PARTIAL — used in existing scripts but only first camera |
| **Resize support?** | YES — `resize_cameras()` in benchmark_framework/cameras.py |
| **Status** | **SUPPORTED** (reuse existing camera infrastructure) |

### 3.3 Gaussian Model

| Property | Value |
|:---------|:------|
| **Exists?** | PARTIAL — `SimpleGaussianModel` defined in phase4/phase5 scripts but not as reusable module |
| **Real GT?** | N/A |
| **Differentiable?** | YES — all params are nn.Parameter |
| **Used in training?** | PARTIAL — scripts create fresh randn models (not initialized from SfM) |
| **Initialization from PLY?** | NO — scripts ignore real point cloud positions |
| **Trainable params?** | xyz, rotations, scales, opacity, shs (5 groups) |
| **Status** | **PARTIAL** (model class exists but is embedded in validation scripts; no SfM initialization) |

### 3.4 Renderer

| Property | Value |
|:---------|:------|
| **Exists?** | YES |
| **Real GT?** | N/A |
| **Differentiable?** | YES — `gsplat.rasterization` has full forward + backward CUDA kernels |
| **Used in training?** | YES — used in all existing training scripts |
| **Backend** | `nerfstudio-project/gsplat` (not original Inria code) |
| **Differentiable params** | means, quats, scales, opacities, colors (SH) — all 5 groups pass gradient |
| **Tile size control** | Runtime parameter (not compile-time) |
| **Status** | **SUPPORTED** (gsplat is fully differentiable; verified in Phase 6+) |

### 3.5 Loss Function

| Property | Value |
|:---------|:------|
| **Exists?** | PARTIAL |
| **Real GT?** | N/A |
| **Differentiable?** | YES |
| **Used in training?** | PARTIAL — MSE/L2 only |
| **3DGS-standard loss (L1 + D-SSIM)?** | **MISSING** — no D-SSIM loss implemented |
| **Loss weight config?** | **MISSING** |
| **Status** | **PARTIAL** (L2 exists; D-SSIM loss needs implementation) |

### 3.6 Optimizer

| Property | Value |
|:---------|:------|
| **Exists?** | YES |
| **Real GT?** | N/A |
| **Differentiable?** | N/A |
| **Used in training?** | YES — torch.optim.Adam |
| **Param-specific LR?** | **MISSING** — single LR for all param groups (3DGS uses different LRs per group) |
| **Spatial LR (xyz)?** | **MISSING** — no spatial learning rate scaling |
| **Status** | **PARTIAL** (Adam available but no param-group LR scheduling) |

### 3.7 Densification

| Property | Value |
|:---------|:------|
| **Exists?** | **MISSING** — NO densification implemented in any script |
| **Real GT?** | N/A |
| **Differentiable?** | Densification is a topological operation (not differentiable) |
| **Used in training?** | **NO** |
| **Clone/split logic?** | **MISSING** |
| **Gradient threshold?** | **MISSING** |
| **Densification interval?** | **MISSING** |
| **Max Gaussians?** | **MISSING** |
| **Status** | **MISSING** (critical for full 3DGS training) |

### 3.8 Pruning

| Property | Value |
|:---------|:------|
| **Exists?** | **MISSING** |
| **Used in training?** | **NO** |
| **Opacity-based?** | **MISSING** |
| **Reset opacity?** | **MISSING** |
| **Pruning interval?** | **MISSING** |
| **Status** | **MISSING** (critical for full 3DGS training) |

### 3.9 SH Degree Scheduling

| Property | Value |
|:---------|:------|
| **Exists?** | **MISSING** |
| **Used in training?** | **NO** |
| **Progressive SH?** | **MISSING** (3DGS starts at SH degree 0 and increases) |
| **Status** | **MISSING** |

### 3.10 Checkpoint

| Property | Value |
|:---------|:------|
| **Exists?** | **MISSING** |
| **Used in training?** | **NO** |
| **Model state?** | **MISSING** |
| **Optimizer state?** | **MISSING** |
| **Iteration?** | **MISSING** |
| **Gaussian count?** | Tied to model state |
| **Status** | **MISSING** |

---

## 4. Training Dependency Graph

```
Dataset Loader (Mip-NeRF 360) ──→ GT Image + Camera Param
                                          │
Gaussian Model (5 params) ──→ Renderer (gsplat) ──→ Rendered Image
                                          │
                                          ├──→ Loss (L1 + D-SSIM) ──→ backward()
                                          │
                                          ├──→ Optimizer (Adam, param-group LR) ──→ step()
                                          │
                                          ├──→ Densification (grad-threshold clone/split)
                                          │
                                          ├──→ Pruning (opacity threshold)
                                          │
                                          ├──→ SH Degree Schedule (0→1→3)
                                          │
                                          └──→ Checkpoint (model + optimizer state)
```

**Colors:**
- 🟢 Currently SUPPORTED: Renderer, Camera, Optimizer (basic)
- 🟡 PARTIAL: Dataset loader (GT loader exists in validate_quality.py), Gaussian model (no SfM init), Loss (no D-SSIM)
- 🔴 MISSING: Densification, Pruning, SH scheduling, Checkpoint, Param-group LR

---

## 5. Training Readiness Summary

| Component | Exists | Real GT | Differentiable | Used in training | Status |
|:----------|:------:|:-------:|:--------------:|:----------------:|:------:|
| Dataset | PARTIAL | ✅ | N/A | ❌ | **BLOCKED** |
| Camera | ✅ | ✅ | ❌ | PARTIAL | **SUPPORTED** |
| Gaussian model | PARTIAL | N/A | ✅ | PARTIAL | **PARTIAL** |
| Renderer | ✅ | N/A | ✅ | ✅ | **SUPPORTED** |
| Loss (L1) | PARTIAL | N/A | ✅ | PARTIAL | **PARTIAL** |
| Loss (D-SSIM) | ❌ | N/A | ✅ | ❌ | **MISSING** |
| Optimizer (Adam) | ✅ | N/A | N/A | ✅ | **SUPPORTED** |
| Param-group LR | ❌ | N/A | N/A | ❌ | **MISSING** |
| Densification | ❌ | N/A | N/A | ❌ | **MISSING** |
| Pruning | ❌ | N/A | N/A | ❌ | **MISSING** |
| SH scheduling | ❌ | N/A | N/A | ❌ | **MISSING** |
| Checkpoint | ❌ | N/A | N/A | ❌ | **MISSING** |

**Overall Assessment:** Full 3DGS training pipeline is **NOT READY**. Critical components (densification, pruning, SH scheduling, proper loss, checkpoint) are completely missing. Estimated implementation effort: ~400-600 lines of modular Python code.

---

## 6. Current State Transition

### What exists (can be reused)
- `gsplat.rasterization` — fully differentiable renderer with tile_size control
- `load_ply()` — PLY checkpoint loader (provides SfM initialization for first training)
- `Camera` dataclass — camera parameters
- `load_cameras_from_json()` — camera loader with image_name mapping
- `resize_cameras()` — resolution control
- GT images on disk with verified camera-to-image mapping

### What needs to be built
- `GaussianModel` — proper 3DGS parameter container with:
  - SfM initialization (from PLY point cloud)
  - Parameter grouping for per-group LR
  - Activation functions (sigmoid for opacity, exp for scales, normalize for quats)
  - SH degree resizing
  - Densification (clone/split logic)
  - Pruning (opacity-based removal)
  - Prune-reset (reset opacity after pruning)
- `TrainingLoss` — L1 + D-SSIM combined loss
- `TrainingPipeline` — orchestrator with:
  - Multi-camera iteration
  - Densification/pruning scheduling
  - SH degree progression
  - Checkpointing
  - Metrics logging

---

## 7. Training Script Architecture (Proposed)

```
scripts/epic05/phase7/
├── __init__.py
├── train_3dgs.py           ← Main training entry point
├── gaussian_model.py       ← GaussianModel with densification/pruning
├── loss.py                 ← L1 + D-SSIM loss
├── dataset.py              ← Real GT dataset loader
├── config.py               ← Training configuration
├── metrics.py              ← Training metrics tracking
├── checkpoint.py           ← Checkpoint save/load
├── run_sanity.py           ← Short sanity run (configurable steps)
├── run_full.py             ← Full training run
└── run_comparison.py       ← Baseline vs candidate comparator
```

---

## 8. Blockers for Phase 7 Start

| Blocker | Impact | Resolution |
|:--------|:-------|:-----------|
| 🔴 Server unreachable | Cannot run on A100 (EPIC-05) | Use local RTX 5070 Laptop for initial development and training |
| 🔴 Densification/pruning missing | Training will not converge properly | Must implement (estimated ~250 lines) |
| 🔴 D-SSIM loss missing | Not comparable to 3DGS-standard training | Must implement (~30 lines) |
| 🟡 Param-group LR missing | Suboptimal training dynamics | Must implement (~20 lines) |
| 🟡 Checkpoint missing | Risk of losing all progress | Must implement (~40 lines) |

**Decision:** Proceed with local RTX 5070 Laptop as primary training hardware. A100 experiments deferred until server connectivity restored. All training code will be designed to be server-agnostic (same code runs on both GPUs).

---

## 9. Immediate Actions

1. ✅ Audit complete
2. ⬜ Create `scripts/epic05/phase7/` directory with modular training infrastructure
3. ⬜ Implement GaussianModel (SfM init, densification, pruning, SH scheduling)
4. ⬜ Implement L1 + D-SSIM loss
5. ⬜ Implement real GT dataset loader
6. ⬜ Implement checkpointing
7. ⬜ Run **short sanity check** (500 steps, room scene, real GT, tile16 only)
8. ⬜ If sanity passes → 3000-step mid training (room, baseline, tile16, tile32)
9. ⬜ If mid passes → Full 30K-step training
10. ⬜ Extend to bicycle, garden

---

*This audit will be updated as components are implemented and validated.*
