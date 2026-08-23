# Phase 8 — Track A: M1 Cross-Scene Full Training Validation

**Date:** 2026-09-15  
**Status:** BLOCKED — Server EPIC-05 unreachable

---

## 1. Current Evidence

### Room 30K (COMPLETED)

| Metric | tile16 | tile32 | Ratio |
|:-------|:------:|:------:|:-----:|
| Wall time | 150.3 min | 94.8 min | **1.58×** |
| Best PSNR | 29.27 dB | 29.39 dB | Δ=+0.12 dB |
| Final Gaussians | 1,193,480 | 1,146,273 | -4.0% |
| Peak VRAM | ~2.0 GB | ~1.8 GB | -10% |
| NaN/Inf | None | None | — |
| Exit code | 0 | 0 | — |

### Room 3000-step Anomaly (RESOLVED — run variance)

- Mid-run tile32: 446ms/iter (2.38× slower than full-run tile32 at same config)
- Full-run (v2) tile32: 188ms/iter
- Within the v2 full-run, tile32 is faster at **every** iteration
- **Conclusion:** The 3000-vs-30K "discrepancy" is run-to-run variance, not training-stage dependence

### Remaining: Controlled single-session rerun — BLOCKED by server

---

## 2. Planned Experiments

### Bicycle 30K

**Scene characteristics:** ~6.1M initial Gaussians, large outdoor scene, wide variance in Gaussian scales  
**Hypothesis:** If tile32's advantage is scene-independent, bicycle should show similar speedup (~1.5×). If the advantage depends on specific workload characteristics, bicycle may show a different ratio.

**Config (same as room — controlled):**

| Parameter | Value |
|:----------|:------|
| Initialization | Same SfM checkpoint |
| Camera sequence | Same order |
| Seed | 42 |
| Optimizer | Adam (same LR schedule) |
| Loss | L1 + D-SSIM (λ=0.2) |
| SH degree | 0→3 progressive (1000-step interval) |
| Densification | Same policy (grad threshold 2e-4, start 500, end 15000) |
| Pruning | Same policy (opacity 0.005, interval 100) |
| Steps | 30,000 |
| Resolution | 1080p |

**Metrics:**
- Total wall time
- Iter/s (mean, median)
- fwd_ms, bwd_ms, opt_ms, topology_ms (per-iteration breakdown)
- Best PSNR
- SSIM (if pipeline supports)
- LPIPS (if stable)
- Final Gaussian count
- Gaussian count trajectory (all 30K)
- Densification events
- Pruning events
- Peak VRAM
- NaN/Inf
- Exit code

### Garden 30K

**Scene characteristics:** ~5.8M initial Gaussians, dense vegetation, high-frequency detail  
**Config:** Identical to bicycle protocol above.

### Controlled 3000-step Rerun

**Purpose:** Definitively resolve the 3000-step anomaly.

**Protocol:**
- Single process/session
- Run tile16 first, then tile32 (or vice versa — randomize)
- Same seed
- Same initialization
- Same camera sequence
- Warmup iterations before timing
- Record per-iteration timing for 3000 steps
- Compare iteration-by-iteration

---

## 3. Cross-Scene Status

| Scene | Initial Gs | tile16 30K | tile32 30K | Comparison |
|:------|:----------:|:----------:|:----------:|:----------|
| room | ~1.6M | ✅ COMPLETED | ✅ COMPLETED | tile32 **1.58× faster**, PSNR equivalent |
| bicycle | ~6.1M | ❌ BLOCKED | ❌ BLOCKED | Server unreachable |
| garden | ~5.8M | ❌ BLOCKED | ❌ BLOCKED | Server unreachable |

---

## 4. Ready-to-Run Commands

When EPIC-05 recovers:

```bash
# Bicycle tile16 30K
cd /path/to/3dgs-renderer-benchmark
CUDA_VISIBLE_DEVICES=0 python scripts/epic05/phase7/run_full.py \
    --scene bicycle --tile-size 16 --steps 30000 \
    --label cross_scene_bicycle_t16

# Bicycle tile32 30K
CUDA_VISIBLE_DEVICES=0 python scripts/epic05/phase7/run_full.py \
    --scene bicycle --tile-size 32 --steps 30000 \
    --label cross_scene_bicycle_t32

# Garden tile16 30K
CUDA_VISIBLE_DEVICES=0 python scripts/epic05/phase7/run_full.py \
    --scene garden --tile-size 16 --steps 30000 \
    --label cross_scene_garden_t16

# Garden tile32 30K
CUDA_VISIBLE_DEVICES=0 python scripts/epic05/phase7/run_full.py \
    --scene garden --tile-size 32 --steps 30000 \
    --label cross_scene_garden_t32

# Controlled 3000-step rerun (within single session)
CUDA_VISIBLE_DEVICES=0 python scripts/epic05/phase7/run_full.py \
    --scene room --tile-sizes 16 32 --steps 3000 \
    --label controlled_rerun
```

---

## 5. Expected Output Location

```
results/epic05/phase8/
├── cross_scene_bicycle_t16.json
├── cross_scene_bicycle_t32.json
├── cross_scene_garden_t16.json
├── cross_scene_garden_t32.json
├── controlled_rerun_t16.json
├── controlled_rerun_t32.json
└── cross_scene_summary.json
```
