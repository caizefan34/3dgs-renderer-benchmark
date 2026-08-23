# Phase 8 — Track C–F: M2–M5 Training Gate

**Date:** 2026-09-15  
**Status:** BLOCKED — Server EPIC-05 unreachable

---

## 1. Overview

M2–M5 are the gsplat configuration parameters that have gradient correctness verified but lack full training evidence. Phase 8 requires each module to pass the real-GT training gate before being considered for composability.

### Current Status

| Module | Gradient | Forward Pixel | GT Quality | Training | Gate |
|:-------|:--------:|:-------------:|:----------:|:--------:|:----:|
| M2 packed/dense | ✅ SUPPORTED | ❌ INCONCLUSIVE | ❌ NOT_TESTED | ❌ NOT_TESTED | ❌ |
| M3 SH degree | ✅ SUPPORTED | ❌ PARTIAL | ❌ NOT_TESTED | ❌ NOT_TESTED | ❌ |
| M4 radius_clip | ✅ SUPPORTED | ❌ INCONCLUSIVE | ❌ NOT_TESTED | ❌ NOT_TESTED | ❌ |
| M5 eps2d | ✅ SUPPORTED | ❌ INCONCLUSIVE | ❌ NOT_TESTED | ❌ NOT_TESTED | ❌ |

---

## 2. M2 — Packed/Dense Training Gate

### Background
- `packed=True` (default): Skips Gaussians invisible from current camera before rasterization
- `packed=False` (dense): Processes all Gaussians regardless of visibility
- Performance effect: ±0.5% (negligible on synthetic data)
- Gradient: ✅ Norms match within 5.5e-7 between packed and dense (Phase 6+)

### Required Steps
1. **Short sanity (500 steps)** with real GT on room
2. Check:
   - Forward executes for both packed and dense
   - Backward executes — no NaN/Inf
   - Loss decreases reasonably
   - PSNR behaves reasonably
   - Gradient norms finite
   - Gaussian count trajectory not obviously corrupted
   - Densification works
   - Pruning works
3. **If PASS**, proceed to 30K full training
4. **If FAIL**, locate root cause (do not adjust tolerance)

### Ready-to-Run Command
```bash
# M2 sanity check (500 steps, room, real GT)
python scripts/epic05/phase7/run_m2_sanity.py \
    --scene room --steps 500 --tile-size 16
```

---

## 3. M3 — SH Degree Training Gate

### Background
- SH degree controls color representation fidelity:
  - SH0: DC only (1 coefficient per color channel)
  - SH1: 4 coefficients per channel (diffuse + simple view-dependence)
  - SH3: 16 coefficients per channel (full 3DGS default)
- Performance effect: ±0.3%
- Gradient: ✅ gradcheck PASS for SH0, SH1, SH3 (Phase 6+)
- **Important:** SH degree changes the **color representation itself**, not just computational efficiency. Training must verify that lower SH degrees don't collapse or degrade quality in an irrecoverable way.

### Required Steps
1. **Short sanity (500 steps)** for each SH degree value (0, 1, 3)
2. Check:
   - Forward executes
   - Backward executes — no NaN/Inf
   - Loss decreases for ALL SH degrees
   - PSNR behaves reasonably for ALL SH degrees
   - Gradients finite
   - No color artifacts or training instability at low SH degrees
   - Progressive SH scheduling (0→1→2→3) works correctly
3. **If ALL PASS** for all 3 SH degrees, proceed to 30K full training
4. **If any FAIL** (e.g., SH0 produces corrupted colors), document as a limitation

### Key Distinction
This is NOT an inference-only SH benchmark. The training gate tests whether SH degree settings work correctly in the full training loop (forward + backward + optimizer + densification).

### Ready-to-Run Command
```bash
# SH degree training gate needs a script (doesn't exist yet)
# Will need: run_m3_sanity.py that tests SH degree=0, 1, 3
python scripts/epic05/phase7/run_sanity.py \
    --scene room --steps 500 --tile-size 16 \
    --sh-degree 0

python scripts/epic05/phase7/run_sanity.py \
    --scene room --steps 500 --tile-size 16 \
    --sh-degree 1

python scripts/epic05/phase7/run_sanity.py \
    --scene room --steps 500 --tile-size 16 \
    --sh-degree 3
```

---

## 4. M4 — radius_clip Training Gate

### Background
- Radius clip thresholds the projected screen-space radius of Gaussians
- Values tested: 0.0 (default/no clip), 0.001, 0.01
- Performance effect: ±0.1% (negligible)
- Gradient: ✅ Norms match within 3.1e-7 vs rclip=0.0 (Phase 6+)

### Required Steps
1. **Short sanity (500 steps)** for each radius_clip value
2. Check:
   - Training stability maintained
   - No NaN/Inf
   - Loss decreases
   - PSNR behaves normally
   - Densification/pruning work correctly
   - No premature Gaussian culling from aggressive clipping
3. Test at least: radius_clip=0.0, 0.001, 0.01
4. **If PASS**, proceed to 30K

### Ready-to-Run Command
```bash
# M4 radius_clip sanity (500 steps)
python scripts/epic05/phase7/run_full.py \
    --scene room --tile-size 16 --steps 500 \
    --radius-clip 0.001

python scripts/epic05/phase7/run_full.py \
    --scene room --tile-size 16 --steps 500 \
    --radius-clip 0.01
```

---

## 5. M5 — eps2d Training Gate

### Background
- eps2d adds a small epsilon to the 2D covariance diagonal for numerical stability
- Values tested: 0.01, 0.1 (default), 0.5
- Performance effect: ±0.1% (negligible)
- Gradient: ✅ Finite gradients, minor variation expected (eps2d changes numerical path)

### Required Steps
1. **Short sanity (500 steps)** for each eps2d value
2. Check:
   - Training stability maintained
   - No NaN/Inf (eps2d is supposed to prevent this)
   - Loss decreases
   - PSNR behaves normally
   - No quality degradation from excessive blur
3. Test at least: eps2d=0.01, 0.1, 0.5
4. **If PASS**, proceed to 30K

### Ready-to-Run Command
```bash
# M5 eps2d sanity (500 steps)
python scripts/epic05/phase7/run_full.py \
    --scene room --tile-size 16 --steps 500 \
    --eps2d 0.01

python scripts/epic05/phase7/run_full.py \
    --scene room --tile-size 16 --steps 500 \
    --eps2d 0.5
```

---

## 6. Output Directory Structure

```
results/epic05/phase8/
├── m2_sanity_packed.json
├── m2_sanity_dense.json
├── m3_sanity_sh0.json
├── m3_sanity_sh1.json
├── m3_sanity_sh3.json
├── m4_sanity_rclip_0.001.json
├── m4_sanity_rclip_0.01.json
├── m5_sanity_eps2d_0.01.json
├── m5_sanity_eps2d_0.5.json
├── m2_full_30k.json  (if sanity passes)
├── m3_full_30k_sh0.json
├── m3_full_30k_sh1.json
├── m3_full_30k_sh3.json
├── m4_full_30k_rclip_0.001.json
├── m5_full_30k_eps2d_0.01.json
└── m2_m5_training_gate_summary.json
```
