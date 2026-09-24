# Phase 14C — Visibility/Culling Reconnaissance

**Date:** 2026-09-23
**Status:** COMPLETE
**Source:** gsplat v1.5.3, EPIC-05 Phase 11 data

---

## 1. Executive Summary

The current renderer already implements **all standard visibility and culling operations** as part of the packed projection pipeline. No new culling optimization exists that isn't already default or already FALSIFIED.

---

## 2. Existing Culling Mechanisms

### 2.1 Frustum Culling (DEFAULT)

**Status:** ALREADY IMPLEMENTED in `projection_ewa_3dgs_packed_fwd_kernel`

The packed projection kernel computes 2D radii for each Gaussian. Gaussians with `radius_x <= 0 || radius_y <= 0` are excluded from the packed output. This implicitly performs frustum culling — Gaussians behind the camera or outside the image plane get zero radius and are filtered.

**Evidence:** Packed output (`nnz`) is typically 20-25% of `N` for room scene (e.g., 1.59M → ~0.38M visible).

### 2.2 Radius Culling via radius_clip (M4 — FALSIFIED)

**Status:** FALSIFIED in Phase 11

`radius_clip` is a user-facing parameter (line 13 in `rendering.py`). Phase 11 tested rclip values from 0.0 to 5.0:
- rclip=0.5: zero Gaussians removed (bit-identical)
- rclip=1.0: 0.003% pixels affected
- rclip=2.0: 0.26% pixels affected, <1% intersection reduction
- rclip=5.0: 7.9% pixels affected, 5.8% fewer intersections → 1.18dB PSNR drop

**Verdict:** `<1% workload reduction` at quality-preserving thresholds. Training-level benefit is zero.

### 2.3 Per-Gaussian Frustum Test (DEFAULT)

**Status:** Already in projection kernel → no new optimization here.

### 2.4 Visibility-Based Compaction (DEFAULT)

**Status:** The packed mode (enabled by default) compacts visible Gaussians into a dense `[nnz]` array after projection. This reduces downstream work (SH evaluation, tile intersection, rasterization) by ~75% for room scene.

---

## 3. What Is NOT Already Implemented

| Optimization | Potential | Risk | Notes |
|:-------------|:---------:|:----:|:------|
| Hierarchical visibility | Medium | High | Would require tile-level visibility test before per-Gaussian projection |
| Occlusion culling | Medium | High | 3DGS has no depth pre-pass; occlusion changes per scene |
| Coarse tile culling | Low | Medium | Tile coverage is already bounded by tile size; Gaussians covering an empty tile still need to be projected |
| Gaussian count culling | FALSIFIED | — | M4 radius_clip already tested this with <1% benefit |

---

## 4. Conclusion

| Question | Answer |
|:---------|:-------|
| New culling optimization available? | **NO** |
| Existing culling already default? | **YES** — packed projection inherently filters invisible Gs |
| radius_clip revisited? | **NO** — FALSIFIED in Phase 11 |
| Next priority? | **NONE** — no viable new optimization found |

---

## 5. Files

- `reports/epic05/phase14_visibility_recon.md` (this file)
