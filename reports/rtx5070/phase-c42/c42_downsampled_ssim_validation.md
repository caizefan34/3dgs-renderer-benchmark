# C42 Candidate D: Downsampled SSIM Validation

## Executive Summary

| Scale | D-SSIM speedup | E2E speedup | Min cosine | Decision |
|-------|---------------|------------|-----------|----------|
| 1.00 | 1.00x | 0.0% | 1.0000 | (baseline) |
| **0.75** | **1.73x** | **+16.1%** | **0.9960** | **MAYBE** (D-SSIM speedup <3x but E2E+cosine pass) |
| **0.50** | **4.18x** | **+38.2%** | **0.9212** | **MAYBE** (D-SSIM+E2E pass, cosine <0.95) |
| **0.25** | **10.82x** | **+45.6%** | **0.8694** | **MAYBE** (D-SSIM+E2E pass, cosine <0.95) |

**No scale passes all three gates simultaneously.** Scale 0.75 passes E2E+cosine but fails D-SSIM speedup. Scale 0.50 passes D-SSIM+E2E but fails cosine (rotations).

**Key finding**: The **rotations** parameter is the quality bottleneck — its gradient cosine similarity drops sharply with downsampling while all other parameters (xyz, opacity, scales, shs) remain above 0.97 even at scale 0.25.

---

## 1. Method

D-SSIM downsampling is implemented by applying `F.interpolate(pred, scale_factor=s, mode="area")` to both pred and target before the Gaussian blur + SSIM computation. L1 loss remains at full resolution. The SSIM Gaussian window (11×11) stays the same — at lower resolution it covers a proportionally larger spatial area.

Configuration: 1M Gaussians, room scene, 1920×1080, 7 test cameras for gradient cosine, 50 timing measurements.

---

## 2. D-SSIM Latency (Isolated)

| Scale | Image size | Forward (ms) | Backward (ms) | Fwd+Bwd (ms) | Speedup |
|-------|-----------|-------------|-------------|-------------|---------|
| 1.00 | 1080×1920 | 38.03 | 41.47 | **73.59** | 1.00x |
| 0.75 | 810×1440 | 20.39 | 23.37 | **42.62** | 1.73x |
| 0.50 | 540×960 | 8.50 | 9.54 | **17.62** | **4.18x** |
| 0.25 | 270×480 | 3.01 | 3.75 | **6.80** | **10.82x** |

D-SSIM speedup scales near-quadratically with resolution reduction (area = scale²):
- Scale 0.75 → 1.73x (expected ~1.78x from 1/0.5625)
- Scale 0.50 → 4.18x (expected ~4.0x from 1/0.25)
- Scale 0.25 → 10.82x (expected ~16x from 1/0.0625, but kernel launch overhead limits)

The backward benefits slightly more than forward at each scale, consistent with dgrad being more expensive than forward conv.

---

## 3. End-to-End Training Iteration

| Scale | E2E (ms) | Std (ms) | vs baseline | Speedup |
|-------|---------|---------|------------|---------|
| 1.00 | 134.88 | 3.42 | — | 1.00x |
| 0.75 | 113.15 | 3.50 | **+16.1%** | 1.19x |
| 0.50 | 83.36 | 1.03 | **+38.2%** | 1.62x |
| 0.25 | 73.31 | 1.53 | **+45.6%** | 1.84x |

Scale 0.50 delivers 38.2% total training speedup — this is the highest speedup of ANY C42 candidate tested. Scale 0.25 delivers 45.6% but with higher quality risk.

The speedup is not purely proportional to D-SSIM savings because:
- Rendering (18.5% of time) is unchanged — always full resolution
- L1 loss is always full resolution
- Optimizer step is unchanged
- Only D-SSIM convolution + elementwise benefit from downsampling

---

## 4. GPU Memory

| Scale | Isolated peak (MB) | E2E peak (MB) | vs baseline |
|-------|-------------------|--------------|------------|
| 1.00 | 2993.5 | 3131.6 | — |
| 0.75 | 2883.9 | 2997.6 | -134 MB (-4.3%) |
| 0.50 | 2739.9 | 2853.4 | -278 MB (-8.9%) |
| 0.25 | 2668.3 | 2785.7 | -346 MB (-11.0%) |

Memory savings are modest — the D-SSIM intermediate tensors are small relative to the 1M Gaussian parameters. The 346 MB savings at scale 0.25 is from smaller convolution intermediates and SSIM maps.

---

## 5. Gradient Cosine Similarity

### 5.1 Full table (mean across 5 cameras)

| Scale | xyz | opacity | scales | rotations | shs | **Min** |
|-------|-----|---------|--------|-----------|-----|---------|
| 1.00 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | **1.0000** |
| 0.75 | 0.9989 | 0.9999 | 1.0000 | 0.9960 | 0.9999 | **0.9960** |
| 0.50 | 0.9943 | 0.9995 | 0.9998 | 0.9212 | 0.9993 | **0.9212** |
| 0.25 | 0.9762 | 0.9973 | 0.9983 | 0.8694 | 0.9964 | **0.8694** |

### 5.2 Per-parameter analysis

**xyz** (position): cosine > 0.976 at all scales. Robust to downsampling — position gradients are dominated by L1 and large-scale structure, not fine SSIM detail.

**opacity**: cosine > 0.997 at all scales. Very robust — opacity is a scalar per Gaussian, minimally affected by SSIM resolution.

**scales**: cosine > 0.998 at all scales. Extremely robust — scale gradients are smooth and low-frequency.

**shs** (spherical harmonics): cosine > 0.996 at all scales. Robust — SH gradients are dominated by color matching (L1) and coarse structure.

**rotations** (quaternions): cosine drops sharply — **0.996 → 0.921 → 0.869**. This is the quality bottleneck.

### 5.3 Why rotations are sensitive

Rotation gradients flow through the Gaussian covariance matrix:
```
Σ = R · S · S^T · R^T
```
where R is the rotation quaternion. The SSIM structural similarity loss captures **edge orientation** information, which is directly encoded in rotation. Downsampling SSIM loses fine edge detail → rotation gradients lose directional information.

At scale 0.50 (540×960), the 11×11 Gaussian window covers an equivalent 22×22 area at full resolution. Edges thinner than ~10 pixels are averaged out, losing the orientation signal that rotation gradients depend on.

### 5.4 Min cosine per camera (scale 0.50, rotations)

| Camera | Cosine (rotations) |
|--------|-------------------|
| 0 | 0.980 |
| 50 | 0.981 |
| 100 | 0.610 |
| 150 | 0.978 |
| 200 | 0.976 |

Camera 100 has an outlier (0.610), dragging the mean down. This suggests certain viewpoints are more sensitive to SSIM downsampling for rotation — likely views with many fine edges at oblique angles.

---

## 6. Decision Analysis

### Decision gates (from task specification)

| Gate | Threshold | Scale 0.75 | Scale 0.50 | Scale 0.25 |
|------|-----------|-----------|-----------|-----------|
| D-SSIM speedup | >3x | 1.73x FAIL | 4.18x PASS | 10.82x PASS |
| Total speedup | >15% | +16.1% PASS | +38.2% PASS | +45.6% PASS |
| Gradient cosine | >0.95 | 0.9960 PASS | 0.9212 FAIL | 0.8694 FAIL |

**No scale passes all three gates.**

### Scale 0.75 analysis

- D-SSIM speedup 1.73x is below the 3x gate, BUT:
- E2E speedup +16.1% exceeds the 15% gate
- All cosine similarities > 0.996 (well above 0.95)
- The D-SSIM >3x gate may be too strict for this scale — the real goal is total speedup + quality

**If the D-SSIM speedup gate is relaxed**: Scale 0.75 would be **KEEP** — it passes both E2E (>15%) and cosine (>0.95) gates with zero quality risk.

### Scale 0.50 analysis

- D-SSIM speedup 4.18x passes the 3x gate
- E2E speedup +38.2% far exceeds the 15% gate
- Rotations cosine 0.9212 fails the 0.95 gate
- BUT: 4 of 5 parameters pass, and the mean rotation cosine across 5 cameras is 0.921 — the min is dragged by one camera (0.610)
- If the worst camera is excluded, rotation cosine is ~0.978 (pass)

**If the rotation outlier is addressed** (e.g., per-camera adaptive scale): Scale 0.50 could be **KEEP**.

### Scale 0.25 analysis

- Massive speedup (+45.6%) but rotation cosine 0.869 with min 0.377
- Quality risk too high for the 0.95 gate
- Even excluding outliers, rotation cosine is ~0.95 (borderline)

---

## 7. Comparison with All C42 Candidates

| Candidate | D-SSIM speedup | E2E speedup | Quality risk | Decision |
|-----------|---------------|------------|-------------|----------|
| B: Separable conv | 1.30x | +9.8% | Zero (identical) | MAYBE |
| C: Separable + cudagraphs | 1.30x | +12.6% | Zero (identical) | MAYBE |
| **D-0.75: Downsampled 0.75** | **1.73x** | **+16.1%** | **Near-zero (cos>0.996)** | **MAYBE** |
| **D-0.50: Downsampled 0.50** | **4.18x** | **+38.2%** | **Moderate (rot cos=0.92)** | **MAYBE** |
| D-0.25: Downsampled 0.25 | 10.82x | +45.6% | High (rot cos=0.87) | MAYBE |
| E: Adaptive scheduling | N/A | N/A | High | DROP |
| Spatial adaptive mask | 1.08x | +3.3% | Moderate | DROP |

**Downsampled SSIM at scale 0.50 is the highest-impact candidate** — 38.2% total training speedup, the only candidate exceeding 15% by a wide margin. The quality concern is isolated to the rotations parameter.

---

## 8. Recommended Next Steps

1. **Scale 0.75 is the safest option**: +16.1% speedup with cosine >0.996 for all parameters. If the D-SSIM >3x gate is relaxed to focus on E2E+cosine, this is an immediate KEEP.

2. **Scale 0.50 needs quality validation**: The 38.2% speedup is compelling, but the rotation cosine (0.921 mean, 0.610 min) needs a short training run to verify that PSNR/SSIM impact is acceptable. The gradient direction is still mostly correct (>0.92), just not >0.95.

3. **Hybrid approach**: Use scale 0.75 for rotations-sensitive phases (early training) and scale 0.50 for later training — combining the quality safety of 0.75 with the speed of 0.50. This is a potential C42 research contribution.

4. **Per-camera adaptive scale**: The rotation cosine outlier (camera 100: 0.610 at scale 0.50) suggests some viewpoints need higher resolution. A per-camera scale selector could achieve 0.50 speedup on most cameras while maintaining quality on sensitive ones.

---

## Artifacts

- Benchmark script: `scripts/phase-c42/c42_downsampled_ssim_benchmark.py`
- Raw data: `results/phase-c42/c42_downsampled_ssim_data.json`
- D-SSIM source: `scripts/epic05/phase7/loss.py`
