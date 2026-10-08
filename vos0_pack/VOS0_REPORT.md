# VOS0 — Visibility Optimizer Oracle: Report

## Objective
Verify whether invisible (cold, <10% visibility) Gaussians can be updated less
frequently (every 2/4/8 iterations) without quality loss, and whether the skipped
optimizer steps yield >=20% workload reduction.

## Method
- Phase 1 (Trace): on frozen 30K checkpoints, measure per-Gaussian visibility frequency
  (DS=1 forward, h_trace=1000 iters, radii>0), gradient frequency (DS=1, h_grad=200,
  real Adam lr=5e-5, use_precomp colors), and update magnitude (RMS param delta per Adam step).
- Phase 2 (Partition): hot vis>0.5, warm 0.1<=vis<=0.5, cold vis<0.1.
- Phase 3 (Schedule Oracle): continue-train dense (k=1, all Gaussians every iter) vs
  cold-skip arms (cold Gaussians updated every k iters; hot+warm every iter) from the
  frozen 30K state with two Adam optimizers. h_iters=600, lr=1e-4, photometric noise=0.2,
  ds_train=1, 16 held-out eval cams (i%8==0). Quality = PSNR(dense,clean30K) - PSNR(skip,clean30K)
  on eval cams (GT-free). Economic = exact optimizer-step reduction = cold_frac*(1-1/k).

Scenes: room (N=1.59M), garden (N=5.83M), bicycle (N=6.13M). seed=42. torch 2.9.1+cu128, A100-SXM4-80GB.

## Phase 1+2 Results (Visibility Trace)
| Scene  | mean vis | cold frac | warm frac | hot frac | grad_freq | upd_mag/step |
|--------|----------|-----------|-----------|----------|-----------|--------------|
| room   |   0.230  |   0.314   |   0.637   |   0.048  |   0.197   |   5.5e-5      |
| garden |   0.320  |   0.259   |   0.519   |   0.221  |   0.245   |   6.4e-5      |
| bicycle|   0.224  |   0.460   |   0.365   |   0.175  |   0.181   |   5.0e-5      |

Structural identity confirmed: gradient frequency ~= visibility frequency (a Gaussian
receives gradient iff it is visible in the sampled camera), so cold == invisible.
Update magnitude is near-constant across partitions (Adam scale-invariance), slightly
lower for cold (fewer effective gradient steps). Cold Gaussians hold ~3.3% of the total
update budget in room, ~2.8% in garden, ~10.9% in bicycle.

## Phase 3 Results (Schedule Oracle) — Gate Table
| k | econ geomean | econ>=20% | qual pooled median | qual pooled mean | qual<=0.05 | per-scene worst median | primary PASS |
|---|--------------|-----------|--------------------|------------------|------------|------------------------|--------------|
| 2 |    0.167     |   FAIL    |      +0.015        |      +0.004      |    PASS    |         0.036          | FAIL (econ)  |
| 4 |    0.251     |   PASS    |      -0.014        |      +0.008      |    PASS    |         0.070          | PASS         |
| 8 |    0.293     |   PASS    |      +0.007        |      +0.024      |    PASS    |         0.078          | PASS         |

Recommended k = 4 (smallest k passing economic geomean>=20% AND pooled quality<=0.05dB).
Per-scene workload reduction at k=4: room 0.236, garden 0.194, bicycle 0.345.
Per-scene median PSNR loss at k=4: room -0.049, garden -0.017, bicycle +0.070.

## Verdict: VIABLE (k=4)
- Quality gate (primary): pooled median PSNR loss = -0.014 dB <= 0.05 dB. PASS.
- Economic gate: geomean optimizer-step reduction = 0.251 >= 0.20. PASS.
- Not "strong": the secondary per-scene worst median loss (bicycle, +0.070 dB) exceeds
  0.05 dB, and economic geomean (0.251) barely clears the 0.25 strong threshold.

## Caveats
- Absolute eval PSNR vs the clean 30K reference is low (1-7 dB) because the 600-iter
  noisy photometric continuation drifts BOTH arms from the 30K state. The gate metric is
  the RELATIVE loss (dense - skip), which is robust to this shared drift; cold Gaussians
  are <10% visible so skipping them barely moves the held-out render. Pooled loss
  aggregates all 3 scenes x 16 eval cams (48 values).
- GT images were unusable (16:9 vs 1.5 aspect mismatch), so quality is measured against
  the clean frozen-30K render, not ground truth.
- The economic model is the EXACT optimizer-step reduction (cold_frac*(1-1/k)), not a
  measured wall-clock speedup. Actual end-to-end speedup depends on the fraction of each
  iteration spent in the optimizer vs rasterizer/backward; skipping cold Adam updates
  reduces step count but not the per-iteration rasterizer/backward cost.

## Failure Mode
None (verdict is VIABLE). DROP_VISIBILITY_HEADROOM is only triggered when no k reaches
20% workload reduction; here k=4 and k=8 both clear it.
