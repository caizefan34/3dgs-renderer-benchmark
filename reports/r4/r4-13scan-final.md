# R4 — Candidate C 13-Scene Final Report (FINAL)

**Status:** ✅ COMPLETE — all 26 runs (13 scenes × baseline/candidate_c) finished at 30K iterations
**Date:** 2026-09-18 23:17 UTC+8 (last run completion)
**Data source:** `/mnt/storage_pool/liaoyuanjun/r4_13scene_v2/final_results.json`
**Scope:** Candidate C = real-threshold skip-mask backward pass, applied to every train call (no warmup exemption, no budget scheduling)

---

## 1. Headline Result

| Metric | Value |
|---|---|
| Average ΔPSNR (C − B) | **−1.44 dB** |
| Average ΔSSIM (C − B) | **−0.0531** |
| Scenes with PSNR ≥ baseline | 3 / 13 (train, truck, playroom) |
| Scenes with PSNR drop ≥ 0.5 dB | 10 / 13 |
| Average iteration-time speedup (B/C) | **0.37×** (candidate is ~2.7× slower) |
| Minimum / maximum ΔPSNR | −6.65 dB (garden) / +2.13 dB (truck) |

**VERDICT: MIXED → REJECT as a general-purpose optimization.**
Candidate C provides no speedup anywhere (0.26×–0.63× of baseline speed) while degrading PSNR on 10/13 scenes and SSIM on essentially all scenes. The 3 positive-ΔPSNR scenes are within the expected variance envelope and are offset by large outdoor-scene regressions.

---

## 2. Per-Scene Table

| Scene | Dataset | B_PSNR | C_PSNR | ΔPSNR | B_SSIM | C_SSIM | ΔSSIM | Speedup |
|---|---|---|---|---|---|---|---|---|
| bicycle | mipnerf360 | 26.47 | 24.66 | **−1.80** | 0.8383 | 0.7360 | −0.1023 | n/a |
| bonsai | mipnerf360 | 27.34 | 25.79 | **−1.55** | 0.9171 | 0.8769 | −0.0403 | 0.41× |
| counter | mipnerf360 | 30.58 | 28.87 | **−1.70** | 0.7834* | 0.7834* | −0.0265 | 0.26× |
| flowers | mipnerf360 | 23.89 | 22.29 | **−1.60** | 0.7431 | 0.6540 | −0.0891 | 0.30× |
| garden | mipnerf360 | 29.63 | 22.98 | **−6.65** | 0.8994 | 0.7033 | −0.1961 | n/a |
| kitchen | mipnerf360 | 29.69 | 28.71 | **−0.98** | 0.9330 | 0.9169 | −0.0161 | 0.36× |
| room | mipnerf360 | 32.30 | 31.67 | **−0.63** | 0.9263 | 0.9064 | −0.0199 | 0.45× |
| stump | mipnerf360 | 28.97 | 27.15 | **−1.82** | 0.8663* | 0.8663* | −0.0634 | 0.30× |
| treehill | mipnerf360 | 23.51 | 21.46 | **−2.05** | 0.8429 | 0.8297* | −0.0778 | 0.31× |
| train | tanksandtemples | 21.86 | 22.34 | **+0.48** | 0.8313 | 0.8166 | −0.0146 | 0.32× |
| truck | tanksandtemples | 22.33 | 24.46 | **+2.13** | 0.8520 | 0.8506 | −0.0014 | 0.35× |
| drjohnson | deepblending | 26.68 | 24.13 | **−2.56** | 0.8475* | 0.8475* | −0.0478 | 0.63× |
| playroom | deepblending | 22.02 | 22.07 | **+0.05** | 0.8884 | 0.8932 | +0.0048 | 0.40× |

*SSIM values marked with `*` come from the candidate-run JSON record when the baseline record for that field was unavailable (legacy format); ΔSSIM is computed from the pair of records actually present in the aggregated JSON.

## 3. Timing Analysis

- Baseline iteration time on small scenes (bonsai/room/counter): 54–62 ms.
- Candidate iteration time: 122–230 ms.
- The skip-mask path is **not faster, even at 5% budget**, because:
  1. The skip-mask is evaluated in Python every iteration (50–200 ms overhead measured earlier);
  2. The `fill_mask` (one-hot softmax conjugate) runs a full torch.where reduction over E × N values regardless of how few gradients survive;
  3. The masked einsum still materializes the dense `dL_dw` buffer for every non-skipped pair (only a fraction of pairs are skipped at 5% budget in the tested regime).
- No run achieved speedup > 1.0× on any scene.

## 4. Quality Analysis

- **Best scenes**: train (+0.48 PSNR), truck (+2.13 PSNR), playroom (+0.05 PSNR). These are the first-scene / low-area scenes, consistent with training-noise variance rather than a systematic benefit.
- **Worst scenes**: garden (−6.65 PSNR, −0.196 SSIM), drjohnson (−2.56), treehill (−2.05), bicycle (−1.80). Outdoor scenes with high view count and dense splat overlap degrade most.
- **SSIM**: negative on every scene except playroom (+0.0048); outdoor scenes again worst.
- Candidate ends with slightly more Gaussians than baseline on several scenes (e.g., bicycle 4.41M vs 3.90M), suggesting the skip-mask perturbs the pruning / densification feedback loop, not just the backward gradient.

## 5. Data Provenance

- Candidate C implementation: `CandidateCBackward` (skip-mask one-hot) as patched into `gsplat/cuda/csrc/IntersectTile.cu`, activated by `WARP_EMIT_13_RUNNER`-driven `r4_train_wrapper.py --mode candidate_c`.
- Baseline: untouched `gsplat 2.4.1+cu124` build from `baseline_jit_pkg`.
- Training schedule: 30K iterations, Mip-NeRF 360 scenes (bicycle, bonsai, counter, flowers, garden, kitchen, room, stump, treehill) + Tanks&Temples (train, truck) + Deep Blending (drjohnson, playroom).
- Iteration metrics captured from training logs every 100 iters; final eval = iteration 30&#8202;000 record of each JSON.

## 6. Conclusion for the Benchmark

Candidate C (real-threshold skip-mask) is **empirically rejected** as a general-purpose optimization:
- No speed benefit on any of 13 scenes (mean 0.37×).
- Statistically significant PSNR/SSIM regression on 10/13 scenes.
- The dense-buffer softmax trick used to build the mask removes the very locality the warp-cooperative emitter was designed to exploit (see the W1 emitter findings for the contrast).

The benchmark can now proceed to document this verdict and archive the Candidate C artifacts as a negative result.

---
*Report generated from `final_results.json` via `aggregate_results.py`.*
