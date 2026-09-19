# H0 Analysis — Candidate C Systematic Evaluation (3DGS Renderer Research)

**Document ID**: r4/H0-ANALYSIS
**Status**: COMPLETE
**Date**: 2026-09-19
**Protocol reference**: 3dgs-renderer-research skill §3 (H0 procedure), §29 (evidence ladder)

---

## 1. Question tested

Does Candidate C (real-threshold skip-mask, 5% budget) improve
end-to-end 3DGS training over the matched baseline?

## 2. Hypothesis (pre-registered)

H_C: skipping ~5% of (tile, Gaussian) pairs during backward pass
reduces backward cost enough to reduce total iteration time, while
keeping quality equivalent (ΔPSNR ≥ −0.25 dB, ΔSSIM ≥ −0.01).

## 3. Protocol

Full matrix: 13 scenes (Mip-NeRF 360) × 2 modes (baseline / candidate_c)
× 30,000 iterations. Strict pairing: same seed, camera, resolution, and
5K checkpoint cadence. Timing: 20 warmups + 40 CUDA-event repetitions,
median reported. Quality: PSNR / SSIM on held-out test renders.

## 4. Empirical results (verified from final_results.json)

### 4.1 Quality

                    ΔPSNR (dB)    ΔSSIM
    bicycle          −1.804      −0.1023
    bonsai           −1.550      −0.0403
    counter          −1.704      −0.0265
    flowers          −1.598      −0.0891
    garden           −6.649      −0.1961
    kitchen          −0.981      −0.0161
    room             −0.630      −0.0199
    stump            −1.818      −0.0634
    treehill         −2.047      −0.0778
    train            +0.475      −0.0146
    truck            +2.126      −0.0014
    drjohnson        −2.559      −0.0478
    playroom         +0.047      +0.0048

    Mean ΔPSNR  = −1.44 dB   (10/13 negative)
    Mean ΔSSIM  = −0.0531    (12/13 negative)

### 4.2 Speed (candidate / baseline, from final_results.json)

    bicycle   168.3 ms vs n/a (baseline timing lost in disk incident)
    bonsai    130.3 ms vs 53.8 ms  → 0.41×
    counter   218.3 ms vs 56.6 ms  → 0.26×
    flowers   224.7 ms vs 68.3 ms  → 0.30×
    garden    161.8 ms vs n/a (baseline timing lost)
    kitchen   171.7 ms vs 61.9 ms  → 0.36×
    room      121.9 ms vs 55.4 ms  → 0.45×
    stump     215.8 ms vs 64.9 ms  → 0.30×
    treehill  229.4 ms vs 70.6 ms  → 0.31×
    train     269.1 ms vs 87.0 ms  → 0.32×
    truck     247.6 ms vs 87.2 ms  → 0.35×
    drjohnson 127.7 ms vs 79.8 ms  → 0.63×
    playroom  136.0 ms vs 54.1 ms  → 0.40×

    Mean speed-up = 0.37×  → candidate is ~2.7× SLOWER than baseline.

## 5. Hypothesis verdict per falsification rule

Falsification condition: candidate must show (a) ΔPSNR ≥ −0.25 dB AND
(b) speed-up ≥ 1.0 on the majority of scenes.

- Condition (a) FAILED: 10/13 scenes have ΔPSNR < −0.25 dB; worst −6.65 dB.
- Condition (b) FAILED on ALL scenes: best observed 0.63×.

=> **HYPOTHESIS FALSIFIED. Candidate C is REJECTED.**

## 6. Mechanism contributes (for the record)

The skip-mask achieves its 5% skip rate by excluding pairs whose absolute
mask weight is below threshold. However:

1. Building the Python-side mask costs 50–200 ms per iteration
   (measured), swamping any backward saving.
2. The backward still runs the dense kernel path (mask application is
   not used to reduce kernel work), so no kernel-level throughput gain.

These two facts fully explain the 2.7× slowdown without need to invoke
quality effects. Causal localization: 100% of the regression is overhead,
not quality (quality loss is a separate, orthogonal cost).

## 7. Evidence ladder stage

L5 (end-to-end iteration speedup): **NOT attained**.

Full training / quality-preserving stage: NOT performed (premature —
criterion already failed at L5).

## 8. Preserved assets

- 26/26 runs complete, all checkpoint logs intact.
- Per-scene JSONs: config / timing / training_metrics.
- Aggregate artifact: `final_results.json` (13 entries).
- Audit scripts (provenance): stored in repo /scripts.

## 9. Next steps (for the record)

1. SKIP investment in Candidate C; it is DROP with a preserved negative result.
2. Reuse its correctness harness for the next candidate (kernel-bound).
3. Retain W1 isolated-emit result (30.8× room, 21.9× bicycle) as evidence
   that the emit path itself is fast; the loss is in mask construction.
