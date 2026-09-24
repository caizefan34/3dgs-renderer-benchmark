# FINAL-30K: C0_V3_FINAL30K vs B1A_ACCUTILE — 13-Scene × 30K Matched Benchmark

**Verdict: FINAL_PASS**

Frozen protocol: seed 42, 1080p-class (max_side 1920), 13 scenes, reference_v1 recipe (densify 500–15000/100, absgrad threshold 0.0008, opacity reset 3000, SH/1000, λ_dssim 0.2, COLMAP SfM init), 30000 iterations, identical seed-42 camera sequence per scene-pair. BASE = B1A_ACCUTILE (`0471fbd9…`); TEST = C0_V3_FINAL30K (`9baf8655…`, F9 + SCALAR_ADJOINT + H8-MR, PX=2, HIGS_BWD_ABSGRAD=1). Both arms run the same unified trainer; only the render call and binary bootstrap differ. Scheduling: dynamic clean-GPU selection (zero processes + <500 MiB in two scans ≥120 s apart), per-run contamination snapshots; contaminated runs are downgraded and listed, never dropped.

## 1. Per-scene results (before any aggregate)

| scene | T b1a (min) | T c0 (min) | speedup (ratio) | reduction (%) | PSNR b1a | PSNR c0 | ΔPSNR (dB) | SSIM b1a | SSIM c0 | ΔSSIM | LPIPS b1a | LPIPS c0 | ΔLPIPS | final N b1a | final N c0 | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bicycle | 26.0 | 24.2 | 1.0720 | 6.72 | 26.43 | 26.36 | -0.073 | 0.8382 | 0.8351 | -0.0030 | 0.2449 | 0.2477 | 0.0027 | 3,843,742 | 3,655,917 | OK |
| bonsai | 17.7 | 17.0 | 1.0448 | 4.29 | 32.69 | 32.71 | 0.018 | 0.9426 | 0.9423 | -0.0003 | 0.2657 | 0.2662 | 0.0005 | 803,731 | 766,685 | OK |
| counter | 19.9 | 17.7 | 1.1254 | 11.14 | 30.05 | 29.93 | -0.113 | 0.9115 | 0.9112 | -0.0003 | 0.2813 | 0.2819 | 0.0007 | 648,912 | 622,000 | OK |
| flowers | 22.7 | 21.5 | 1.0544 | 5.16 | 24.40 | 24.39 | -0.007 | 0.7631 | 0.7611 | -0.0020 | 0.2927 | 0.2932 | 0.0005 | 2,549,162 | 2,463,199 | OK |
| garden | 23.6 | 22.4 | 1.0548 | 5.20 | 24.19 | 24.16 | -0.032 | 0.7324 | 0.7332 | 0.0008 | 0.3205 | 0.3197 | -0.0007 | 2,555,871 | 2,548,077 | OK |
| kitchen | 20.7 | 19.5 | 1.0615 | 5.79 | 31.69 | 31.71 | 0.018 | 0.9306 | 0.9326 | 0.0020 | 0.1753 | 0.1742 | -0.0011 | 830,007 | 846,045 | OK |
| room | 18.7 | 17.3 | 1.0757 | 7.04 | 31.93 | 32.13 | 0.200 | 0.9271 | 0.9279 | 0.0008 | 0.3033 | 0.3025 | -0.0009 | 863,829 | 835,317 | OK |
| stump | 21.6 | 20.5 | 1.0569 | 5.38 | 30.70 | 30.59 | -0.111 | 0.8925 | 0.8910 | -0.0014 | 0.2254 | 0.2270 | 0.0016 | 2,577,962 | 2,604,637 | OK |
| treehill | 23.8 | 22.3 | 1.0650 | 6.10 | 25.78 | 25.76 | -0.019 | 0.8215 | 0.8178 | -0.0037 | 0.2956 | 0.2981 | 0.0025 | 2,918,772 | 2,743,550 | OK |
| train | 20.9 | 19.3 | 1.0862 | 7.94 | 24.78 | 24.58 | -0.205 | 0.8229 | 0.8231 | 0.0001 | 0.3262 | 0.3256 | -0.0007 | 483,515 | 478,478 | OK |
| truck | 20.0 | 18.6 | 1.0773 | 7.18 | 26.84 | 26.86 | 0.022 | 0.8915 | 0.8906 | -0.0009 | 0.2683 | 0.2691 | 0.0008 | 1,042,891 | 993,839 | OK |
| drjohnson | 18.5 | 17.6 | 1.0482 | 4.60 | 29.09 | 30.19 | 1.102 | 0.9106 | 0.9220 | 0.0114 | 0.3725 | 0.3452 | -0.0273 | 931,608 | 1,026,446 | OK |
| playroom | 17.8 | 16.7 | 1.0708 | 6.61 | 32.82 | 32.92 | 0.104 | 0.9476 | 0.9478 | 0.0002 | 0.2916 | 0.2912 | -0.0003 | 826,781 | 793,375 | OK |

ΔLPIPS = candidate − baseline; **negative is improvement** (sign not flipped). Speedup (ratio) and reduction (%) are separate representations of the same T pair.

## 2. Aggregate performance

- geomean speedup (primary): **1.0685×**
- arithmetic mean speedup (secondary): 1.0687×
- geomean time reduction: 6.20%
- success pairs: 13/13

## 3. Aggregate quality

- mean ΔPSNR: **0.070 dB**
- mean ΔSSIM: 0.0003
- mean ΔLPIPS: -0.0017 (negative = improvement)
- Garden (primary sensitivity scene) is reported in the per-scene table above, in the primary aggregate (no exclusion).

## 4. Time-to-quality (TTQ, R1 censoring semantics)

- Thresholds are derived from the FROZEN B1A final-eval quality per scene (not chosen after seeing results).
- geomean TTQ speedup to PSNR (reached/reached pairs only): **1.1374×** (11/13 comparable pairs)

| scene | threshold PSNR (b1a final) | TTQ c0 (s) | status | censor_time (s) |
|---|---|---|---|---|
| bicycle | 26.69 | — | TTQ_NOT_REACHED | 1468.3 |
| bonsai | 32.37 | 1019.2 | REACHED | — |
| counter | 29.44 | 1077.2 | REACHED | — |
| flowers | 23.97 | 1289.9 | REACHED | — |
| garden | 22.91 | 900.6 | REACHED | — |
| kitchen | 31.42 | 1169.6 | REACHED | — |
| room | 31.33 | 544.8 | REACHED | — |
| stump | 30.83 | — | TTQ_NOT_REACHED | 1237.4 |
| treehill | 27.07 | 1112.0 | REACHED | — |
| train | 24.30 | 1178.2 | REACHED | — |
| truck | 27.21 | 1115.4 | REACHED | — |
| drjohnson | 30.66 | 1057.8 | REACHED | — |
| playroom | 32.64 | 999.5 | REACHED | — |

Censored scenes (TTQ_NOT_REACHED) contribute their censor time as a bound only and are excluded from geomeans; reach counts are stated explicitly. Baseline TTQ to its own threshold is 30,000-step total wall time by construction (threshold = baseline final).

## 5. Phase timing (means over 30K iterations)

| phase | b1a mean (ms) | c0 mean (ms) |
|---|---|---|
| forward | 4.507 | 4.523 |
| backward | 21.596 | 19.236 |
| loss | 7.000 | 6.939 |
| densify | 1.105 | 0.990 |
| optimizer | 5.014 | 4.925 |

Phase timing describes WHERE time goes; it does not by itself attribute causality to individual composed modules (§35 discipline).

## 6. N_GS comparison

- geomean final-N ratio (c0/b1a): **0.9813×**
- mean final N: b1a 1,605,906 / c0 1,567,505
- Per-scene values in the table above; gate E preflight showed ≤3.9% max N_GS trajectory deviation at 2K on room/bicycle/garden.

## 7. Failure manifest

- none (26/26 SUCCESS, zero contamination)

## 8. Verdict

**FINAL_PASS**

Rule: PRE-REGISTERED 2026-09-23 before any 30K result was seen: FINAL_PASS = 26/26 SUCCESS, zero contamination, geomean speedup > 1.00, |mean dPSNR| <= 0.25 dB, no scene dPSNR < -0.50 dB; FINAL_FAIL = any run not SUCCESS, or geomean speedup <= 1.00 with dPSNR < 0, or mean dPSNR <= -0.50 dB; FINAL_MIXED = otherwise.

Inputs: {
  "n_success_pairs": 13,
  "n_failures_or_contaminated": 0,
  "n_contaminated": 0,
  "geomean_speedup": 1.0685072019237971,
  "mean_dPSNR": 0.06958029793481793,
  "worst_scene_dPSNR": -0.20507470408602657,
  "per_scene_dPSNR": {
    "bicycle": -0.0730262385849123,
    "bonsai": 0.018440522528884173,
    "counter": -0.11282174379674359,
    "flowers": -0.0070570475701110524,
    "garden": -0.03165339375081189,
    "kitchen": 0.017517557191045086,
    "room": 0.19987984535459447,
    "stump": -0.1111135180592484,
    "treehill": -0.01864959524113985,
    "train": -0.20507470408602657,
    "truck": 0.022196958837117364,
    "drjohnson": 1.1023225407275241,
    "playroom": 0.10358268960246164
  }
}

## 9. Provenance and limitations

- Binary identities: B1A_ACCUTILE `0471fbd95cae4cfa658bc1e2e3f7abe801c6f58a36a2e681a6d16af84279986b`; C0_V3_FINAL30K `9baf8655f859f3e0f456e99a2d8e5b3ddfe414fe17a1a3e9d8072e0afed95b91` (gates C+D+E PASS, `artifacts/higs-final30k-compat/final_gate.json`).
- eps2d: B1A 0.1 (frozen recipe), C0 0.3 (frozen validated stack) — renderer-internal, disclosed and quantified by gate D1.
- DSH-E P3 temporal speedup model: INVALID_SUPERSEDED_MODEL/DO_NOT_USE (§33). No E2 in the tested stack (§32). No causal H8 claims (§35).
- Timing is publication-grade only for runs launched on verified-clean GPUs without foreign-process contamination; contaminated runs are downgraded and listed in §7.
- Historical planning figures (B1A 13-scene cohort) were used for capacity planning only, never as results.
