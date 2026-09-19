# VERDICT REPORT — Candidate C (final, R4 + R5-A)

Status: COMPLETE
Assembly date: 2026-09-19 10:30 CST+8

---

## 1. R4 full-matrix verdict

All 26 runs (13 scenes x baseline/candidate_c) reached 30k iterations.

Metric | Result
-------|-------
avg dPSNR | -1.44 dB (10/13 below baseline, 3/13 >= baseline)
avg dSSIM | -0.0531 (12/13 negative, worst -0.1961 garden)
speed  | 0.37x average, 0.26x-0.63x range (all slower)

Per-scene highlights:
- garden is the catastrophic case: dPSNR -6.65 dB, dSSIM -0.1961.
- truck is the only clearly positive outlier: dPSNR +2.13 dB, though dSSIM still -0.0014.
- indoor scenes (room -0.63, bonsai -1.55, kitchen -0.98, counter -1.70) degrade less than
  outdoor (bicycle -1.80, treehill -2.05, flowers -1.60, garden -6.65, stump -1.82).

Timing: candidate_c took 0.26x-0.63x baseline throughput everywhere.
=> There is NO case where candidate_c is both faster and better.

## 2. R5-A multi-seed check (train/truck, seeds 0-2)

Paired deltas (candidate_c vs baseline):

train: seed0 +0.52 dB, seed1 -0.64 dB, seed2 -1.16 dB  (dSSIM all negative)
truck: seed0 +1.94 dB, seed1 0.00 dB, seed2 +0.81 dB   (dSSIM all negative)

=> Even the two scenes with best mean dPSNR show dSSIM < 0 in 6/6 runs.
The candidate's apparent "wins" are quality-reducing (detail loss), not neutral.

## 3. Evidence chain

- Correctness (0 mismatches, exact identity) and isolated-emit speedups
  (room 30.8x, bicycle 21.9x, garden 11.9x) are confirmed and reported as W1.
- Full-pipeline timing, all 13 scenes: candidate_c slower, worst 0.26x.
- Root causes identified: python-side skip-mask + dense softmax fill_mask
  (the one-hot choices degenerate to the same dense kernel cost),
  no save of intermediate gradients once the mask is all-ones (worst case),
  perturbation to densification/pruning feedback counted as slight geometry drift.

## 4. Recommendation

1. Archive candidate_c as a NEGATIVE result with this report.
2. Retain W1 isolated-emit positive findings (they validate the emission path).
3. Do NOT allocate a full 26-run WarmEmit matrix until the machine-level
   budget (disk ~20GB, free GPU slots) is re-confirmed by a separate review.
