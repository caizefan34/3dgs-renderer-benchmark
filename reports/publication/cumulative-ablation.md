# P2: cumulative ablation A0 → A1 → A2 → A3(=C0)

**Status: COMPLETE — Stage A (3 scenes × 3 arms, gate protocol) + Stage B (13 scenes ×
3 arms, full 30K), all 39 runs + 13 C0 references clean.** Frozen evidence:
`ablations/`, `a_arm_functional_gate.json`, `aggregates/p2_*.json`,
`tables/table2..3`, `figures/figCD_ablation.png`. Regeneration:
`aggregate_publication.py` → `make_figtables.py`.

## Design

One frozen binary (ext `9baf8655`), env-switched arms — every pair differs ONLY in the
switched optimization, same data order, same seed, same GPUs class:

- **A0** = internal parent: `HIGS_DISABLE_F9=1`, scalar-adjoint off, H8-MR off
- **A1** = A0 + F9 (forward exact-gather)
- **A2** = A1 + SCALAR_ADJOINT (backward)
- **A3 = C0** = A2 + H8-MR (all three on; the frozen FINAL method)

Functional gate (pre-registered): frames bit-identical across all four arms; gradient
rel-err ≤ 5e-05; absgrad present everywhere; PX invariance — ALL PASS
(`a_arm_functional_gate.json`). The chain is a controlled numerical-equivalence
ablation: any wall/quality delta is execution-path, not math.

## Final results (13 scenes, seed 42, clean pairs)

| increment | geomean speedup | faster | mean ΔPSNR | mean ΔSSIM | mean ΔLPIPS | N geomean ratio |
|---|---|---|---|---|---|---|
| A0→A1 (F9) | 1.0076 | 8/13 | +0.092 dB | +0.00218 | −0.00270 | 1.0126 |
| A1→A2 (SCALAR_ADJOINT) | 0.9985 | 4/13 | −0.048 dB | −0.00188 | +0.00214 | 0.9988 |
| A2→A3 (H8-MR) | 0.9984 | 4/13 | +0.035 dB | +0.00098 | −0.00127 | 0.9996 |
| **A0→C0 (full stack)** | **1.0045** | **10/13** | **+0.079 dB** | +0.00128 | −0.00183 | 1.0110 |

**The three frozen CUDA optimizations are wall-small (~0.45% total geomean; 10/13
faster) and quality-neutral** (mean ΔPSNR +0.079 dB — but 12/13 scenes within ±0.15 dB
with mean **+0.022**; drjohnson alone +0.770, the same trajectory-sensitive scene the
headline reports, here too the single large cell). The 5-scene interim (1.0038) and the
final 13-scene value (1.0045) agree — the interim did not mislead. Per-module wall
deltas at this scale are NOT individually attributable (§9: no causal kernel
attribution from ±0.5% wall); the per-increment faster/slower splits (8/4/4 of 13)
make the sub-percent noise explicit — no single increment carries a uniform direction,
only the full stack does (10/13).

**drjohnson cross-check:** the scene diverges in BOTH comparisons — cross-codebase
(b1a vs c0: +1.102 dB) and intra-binary (a0 vs c0: +0.770 dB) — while all other scenes
stay within ±0.15 dB of zero in the intra-binary chain. drjohnson's optimization
trajectory is unusually sensitive to small execution-path changes; P6's seed pairs
decide whether the effect is reproducible or seed-specific. Either way it is labeled
per-scene, never generalized (§20).

## Attribution honesty — what the ablation does and does not explain

The headline speedup (C0 vs B1A = 1.0685×, §19 geomean of per-scene ratios) is a
CROSS-CODEBASE result: C0 runs the frozen HIGS binary (backward_mode `higs_native`);
B1A runs gsplat v1.5.3 + accutile. Three separable measurements exist, on two families:

1. **Within the HIGS binary (this ablation):** F9 + SCALAR_ADJOINT + H8-MR together add
   ~+0.45% wall over the internal parent A0 (A0: all three off via env; 10/13 faster),
   quality-neutral (12/13 within ±0.15 dB). A0 itself already trains ~6.3% faster than
   B1A on room — so the three switched branches are a small fraction of the
   cross-codebase gap.
2. **Within the gsplat family (B1 gate):** accutile is worth +2.2–4.6% wall (b1a vs b1,
   consistent across scenes and seeds) plus the garden quality effect (~0.6–0.8 dB).
3. **Residual:** the HIGS base execution paths (data pipeline, trainer loop, native
   backward layout outside the three switched branches) vs gsplat+accutile carry the
   remainder. Per §21 (research-tree freeze) this is NOT further decomposed — no new
   optimization branches are opened in the publication phase.

The publication does NOT claim the three optimization branches explain the headline;
the ablation section states the decomposition plainly and stops at the frozen boundary.

## Reporting discipline

- §19: geomean-of-ratios speedups; per-scene reduction geomean labeled separately.
- §20: quality described as "quality-neutral within the matched benchmark" — supported
  here by ΔPSNR/ΔSSIM/ΔLPIPS within noise on the accumulating scene set.
- Garden's ±0.6 dB wiggles between adjacent arms are disclosed as that scene's
  demonstrated trajectory-noise scale (cf. B1-gate seed evidence), not smoothed away.
