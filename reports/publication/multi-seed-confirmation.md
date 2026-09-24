# P6: multi-seed confirmation — seed robustness of the headline

**Status: COMPLETE — 16 P6 runs (4 scenes × {B1A, C0} × seeds {43, 44}) + 8
FINAL-30K seed-42 references, all clean at PUBLICATION grade.** One contamination
incident (b1as43_train, foreign pid mid-run) was re-run clean per protocol; the
contaminated attempt is preserved as `b1as43_train_contaminated_attempt1` with
forensics. Frozen evidence: `multiseed/` (run trees), `aggregates/p6_results.json`,
`tables/table5_p6_multiseed.md` (all derived; regeneration: `p6_analysis.py` →
`aggregate_publication.py` → `make_figtables.py`).

## Results (executed framework — `p6_results.json`)

### 1. Speedup stability: CONFIRMED

| seed | drjohnson | train | bicycle | room | 4-scene geomean |
|---|---|---|---|---|---|
| 42 | 1.0482 | 1.0862 | 1.0720 | 1.0757 | **1.0705** |
| 43 | 1.0590 | 1.1022 | 1.0607 | 1.0727 | **1.0735** |
| 44 | 1.0467 | 1.0869 | 1.0652 | 1.0267 | **1.0561** |

**All 12 per-scene speedups > 1** (range 1.0267–1.1022); per-seed geomeans
1.0561–1.0735 straddle the FINAL-30K headline (1.0685, 13-scene). The headline
direction and magnitude are seed-robust on the subset.

### 2. Quality-delta stability: CONFIRMED (§20 wording unchanged)

| scene | ΔPSNR per seed (42/43/44) | spread |
|---|---|---|
| drjohnson | +1.102 / +0.787 / +1.597 | 0.810 |
| train | −0.205 / −0.004 / −0.051 | 0.201 |
| bicycle | −0.073 / −0.088 / −0.025 | 0.064 |
| room | +0.200 / −0.100 / +0.059 | 0.300 |

train/bicycle/room straddle zero within ±0.3 dB — "quality-neutral within the
matched benchmark" holds per-seed on 3 of 4 scenes. drjohnson is the reproducible
per-scene positive (below).

### 3. drjohnson R-13: REPRODUCIBLE (see claim map §4.3)

ΔPSNR +1.102/+0.787/+1.597 (all > +0.5, mean +1.16); speedup 1.0482/1.0590/1.0467
(all > 1). Upgraded from "outlier" to "reproducible scene-level effect" per the
pre-registered rule — reported per-scene, never generalized (§20).

### 4. Without-drjohnson sensitivity

| seed | 3-scene geomean | mean ΔPSNR |
|---|---|---|
| 42 | 1.0780 | −0.026 |
| 43 | 1.0784 | −0.064 |
| 44 | 1.0593 | −0.006 |

The aggregate is robust to excluding the reproducible per-scene effect: geomeans
stay > 1, mean ΔPSNR ≈ 0.

### 5. No new claims from P6 alone

P6 quantifies variance around C-01/C-03; it creates no new speedup claim. The
multi-seed "confirmation" wording (N-07) is now available, scoped to the subset:
"seed-robust on the 4-scene subset (3 seeds, all per-scene speedups > 1, per-seed
geomeans 1.056–1.074)".

## Question

FINAL-30K is a single-seed (42) point estimate. P6 quantifies the seed-level variance
of (a) the headline speedup and (b) the quality deltas, on a 4-scene subset
(drjohnson, train, bicycle, room) chosen to include the drjohnson outlier (R-13) and
three representative scenes.

## Garden seed pairs (COMPLETE, 3 seeds — evidence for C-11)

| garden | b1a | b1 | b1a − b1 | wall edge (b1a faster) |
|---|---|---|---|---|
| seed 42 | 24.194 | 23.552 | +0.642 dB | +3.88% |
| seed 43 | 24.969 | 24.163 | +0.806 dB | +2.21% |
| seed 44 | 24.613 | 23.480 | +1.132 dB | +1.36% |

**Sign-consistent across all three seeds on BOTH axes**: accutile improves garden
PSNR by +0.642/+0.806/+1.132 dB (mean **+0.860**) AND is faster by +1.4–3.9% wall
(mean +2.48%). The effect is systematic, not seed luck — claim C-11 is settled on
garden. (Table = `garden_seed3_table.json`; note the Δ convention here is
b1a − b1, positive = accutile helps — the B1-gate report quotes the same numbers as
b1 − b1a losses when REMOVING accutile.)
- Within-arm seed spread: b1a 24.194→24.969→24.613 (spread 0.78), b1
  23.552→24.163→23.480 (spread 0.68).
- **Scale lesson for P6**: on garden, seed-level PSNR movement (±0.6–0.8 dB) exceeds
  the headline's mean ΔPSNR (+0.070 dB) by an order of magnitude. Per-scene single-seed
  deltas of that size are not evidence of anything; only cross-arm differences that
  persist across seeds are — and this one does.

## Analysis framework (pre-registered; executed above)

For each scene s and seed k ∈ {42, 43, 44}:

1. **Speedup stability**: `speedup[s,k] = wall_b1a[s,k] / wall_c0[s,k]`. Report
   per-scene mean ± spread across seeds, and the 4-scene geomean per seed. The
   headline claim is reinforced if per-seed geomeans stay within the FINAL-30K
   band (all scene speedups > 1; geomean near 1.07).
2. **Quality-delta stability**: `ΔPSNR[s,k] = psnr_c0[s,k] − psnr_b1a[s,k]`. Report
   per-scene mean ± spread. Wording stays §20-compliant: "quality-neutral within the
   matched benchmark" requires the per-seed deltas to straddle zero within the
   seed-spread scale.
3. **drjohnson outlier (R-13)**: does +1.102 dB at seed 42 persist at seeds 43/44?
   **VERDICT (quartet complete): YES — reproducible.** ΔPSNR = +1.102/+0.787/+1.597
   (seeds 42/43/44, all > +0.5, mean +1.16); speedup 1.0482/1.0590/1.0467 (all > 1).
   Upgraded from "outlier" to "reproducible scene-level effect" per the pre-registered
   rule — reported per-scene, never generalized (§20).
   **Pre-seed characterization (executed, `r13_drjohnson_review.json`):** identical
   start (−0.001 dB @500), divergence to **+3.37 dB peak @15K**, b1a recovery jump
   +4.74 dB (15K→20K), final +1.102; same 263 eval cameras (not an eval artifact);
   N ratio 1.1018 (only scene >1.05). A genuine trajectory difference — now confirmed
   seed-stable.
4. **No new claims from P6 alone**: P6 quantifies variance around C-01/C-03; it does
   not create a new speedup claim. Multi-seed "confirmation" wording (N-07) becomes
   available only if 1–2 hold on the subset, scoped to the subset.

## Disclosures that travel with every P6 table

- eps2d asymmetry (C-13): B1A @0.1, C0 @0.3 — the P4 bounds apply at every seed.
- Subset scope: 4 scenes, 3 seeds — not a 13-scene multi-seed cohort.
- Clean-GPU discipline per run (PUBLICATION grade only).
