# P1: matched baselines — B0 (original Graphdeco) and B1 (pristine gsplat v1.5.3)

**Status: 3-scene reproduction gates COMPLETE for both arms (PASS / PASS_WITH_FINDING);
13-scene × 30K expansions running.** Frozen evidence: `baselines/` (run trees),
`b1_gate_verdict_v2.json`, `b0_gate_verdict.json`, `tables/table1_p1_baselines.md`.

## Why these baselines

- **B0** — the original Graphdeco rasterizer @`54c035f` is the external scientific
  reference: the method the field compares against, under its own densification
  statistic (signed-grad L2-norm ≥ 2e-4, pixel space, no rescale).
- **B1** — pristine gsplat v1.5.3 (`860936f2…` so identity, no accutile kwarg anywhere)
  is the modern training baseline the final method actually builds on.
- **B1A** — clean gsplat v1.5.3 + accutile is the FINAL-30K baseline arm (Table 1).

## Wrapper policy (single disclosed protocol difference)

B0 and B1 run under the unified matched trainer (same data loading, seed-42 camera
sequence, eval grid, timing instrumentation, 30K iters, matched loss). For B0 the
densification statistic is the ORIGINAL method's — the one difference from the matched
protocol, stated in every table carrying B0. Forcing absgrad on B0 would not be B0.

## B1 gate: PASS_WITH_FINDING (wiring verified; a real accutile effect surfaced)

Protocol identity C1/C2 pass on all 3 gate scenes (metadata, so-sha, no-accutile
signature, eps2d 0.1, absgrad). Wall ±10% and N ±20% (C4/C5) pass. The garden C3
exceedance is **systematic, not noise**:

| garden | b1a | b1 | b1a − b1 | wall edge |
|---|---|---|---|---|
| seed 42 | 24.194 | 23.552 | **+0.642 dB** | +3.88% |
| seed 43 | 24.969 | 24.163 | **+0.806 dB** | +2.21% |
| seed 44 | 24.613 | 23.480 | **+1.132 dB** | +1.36% |

- **Finding: accutile (exact tile accumulation) improves garden final PSNR by
  +0.642/+0.806/+1.132 dB across seeds 42/43/44** (mean +0.860, sign-consistent) vs
  pristine gsplat, on top of a consistent wall edge (+1.4–3.9% on garden, +2.2–4.6%
  all gate scenes). B1A is not merely "B1 + speed". **3-seed confirmation complete**
  (`garden_seed3_table.json`).
- Headline unaffected: C0-vs-B1A both use accutile — the effect cancels in the matched
  comparison (garden ΔPSNR −0.032).

### B1 13-scene expansion: COMPLETE

The gate findings hold at cohort scale (`aggregates/p1_b1_vs_b1a.json`, 13 pairs):

- **Wall: B1A faster than B1 on 12/13 scenes; b1a/b1 wall geomean 0.9662** — accutile
  is worth ~3.4% across the full cohort (gate estimate +2.2–4.6% confirmed).
- **Quality: mean ΔPSNR (b1 − b1a) −0.097 dB** — accutile improves quality on average,
  garden-dominated (the 3-seed garden effect +0.642/+0.806/+1.132; the other 12 scenes
  near zero). N geomean ratio 0.9999 — accutile does not change density.
- The wall effect is uniform; the quality effect is scene-concentrated. Both disclosures
  travel with every B1A citation.

## B0 gate: PASS

All 3 gate scenes at PUBLICATION timing grade, 30K iters, clean GPUs:

| scene | B0 wall | B1A wall | B1A/B0 | PSNR B0 | PSNR B1A | ΔPSNR | N B0 / N B1A |
|---|---|---|---|---|---|---|---|
| room | 47.7 min | 18.7 min | 0.391 | 31.695 | 31.934 | −0.239 | 0.874 |
| bicycle | 56.0 min | 26.0 min | 0.464 | 24.246 | 26.435 | **−2.189** | 0.773 |
| garden | 45.7 min | 23.6 min | 0.516 | 24.391 | 24.194 | +0.197 | 0.784 |

- **B0 trains 2.20× slower than B1A on the gate scenes** (geomean of per-scene
  ratios) — the original pipeline's Python-side render loop and older kernels.
- **Bicycle −2.189 dB is the original densification statistic failing on a
  texture-heavy unbounded scene**: the signed-grad-norm ≥2e-4 criterion under-densifies
  (N only 77% of B1A). This reproduces the known absgrad motivation (Yu et al.) under a
  matched protocol — scene-dependent, reported, not hidden.
- room/garden quality land in the ±0.24 dB band with 12–22% fewer Gaussians.

### B0 13-scene expansion: COMPLETE

The gate picture holds at cohort scale and sharpens (`aggregates/p1_b0_vs_b1a.json`,
13 pairs, all clean — counter via clean re-run after one contaminated attempt,
forensics preserved in `b0_counter_contaminated_attempt1`):

- **Wall: B0 slower on 13/13 scenes; B1A/B0 wall geomean 0.4142 → B0 is 2.42×
  slower than B1A** across the full cohort (the 3-scene gate's 2.20× understated it;
  train 0.292, kitchen 0.344, drjohnson 0.355 are the worst ratios).
- **Quality: mean ΔPSNR (B0 − B1A) −0.899 dB** — but the per-scene spread is the
  finding: the original signed-grad statistic fails in BOTH directions.
  - Under-densifies the unbounded outdoor scenes: stump **−3.140** (N 1.13× but
    worse), flowers **−2.515** (N 0.77×), bicycle **−2.189** (N 0.77×),
    treehill **−2.104** (N 0.72×) — the absgrad-motivation reproduction at cohort
    scale.
  - Over-densifies drjohnson: **+1.580 dB with N 1.54×** B1A — 54% more Gaussians
    buys quality there, at 2.8× the wall. The same trajectory-sensitive scene as
    R-13. truck N 1.42× yet −1.061 dB — density alone does not buy quality there.
  - garden/playroom ≈ neutral (+0.197/+0.059).
- **Table 1 carries the full per-scene B0 columns** (wall, PSNR, N) with the §19
  orientation note; the B0-vs-B1A speed comparison is claimable under the matched
  protocol with the densification-statistic disclosure attached.

## Integration fixes (disclosed, in `make_publication_trainer.py` P1–P20)

B0: `opacities` unsqueezed to [N,1]; radii reshaped to [1,N,1] (the original rasterizer
returns 1-D [N], which collapsed the trainer's visibility mask to 0-D and crashed
densification-stat accumulation — iteration-1 failure class); `means2D` kwarg spelling;
3-tuple return. Verified by a 200-iteration trainer-integration smoke (RC=0) before the
full gate, then by the gate runs themselves densifying normally (room N 528K→755K).
B1: the pristine v1.5.3 signature has no `accutile` kwarg — the arm passes no extra
kwargs at all (the first attempt crashed with TypeError; fixed and re-run clean).

## 13-scene expansions: BOTH COMPLETE

B1 ×13 and B0 ×13 finished under the file-driven queue (91/91 planned runs accounted;
B0 counter re-run clean after one contaminated attempt, recorded with forensics).
Table 1 regenerates from the frozen run trees as they land (never hand-transcribed).
§19/§20: all wall ratios reported as geomean speedups with per-scene reduction geomean
labeled separately; quality deltas described as "within the matched benchmark".
