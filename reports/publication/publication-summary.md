# Publication summary — renderer-internal optimization of 3DGS training under a matched benchmark

**Status: COMPLETE — all publication-phase compute finished (92 runs; 91 planned +
1 clean contamination re-run; 0 failed). P1 gates + 13-scene expansions, P2 Stage A/B,
P4, P5, P6, and the garden 3-seed pairs are all in.** Frozen sources: FINAL-30K
(headline, unchanged, self-checked at every regeneration), publication run trees
(`artifacts/publication/`), all figures/tables regenerated from aggregates.

## The claim

Under a matched benchmark (one trainer, one protocol, one cohort: 13 scenes, 30K
iterations, seed 42, COLMAP, 0.8·L1 + 0.2·(1−SepSSIM), clean-GPU scheduling), the frozen
final method **C0** trains **1.0685× faster (geomean of per-scene ratios)** than the
clean gsplat v1.5.3 + accutile baseline **B1A** — per-scene-reduction geomean **6.20%**
(the reported reduction variant; derived-from-geomean 6.41% named separately, never as
"speedup excess" 6.85%) — is faster on **13/13 scenes**, and is
**quality-neutral within the matched benchmark** (mean ΔPSNR +0.070 dB, ΔSSIM/ΔLPIPS
within noise; drjohnson +1.102 dB is a per-scene outlier, not generalized).

## Method and identity

- C0 = F9 + SCALAR_ADJOINT + H8-MR on the frozen HIGS binary (ext `9baf8655`, core
  `361b216b`), eps2d 0.3, PX=2, absgrad on. §21 research-tree freeze: no new
  optimization branches in the publication phase.
- B1A = clean gsplat v1.5.3 (`937e2991`, so `860936f2…`) + accutile, eps2d 0.1.
- B1 = pristine v1.5.3 (no accutile anywhere), eps2d 0.1.
- B0 = original Graphdeco rasterizer @`54c035f` under the matched trainer
  with its original densification statistic (single disclosed protocol difference).

## What the evidence shows

1. **Baselines (P1, COMPLETE — 13 scenes each).** B0 trains **2.42× slower** than B1A
   (13/13 scenes; the 3-scene gate's 2.20× understated it) and its original signed-grad
   densification statistic fails bidirectionally: under-densifying the unbounded
   outdoor scenes (stump −3.14, flowers −2.52, bicycle −2.19, treehill −2.10 dB) while
   over-densifying drjohnson (+1.58 dB at N 1.54×, 2.8× the wall — the same
   trajectory-sensitive scene as the R-13 outlier). The B1 gate isolated accutile:
   +2.2–4.6% wall on every scene and a real garden quality effect, sign-consistent
   across seeds 42/43/44 (+0.642/+0.806/+1.132 dB, mean +0.860; wall edge +2.48%
   mean) — B1A is not merely "B1 + speed", which the headline comparison controls for
   (both C0 and B1A arms use exact accumulation).
2. **Ablation (P2, COMPLETE — 13 scenes).** One binary, env-switched, frames
   bit-identical: F9, SCALAR_ADJOINT, H8-MR together add ~0.45% wall over the internal
   parent (geomean 1.0045, 10/13 faster) and are quality-neutral (12/13 within ±0.15
   dB; drjohnson +0.770 is the one large cell, labeled per-scene). The headline
   speedup is cross-codebase: the HIGS base execution paths carry most of it. The
   paper does not attribute the headline to the three switched branches.
3. **eps2d sensitivity (P4).** MATERIAL_CONFOUND on the quality axis (C0@0.3 vs C0@0.1:
   −0.100 dB mean, 1.07% wall). Direction: the asymmetry inflates the headline speedup
   ~1.1% (matched-eps2d ≈ 1.057×) and deflates C0 quality ~0.10 dB (matched ≈ +0.170) —
   biases oppose, conclusion robust, disclosure carried in every C0/B1A table.
4. **Multi-seed (P6, COMPLETE).** 4 scenes × {B1A, C0} × seeds {42, 43, 44}: all 12
   per-scene speedups > 1; per-seed geomeans **1.0705 / 1.0735 / 1.0561** straddle the
   headline 1.0685. Quality straddles zero on 3 of 4 scenes; drjohnson's +1.1 dB is
   **reproducible** (+1.102/+0.787/+1.597) and labeled per-scene. Without drjohnson
   the geomeans stay 1.059–1.078 with mean ΔPSNR ≈ 0. The headline is seed-robust on
   the subset; no new claim is created.
5. **External systems (P5, SYSTEM_LEVEL).** faster-gs / fastgs / speedy-splat under
   their own protocols trade quality for speed; all three collapse on garden
   (12.4–13.9 dB) **on their garden-data variant** (manually-created COLMAP, 0.5×
   downsampled DSSIM — not the matched scene), so the collapse is reported as a
   shared failure mode of approximate pipelines on garden-class content, not as a
   controlled contrast.

## Discipline

- §19 speedup terminology (geomean-of-ratios; reduction percentages separately labeled).
- §20 quality wording ("quality-neutral within the matched benchmark"; per-scene
  outliers labeled).
- Clean-GPU scheduling with contamination forensics; contaminated runs recorded, never
  dropped silently (one historical attempt preserved).
- All tables/figures derive programmatically from frozen run trees; nothing
  hand-transcribed (the aggregation engine reproduces the FINAL-30K headline exactly
  as its self-check).

## Artifact index

| deliverable | path |
|---|---|
| Headline runs (frozen) | `final30k_runs/` (mx) |
| Matched baselines | `artifacts/publication/baselines/` + `reports/publication/matched-baselines.md` |
| Ablation | `artifacts/publication/ablations/` + `reports/publication/cumulative-ablation.md` |
| eps2d 2×2 | `artifacts/publication/eps2d/` + `reports/publication/eps2d-sensitivity.md` |
| Multi-seed | `artifacts/publication/multiseed/` + `reports/publication/multi-seed-confirmation.md` |
| External systems | `aggregates/p5_external.json` + `reports/publication/external-systems.md` |
| Figures A–G | `artifacts/publication/figures/` |
| Tables 1–6 | `artifacts/publication/tables/` |
| Checkpoints | `reports/publication/checkpoint-1.md`, `checkpoint-2.md` |
