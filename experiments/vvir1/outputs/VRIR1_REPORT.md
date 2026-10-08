# VRIR1-P0 Report

## CONCLUSION
VERDICT: INCREMENTAL_HEADROOM_CERTIFIED

## QUESTION
When Gaussian state evolves slightly between training checkpoints, how much
raster structure (Gaussian->tile intersections) can be updated incrementally
without a full rebuild? Certified iff >=70% of structure is unaffected (G5).

## OBSERVED
- Provenance: commit abfa6c3, 1.5.3 (frozen fork b1-clean-v153), NVIDIA A100-SXM4-80GB (EPIC-05, 8x), seed 42.
- Window: 15001-30000 (densification OFF, frozen topology, identical to VRIR0).
- Tracked cameras: 8/scene; snapshot gap 500 iters; 31 snapshots/scene.
- room: N=1593376, cameras=311, t_iter=30.1ms.
    per-gap(500) unaffected agg=0.9346 (min 0.8830); full-window(14999) agg=0.7698 (min 0.6236).
    membership_model_ok_rate=1.0000.
- bicycle: N=6131954, cameras=194, t_iter=51.3ms.
    per-gap(500) unaffected agg=0.9446 (min 0.8247); full-window(14999) agg=0.7656 (min 0.6914).
    membership_model_ok_rate=1.0000.
- garden: N=5834784, cameras=185, t_iter=51.1ms.
    per-gap(500) unaffected agg=0.9588 (min 0.8669); full-window(14999) agg=0.7977 (min 0.7498).
    membership_model_ok_rate=1.0000.

## EVIDENCE
- Raw records: `experiments/vvir1/runs/<scene>_15001_30000_g500_vir1_records.jsonl`
  (one JSON object per camera x snapshot x horizon; fields: unchanged/added/removed/unaffected_uf/drift).
- Aggregates: `VRIR1_STATE_DRIFT.json`, `VRIR1_STRUCTURE_DELTA.json`.
- Timing: `experiments/vvir1/runs/<scene>_..._vir1_timing.json`.
- Membership model = VRIR0-verified tile_bounds (bit-matched to gsplat 1.5.3 per-axis radii).

## STATE DRIFT (G3)
- room @ 500-iter gap: xyz p95=2.29e-04 world-unit, means2d p95=0.070px, depth p95=1.47e-04, radius p95(x/y)=2.672/2.487px, active_union~335947.
- bicycle @ 500-iter gap: xyz p95=2.64e-04 world-unit, means2d p95=0.071px, depth p95=1.47e-04, radius p95(x/y)=1.853/1.103px, active_union~1290184.
- garden @ 500-iter gap: xyz p95=2.61e-04 world-unit, means2d p95=0.047px, depth p95=1.45e-04, radius p95(x/y)=1.103/1.047px, active_union~1880872.
  (xyz is camera-independent; means2d/depth/radius are per-camera, pooled over tracked views.)

## STRUCTURE DELTA (G4)
Intersection = (Gaussian, tile) membership. unchanged = present in both checkpoints;
added/removed = set difference. unaffected = unchanged / (unchanged+added+removed), exact.

## GATE TABLE
| Gate | Result | Evidence |
|---|---|---|
| G0 remote-only | PASS | EPIC-05 8xA100, gsplat 1.5.3 frozen fork, repo abfa6c3 |
| G1 source lock | PASS | membership_model_ok_rate=1.0 all scenes; VRIR0 tile_bounds reused verbatim |
| G2 prior art | PASS | VRIR1_PRIOR_ART.md |
| G3 state drift | PASS | VRIR1_STATE_DRIFT.json (small drift at 500-iter gap) |
| G4 structure delta | PASS | VRIR1_STRUCTURE_DELTA.json (exact unchanged/added/removed) |
| G5 economic ceiling | PASS | per-gap agg={'room': 0.9346, 'bicycle': 0.9446, 'garden': 0.9588}, full-window agg={'room': 0.7698, 'bicycle': 0.7656, 'garden': 0.7977} |

## G5 BASIS
Threshold: >= 70% structure unaffected.
- Per-gap (500-iter refresh, 'slight evolution'): {'room': '0.9346', 'bicycle': '0.9446', 'garden': '0.9588'} -> all >= 0.70.
- Full-window cumulative (15001->30000): {'room': '0.7698', 'bicycle': '0.7656', 'garden': '0.7977'} -> aggregate >= 0.70.
- Verdict basis: both the operational per-gap and the cumulative aggregate clear 0.70.

## INTERPRETATION
VRIR0 demanded EXACT cross-visit reuse and dropped (0% tile-list survival). VRIR1 asks the
complementary INCREMENTAL question and finds ample headroom: at a 500-iteration refresh
cadence, 93-96% of (Gaussian, tile) intersections are unchanged, so only ~4-7% must be
rebuilt. Even accumulated over the full 15000-iteration window, 77-80% of intersections
are unaffected in aggregate. The unchanged fraction degrades as the gap grows (see
uf_vs_gap_curve); at the longest single gaps (~15000 iters) room/bicycle per-snapshot
minima dip toward/below 0.70, so the economic ceiling holds at the operational cadence
and in aggregate, but refresh cadence must be bounded. No CUDA incremental kernel is
implemented (prereg); this certifies headroom only.

