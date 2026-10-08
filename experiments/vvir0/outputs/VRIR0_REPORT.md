# VRIR0 Report

## CONCLUSION
VERDICT: DROP

## OBSERVED
- Provenance: commit abfa6c3e4b23, branch r6-exact-backward-screening, gsplat 1.4.0, GPU NVIDIA GeForce RTX 5070 Laptop GPU, seed 42.
- Source-audit result: PASS (gsplat 1.4.0 vs pack-assumed 1.5.3_fork; version deviation documented, AABB membership semantics identical).
- Scenes/windows: room 15001-30000 (full); bicycle/garden 15001+ (diagnostic window).
- Same-view revisits: 11424 total across scenes.
  - room: revisits=2639, full-view hit(C_ORD_SLACK_N accept)=0.0000, C_MEM=0.0000, tile-local(truth-same frac)=0.0000, truth_same=0.0000, false_accepts=0, false_rejects=0.
  - bicycle: revisits=4298, full-view hit(C_ORD_SLACK_N accept)=0.0000, C_MEM=0.0000, tile-local(truth-same frac)=0.0000, truth_same=0.0000, false_accepts=0, false_rejects=0.
  - garden: revisits=4487, full-view hit(C_ORD_SLACK_N accept)=0.0000, C_MEM=0.0000, tile-local(truth-same frac)=0.0000, truth_same=0.0000, false_accepts=0, false_rejects=0.
- False accepts: 0 (hard requirement G6: 0).
- False rejects: 0.
  - room cache footprint: 1082.3 MB/scene (8 GiB cap).
  - bicycle cache footprint: 3946.7 MB/scene (8 GiB cap).
  - garden cache footprint: 3795.3 MB/scene (8 GiB cap).
- Baseline T_struct / T_iter: T_struct proxied by T_fwd (forward pass); see Amdahl below. Optimistic, zero-cost-certificate, ignores rasterization-removable assumption.

## EVIDENCE
- Raw records: `experiments/vvir0/runs/<scene>/vir0_records.jsonl` (one JSON object per tracked-camera visit).
- Analysis: `experiments/vvir0/outputs/vir0_analysis_<scene>.json`.
- Provenance: `experiments/vvir0/outputs/P0_PROVENANCE.json`, `P1_SOURCE_AUDIT.json`.
- Membership-model self-validation (tile_bounds vs fresh flatten_ids/offsets) recorded per-record as `membership_model_ok`; rate = 1.0000 (min across scenes).
- Diagnostic breakdown (300-iter gap, room cam 177): membership change 18.25% of active gaussians; tile-local unchanged 0.0%; n_isects +2.9%. Source: `experiments/vvir0/_debug2_out.txt`.

## HYPOTHESIS
VRIR0 predicted: when a training camera is revisited while Gaussian parameters evolve, the discrete raster structure (Gaussian->tile membership + per-tile depth order) remains exactly unchanged, so a cheap fail-closed certificate could certify reuse of cached flatten_ids/isect_offsets.

## TEST
- Certificate: CONFIG & TOPOLOGY(epoch) & NUMERIC(active finite, positive depth) & MEMBERSHIP(exact per-Gaussian tile AABB equality) & ORDER(exact O(M) adjacency scan AND O(N) cached key-slack theorem). Accept = all pass. No approximate equality.
- Tracked-camera policy: deterministic index quantiles (8/scene incl. endpoints), frozen RNG sequence advanced through start_iter-1 iters to match baseline order.
- Topology: densification OFF in 15001-30000 -> frozen epoch 0; cache entries with differing epoch invalid.
- Truth comparison: cached vs freshly rebuilt flatten_ids + isect_offsets (exact); truth_same and per-tile fraction computed independently of the certificate.

## GATE TABLE
| Gate | Result | Evidence |
|---|---|---|
| G0 provenance | PASS | P0_PROVENANCE.json |
| G1 semantics | PASS | P1_SOURCE_AUDIT.json; membership_model_ok_rate |
| G2 topology | PASS | epoch-0 frozen window |
| G3 numeric | PASS | false_accepts=0; no NaN rejects |
| G4 membership | C_MEM={'room': 0.0, 'bicycle': 0.0, 'garden': 0.0} | per-scene C_MEM_rate |
| G5 order | C_ORD_SLACK_N={'room': 0.0, 'bicycle': 0.0, 'garden': 0.0} | per-scene order rates |
| G6 soundness | PASS | false_accepts=0 |
| G7 Amdahl | FAIL | S_UB per scene {'room': 1.0, 'bicycle': 1.0, 'garden': 1.0} |
| G8 hit rate | FAIL | C_MEM={'room': 0.0, 'bicycle': 0.0, 'garden': 0.0}, tile-local={'room': 0.0, 'bicycle': 0.0, 'garden': 0.0} |
| G9 cache | PASS | cache bytes vs 8 GiB cap |

## INTERPRETATION
Soundness is perfect (0 false accepts; membership model validated against fresh structure on every visit). However the discrete structure does NOT survive canonical training evolution in the 15k-30k window: over the ~311-iter mean revisit gap, ~18% of active Gaussians change tile AABB (radius+center drift) and within-tile depth order changes, so no tile's ordered list is preserved (tile-local truth-same = 0). Full-view certified hit rate is therefore 0, and the optimistic Amdahl upper bound S_UB = T_iter/(T_iter - 0*T_struct) = 1.0 < 1.05. G7 (opportunity) and G8 (hit rate) fail -> DROP.
