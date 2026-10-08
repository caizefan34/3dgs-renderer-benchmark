# P13-X Discovery Report

Generated: 2026-09-27T15:55:48Z

Scenes: room, bicycle


Preregistration: P13X_PREREG.md (frozen; executed as written). All thresholds fixed before results.

## X-IDENT Source identity

**OBSERVED.** P13-X trainer is a byte-derived copy of the P12 trainer with side-channel diagnostic hooks only; frozen binaries match.

**EVIDENCE.** baseline trainer sha256 675b0267f3a1a6a0...; C0 ext d26cccf46c79..., core 935ea6c0dd04..., fused-ssim dd6a4c799b98...; match_prereg=True; trainer diff reviewed (P13X_TRAINER_DIFF.patch).

**HYPOTHESIS.** Diagnostics cannot perturb training semantics.

**TEST.** Preflight A (P12 vs P13X diag-off, 220 iters) + Preflight B (diag-on vs diag-off, 300 iters, RNG footprints + camera sequence byte identity).

**CONCLUSION.** Preflight overall_pass = True.

## X0 P12 prior

**OBSERVED.** Constant-N claim verified across 13 scenes post-densification; mean visible fraction 0.2927607371070248.

**EVIDENCE.** P13X_X0_P12_PRIOR.json (from P12 artifacts, no re-run).

**HYPOTHESIS.** Post-15K training states are count-stable, enabling frozen-state probes.

**TEST.** Audit of P12 per-iteration N and visibility records.

**CONCLUSION.** Constant-N verified; X1-X5 probes operate on count-stable states.

## X1 visible stability — room

**OBSERVED.** Median visible-set Jaccard at lag 250: 0.9976091894904837.

**EVIDENCE.** 32 anchor pairs; retained=0.9986636828122812, entered=0.0011236731752441915.

**HYPOTHESIS.** H-D: the visible set is stable enough that projection work for a large retained majority could be reused.

**TEST.** Prereg gate: median Jaccard >= 0.95 pass; < 0.85 drop.

**CONCLUSION.** jaccard_pass=True (hard_drop=False).

## X2 membership/order — room

**OBSERVED.** Membership Jaccard 0.9311590968241263, median per-tile inversion rate 0.001019944531131329.

**EVIDENCE.** 32 pairs; membership-hash-unchanged tiles 0.003492647058823529, order-hash-unchanged 0.0030637254901960784.

**HYPOTHESIS.** H-D: exact (tile, gaussian) memberships and depth orders are stable enough for cached reuse of intersection work.

**TEST.** Prereg gate: membership Jaccard >= 0.95 AND inversion <= 0.02; drop < 0.85 / > 0.10.

**CONCLUSION.** pass=False (hard_drop=False).

## X3 dirty work — room

**OBSERVED.** At eps=1e-5: dirty-isect fraction 1.0, replay rel L2 0.0.

**EVIDENCE.** Per-eps medians: {"eps_0": {"dirty_isect": 1.0, "dirty_tile": 1.0, "replay_rel_l2": 0.0}, "eps_1e-06": {"dirty_isect": 1.0, "dirty_tile": 1.0, "replay_rel_l2": 0.0}, "eps_1e-05": {"dirty_isect": 1.0, "dirty_tile": 1.0, "replay_rel_l2": 0.0}, "eps_0.0001": {"dirty_isect": 1.0, "dirty_tile": 1.0, "replay_rel_l2": 0.0}}.

**HYPOTHESIS.** H-D: only a minority of intersections carry changed renderer inputs; clean work is exactly reusable.

**TEST.** Prereg gate: dirty-isect <= 0.30 on all primary scenes AND replay rel L2 <= 1e-6; drop >= 0.70 on two scenes.

**CONCLUSION.** promising=False (replay_pass=True, hard_drop_dirty=True).

## X4 tile adjoint DCT — room

**OBSERVED.** K=32: energy retention 0.463982492685318, rel L2 median 0.7685291171073914, cosine mean 0.6669420897960663.

**EVIDENCE.** 256 deterministic samples post-15K; full K sweep in P13X_X4_ADJOINT_DCT.json.

**HYPOTHESIS.** H-A: the pixel adjoint is low-dimensional per tile, so backward interactions could be compressed to a K-basis.

**TEST.** Fixed orthonormal DCT-II, JPEG zig-zag K sweep, no data-dependent top-K.

**CONCLUSION.** Informational per prereg (rasterizer Jacobians are smoothing operators; X5 carries the decision weight).

## X5 induced-gradient fidelity — room

**OBSERVED.** candidate_K=None; K=32 means cosine 0.8993455171585083, opacity cosine 0.9986988306045532; one-step delta (K=32) None.

**EVIDENCE.** 40 probes (0 errors); full sweep in P13X_X5_PARAM_GRAD_FIDELITY.json.

**HYPOTHESIS.** H-A: DCT-K reconstructed adjoints induce Gaussian gradients close enough to exact to preserve one-step training progress.

**TEST.** Prereg candidate gate at K<=32: cosine >= 0.995, rel L2 <= 0.05 (means & opacity), top-10% sign >= 0.98, one-step <= 0.5%; all conditions on all scenes.

**CONCLUSION.** candidate_K=None (drop_cosine_0.99_K<=64=False).

## X6 oracle timing — room

**OBSERVED.** E2E baseline 15.557304337620735 ms; delta speedup (conservative) 0.9573716120004906; adjoint best 2.174351619888652; composed best 1.9312417494407919.

**EVIDENCE.** Component medians {"cull_ms": 0.16679999977350235, "producer_ms": 0.11979199945926666, "isect_ms": 0.833952009677887, "raster_ms": 2.1972399950027466, "compare_ms": 0.9006800055503845, "fwd_ms": 3.525752007961273, "loss_ms": 0.5087520033121109, "bwd_ms": 8.87003231048584, "opt_ms": 2.6527680158615112}; backward split regression R^2=0.9288626225999932; dirty-isect median 1.0.

**HYPOTHESIS.** If the identified reusable/compressible work were removed in the best case, E2E would speed up by the oracle factors.

**TEST.** CUDA-event component timing on the frozen 16K state; oracles constructed only from measured components; composed oracle charges overlap conservatively (adjoint applies only to the dirty portion).

**CONCLUSION.** Standalone >= 2.0 and composed >= 4.0 required for GO (see verdict).

## X1 visible stability — bicycle

**OBSERVED.** Median visible-set Jaccard at lag 250: 0.9984598708191152.

**EVIDENCE.** 32 anchor pairs; retained=0.9991585087130957, entered=0.0007059312264335936.

**HYPOTHESIS.** H-D: the visible set is stable enough that projection work for a large retained majority could be reused.

**TEST.** Prereg gate: median Jaccard >= 0.95 pass; < 0.85 drop.

**CONCLUSION.** jaccard_pass=True (hard_drop=False).

## X2 membership/order — bicycle

**OBSERVED.** Membership Jaccard 0.9308056415742785, median per-tile inversion rate 0.0012131753078977918.

**EVIDENCE.** 32 pairs; membership-hash-unchanged tiles 0.0007352941176470588, order-hash-unchanged 0.00024509803921568627.

**HYPOTHESIS.** H-D: exact (tile, gaussian) memberships and depth orders are stable enough for cached reuse of intersection work.

**TEST.** Prereg gate: membership Jaccard >= 0.95 AND inversion <= 0.02; drop < 0.85 / > 0.10.

**CONCLUSION.** pass=False (hard_drop=False).

## X3 dirty work — bicycle

**OBSERVED.** At eps=1e-5: dirty-isect fraction 1.0, replay rel L2 0.0.

**EVIDENCE.** Per-eps medians: {"eps_0": {"dirty_isect": 1.0, "dirty_tile": 1.0, "replay_rel_l2": 0.0}, "eps_1e-06": {"dirty_isect": 1.0, "dirty_tile": 1.0, "replay_rel_l2": 0.0}, "eps_1e-05": {"dirty_isect": 1.0, "dirty_tile": 1.0, "replay_rel_l2": 0.0}, "eps_0.0001": {"dirty_isect": 1.0, "dirty_tile": 1.0, "replay_rel_l2": 0.0}}.

**HYPOTHESIS.** H-D: only a minority of intersections carry changed renderer inputs; clean work is exactly reusable.

**TEST.** Prereg gate: dirty-isect <= 0.30 on all primary scenes AND replay rel L2 <= 1e-6; drop >= 0.70 on two scenes.

**CONCLUSION.** promising=False (replay_pass=True, hard_drop_dirty=True).

## X4 tile adjoint DCT — bicycle

**OBSERVED.** K=32: energy retention 0.4217030107975006, rel L2 median 0.7441815137863159, cosine mean 0.6274644434452057.

**EVIDENCE.** 256 deterministic samples post-15K; full K sweep in P13X_X4_ADJOINT_DCT.json.

**HYPOTHESIS.** H-A: the pixel adjoint is low-dimensional per tile, so backward interactions could be compressed to a K-basis.

**TEST.** Fixed orthonormal DCT-II, JPEG zig-zag K sweep, no data-dependent top-K.

**CONCLUSION.** Informational per prereg (rasterizer Jacobians are smoothing operators; X5 carries the decision weight).

## X5 induced-gradient fidelity — bicycle

**OBSERVED.** candidate_K=None; K=32 means cosine 0.763083428144455, opacity cosine 0.9878215789794922; one-step delta (K=32) None.

**EVIDENCE.** 40 probes (0 errors); full sweep in P13X_X5_PARAM_GRAD_FIDELITY.json.

**HYPOTHESIS.** H-A: DCT-K reconstructed adjoints induce Gaussian gradients close enough to exact to preserve one-step training progress.

**TEST.** Prereg candidate gate at K<=32: cosine >= 0.995, rel L2 <= 0.05 (means & opacity), top-10% sign >= 0.98, one-step <= 0.5%; all conditions on all scenes.

**CONCLUSION.** candidate_K=None (drop_cosine_0.99_K<=64=False).

## X6 oracle timing — bicycle

**OBSERVED.** E2E baseline 30.644424200057983 ms; delta speedup (conservative) 0.9440893862825006; adjoint best 1.4741893051153543; composed best 1.3324199116501838.

**EVIDENCE.** Component medians {"cull_ms": 0.3362319990992546, "producer_ms": 0.29068800061941147, "isect_ms": 1.2140080034732819, "raster_ms": 3.3770639896392822, "compare_ms": 2.2117680311203003, "fwd_ms": 5.614943981170654, "loss_ms": 0.5088880062103271, "bwd_ms": 14.350504159927368, "opt_ms": 10.170088052749634}; backward split regression R^2=0.9466299895735191; dirty-isect median 1.0.

**HYPOTHESIS.** If the identified reusable/compressible work were removed in the best case, E2E would speed up by the oracle factors.

**TEST.** CUDA-event component timing on the frozen 16K state; oracles constructed only from measured components; composed oracle charges overlap conservatively (adjoint applies only to the dirty portion).

**CONCLUSION.** Standalone >= 2.0 and composed >= 4.0 required for GO (see verdict).

## Contamination

**OBSERVED.** Diagnostic side channels did not perturb training trajectories or timed measurements.

**EVIDENCE.** Preflight: {"A_equivalence_pass": true, "B_contamination_pass": true}. 16K companions (pre-divergence loss gates at 1e-5, camera-sequence byte identity, wall-time reconciliation extra_wall≈diag_wall, paired phase timing within run-to-run noise): {"room": {"pre_div_max_rel": 4.812379411748826e-06, "camera_seq": true, "wall_ok": true, "clean": true}, "bicycle": {"pre_div_max_rel": 2.4365291763709054e-06, "camera_seq": true, "wall_ok": true, "clean": true}}. Post-divergence trajectory differences (N, loss) are inherent atomicAdd FP nondeterminism, present between any two independent runs; the side-channel transparency itself is established by the preflight RNG-footprint and byte-identity checks.

**HYPOTHESIS.** Side-channel instrumentation must be measurement-transparent.

**TEST.** Back-to-back paired runs on one physical GPU; RNG footprint hash equality; pre-divergence per-iter loss gates; diagnostics wall time reconciled against total wall time (structural exclusion from phase timing).

**CONCLUSION.** clean=True.

## Verdict

**OBSERVED.** **DROP**

**EVIDENCE.** {
  "verdict": "DROP",
  "scenes_evaluated": [
    "room",
    "bicycle"
  ],
  "protocol_note": "DSH-task execution scale: probes ran on the listed scenes only; lags 1000/2500 outside the 15K-16K diagnostic window are unmeasured protocol truncations, recorded as such.",
  "H_D_delta": {
    "viable": false,
    "x1_pass": true,
    "x2_pass": false,
    "x3_promising": false,
    "standalone_speedup_cons_min": null,
    "standalone_min_required": 2.0
  },
  "H_A_adjoint": {
    "viable": false,
    "candidate_K": null,
    "candidate_K_per_scene": {
      "room": null,
      "bicycle": null
    },
    "standalone_speedup_cons_min": null,
    "standalone_min_required": 2.0
  },
  "composed": {
    "speedup_cons_min": 1.3324199116501838,
    "min_required": 4.0,
    "requires_both_hypotheses_viable": true
  },
  "hard_drop_reasons": [
    "X3 dirty-isect>=0.70 on >=2 scenes: ['room', 'bicycle']",
    "X5 adjoint-fidelity hard-drop"
  ],
  "x3_replay_fail": false,
  "go_conditions": {
    "standalone_ok": false,
    "composed_ok": false,
    "no_quality_gate_failed": false
  },
  "contamination_clean": true,
  "preflight_pass": true
}

**HYPOTHESIS.** H-D (delta execution) and/or H-A (tile-basis adjoint) can plausibly deliver >3x E2E over the frozen P12 C0+fused-SSIM baseline.

**TEST.** Frozen prereg gates X1-X6 evaluated without post-hoc adjustment; GO requires standalone >= 2.0x, composed conservative >= 4.0x, and no failed quality/correctness gate.

**CONCLUSION.** verdict=DROP. No >3x E2E claim is made unless the X6 conservative composed oracle >= 4.0x (currently 1.3324199116501838).

