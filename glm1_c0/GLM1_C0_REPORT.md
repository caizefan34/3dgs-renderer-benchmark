# GLM1-C0 — Cost-Weighted Headroom Audit Report

- Packet: GLM1-C0 Cost-Weighted Headroom Audit
- Host: EPIC-05 (8.130.30.251:1024), GPU: NVIDIA A100-SXM4-80GB ×8
- Timing GPU: cuda:0 (uncontended, 0% util, 4 MiB)
- Seed: 42 | Iterations: 30000 | Scenes: room, bicycle, garden
- Frozen policy: mature = (vis_frac >= 0.05) AND (age >= 3000); candidate cadence = every 2 iters
- Timing grade: FUNCTIONAL_ONLY (phases sum to mean_iter_ms)

## Verdict

**GLM1_C0_NO_COST_HEADROOM**

Per PREREG §6 fail-closed continuation gate: GLM1-C0 PROMOTE requires G1 AND G2 AND G3 AND G4 AND G5. G2 fails, so the verdict is `GLM1_C0_NO_COST_HEADROOM`. Do NOT launch a quality campaign; close GLM scheduling.

## Gate evaluation (frozen analyzer)

| Gate | Threshold | room | bicycle | garden | Pass |
|------|-----------|------|---------|--------|------|
| G1 | opt-only >= 0.02 in >=2/3 | 0.0159 | 0.0260 | 0.0312 | PASS (2/3) |
| G2 | grad-consumer >= 0.03 in >=2/3 | 0.0159 | 0.0260 | 0.0312 | FAIL (1/3) |
| G3 | backward-oracle >= 0.05 in >=2/3 | 0.0951 | 0.1116 | 0.1536 | PASS (3/3) |
| G4 | backward-oracle >= 0.03 all | 0.0951 | 0.1116 | 0.1536 | PASS |
| G5 | cand fraction >= 0.10 all | 0.8696 | 0.8638 | 0.8793 | PASS |

STRONG requires backward-oracle >= 0.08 in every scene AND geomean speedup >= 1.08. Not reached (promote=False).

## Mature fraction among visible (phase-weighted, PREREG §5)

| Scene | N | N_alive | N_visible | N_mature | f_cand_alive (iter-wt) | f_cand_visible (iter-wt) | f_tile_mature (iter-wt) |
|-------|---|---------|-----------|----------|------------------------|--------------------------|------------------------|
| room | 1,616,133 | 835,653 | 698,827 | 698,827 | 0.2246 | 0.8696 | 0.3326 |
| bicycle | 6,566,757 | 3,681,789 | 3,318,662 | 3,318,662 | 0.1868 | 0.8638 | 0.4212 |
| garden | 4,424,169 | 2,568,130 | 1,988,864 | 1,988,864 | 0.2765 | 0.8793 | 0.5579 |

Note: at the final snapshot f_cand_visible = 1.0 (every visible Gaussian is mature by end of training), but PREREG §5 mandates phase-weighted 0..30K estimates. The phase-weighted candidate-among-visible fraction is ~0.86–0.88; the phase-weighted candidate-among-alive fraction is ~0.19–0.28 (most alive rows are young/newly-born).

## Cost-weighted oracles (analytical upper bounds)

| Scene | mean_iter_ms | opt_ms | bwd_ms | removable_opt_ms | O1 | O2 | O3 | S=1/(1-O3) |
|-------|-------------|--------|--------|------------------|-----|-----|-----|-----------|
| room | 17.318 | 2.456 | 8.248 | 0.2758 | 0.0159 | 0.0159 | 0.0951 | 1.1051 |
| bicycle | 30.011 | 8.350 | 12.207 | 0.7797 | 0.0260 | 0.0260 | 0.1116 | 1.1257 |
| garden | 29.449 | 6.654 | 12.918 | 0.9200 | 0.0312 | 0.0312 | 0.1536 | 1.1815 |

Formulas: `O1 = opt_ms * f_cand_alive_iter * (1/cadence) / mean_iter_ms`; `O2 = (O1*mean_iter + G)/mean_iter` with G=0; `O3 = O1 + B/mean_iter` where `B = bwd_ms * f_tile_mature_iter * (1/cadence)` (MODELED).

## Decisive caveat: dense Adam yields ~0 realizable optimizer saving

Microbench on cuda:0 (uncontended, 0% util, 4 MiB), room checkpoint N=835,653, 5 full-tensor groups, zeroing 82% of row grads:

- full step (median): 2.711 ms
- zeroed-82% step (median): 2.705 ms
- ratio (median): 0.9975  |  ratio (min): 0.9537

The trainer uses ONE persistent dense `torch.optim.Adam` over 5 full-tensor groups (xyz/shs/opacity/scaling/rotation). Dense Adam iterates every row regardless of grad value, so row-skipping yields ~0 realizable wall-time saving unless a sparse optimizer is adopted. O1/O2 are therefore theoretical upper bounds, NOT realizable under the current optimizer.

## Interpretation

- The GLM0 signal is NOT merely visibility (G5 passes: ~86–88% of visible updates are candidates).
- But the mature signal touches very little REAL cost: the only guaranteed-removable work is the optimizer parameter update (O1=O2 ≈ 1.6–3.1% of iteration wall), because visible-mature Gaussians must still render forward, produce visibility, contribute to loss, and contribute densification stats (anti-overcount rule). G=0.
- G2 fails: grad-consumer upper bound >= 0.03 in only 1/3 scenes (garden 0.0312 passes; room 0.0159 and bicycle 0.0260 fail).
- The backward oracle O3 (9.5–15.4%) passes G3/G4, but it is an explicit MODELED upper bound that assumes mature Gaussians could skip backward accumulation — which the anti-overcount rule forbids (they still contribute to loss/gradients). It must not be read as a realizable speedup.
- This replicates LAVOS0's lesson at the optimizer level: a large update-count headroom (GLM0: 12.84% additional candidate headroom) is end-to-end irrelevant when the removable stage is a small fraction of iteration wall, AND dense Adam does not even realize the optimizer-stage saving without a sparse rewrite.

## Required outputs (PREREG §8)

- GLM1C0_ENVIRONMENT.json
- GLM1C0_SOURCE_IDENTITY.json
- GLM1C0_STAGE_TIMING.json
- GLM1C0_MATURE_WORK_FRACTION.json
- GLM1C0_ORACLES.json
- GLM1C0_SUMMARY.json
- GLM1C0_VERDICT.json
- GLM1_C0_REPORT.md
- MANIFEST.sha256

## Do-NOT (per PREREG / DSH handoff)

- Did not change age threshold 3000 / cadence 2 / per-scene thresholds.
- Did not run quality rescue; did not count forward rendering as removable without proof.
- Did not claim oracle speedup as real speedup; did not modify VOS/LAVOS/ReproTopo; did not use local GPU.
- Verdict is NO_COST_HEADROOM: close GLM scheduling and do NOT launch a full quality campaign.
