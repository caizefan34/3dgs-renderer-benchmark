# DEA-GS M0 Report

## Verdict

`DROP_INSUFFICIENT_HEADROOM`

## Baseline

- repo / commit: 3dgs-renderer-benchmark @ `32ab80e773f74f4d8e40ff4e338c86b29c7957f5`, branch `master` (verified on mx host bms-39468022-001)
- scene / resolution: `room` (mipnerf360, 311 cameras, native res), 1080p (1920x1080), data-factor 1
- GPU / CUDA / PyTorch: NVIDIA A100-PCIE-40GB (sm_80) / CUDA 12.4 / torch 2.4.1+cu124 (conda env `anysplat`); gsplat 1.5.3 rasterization(absgrad=True, packed=False, tile_size=16)
- optimizer implementation: stock `torch.optim.Adam` (one persistent optimizer; state migrated across topology via `cat_tensors_to_optimizer`/`_prune_optimizer`). NOT SparseAdam, NOT fused.
- Adam beta1 / beta2 / eps: 0.9 / 0.999 / 1e-15
- feature LR semantics: `_shs` (appearance, DC+rest combined as one `[N,16,3]` tensor) uses constant LR = `feature_lr` = 0.0025 (no scheduler); xyz uses exponential decay (1.6e-4 -> 1.6e-6, max_steps 30000); opacity 0.025, scaling 0.005, rotation 0.001 constant
- profile point: real continuation from `iter_15000.pt` (post-densification, N=933590), start-iter 15001, 2000 warmup + 500 profiled iterations; step_at_export 2500

## OBSERVED

### Baseline runtime
- total iteration ms: 62.34 ms mean (render 4.50, loss/SSIM 25.26, backward 21.66, densify 0.014, optimizer 3.70)
- optimizer ms: 3.70 ms mean
- optimizer fraction: 0.0594 (5.94%)  [immediate DROP trigger: < 15%]

### Parameter/work composition
- appearance scalar fraction: 0.8136 (`_shs` = 44,812,320 of 55,081,810 optimizer scalars)
- appearance optimizer-work fraction: 0.8136 (scalar proxy; appearance tensors are the shs group only)

### Visibility structure
- mean / median / p90 visible fraction: 0.2477 / 0.2800 / 0.3683 (N=933590, mean visible 231252)
- P(streak>=2/4/8/16/32): 0.438 / 0.217 / 0.0750 / 0.0231 / 0.00847
- p50 / p90 / p99 re-entry streak: 1 / 6 / 29 (max 224); 114,718,605 streak samples
- topology forced-materialization rate: 0 (stable post-densification window, zero topology events)
- projected appearance update work reduction: 4.037x (>= 3.0x gate PASS)

### Adam oracle
- real samples: 150,000 (>= 100k gate; real npz from iter_15000 continuation, step=2500, k p50/p90/p99/max = 4/58/500/500)
- fallback fraction: 0.1140 (> 5% gate FAIL) — driven by near-zero v (median 8e-18) with nonzero m; 45,654 exact-zero states are always certified
- theta rel err p99.99 / max: 3.88e-06 (> 1e-6 FAIL) / 1.09e-03 (> 1e-5 FAIL)
- m rel err p99.99: 9.6e-16 (<= 1e-7 PASS)
- v rel err p99.99: 6.0e-16 (<= 1e-7 PASS)
- NaN/Inf: 0
- eps_ratio p50/p99/max: 3.4e-07 / 4.5e+292 / 4.5e+292 (huge eps/sqrt(vhat) for tiny-v, nonzero-m states forces fallback)

### Speed bound
- optimistic: 1.027x (time_ratio 0.9736)
- best estimate: 1.006x (time_ratio 0.9936)
- conservative: 0.973x (time_ratio 1.0279)  [< 1.25x gate FAIL; method is net-SLOWER under conservative assumptions]

## EVIDENCE
- Host: bms-39468022-001 (MX cohort), 8x A100-PCIE-40GB. GPUs 0,2,3,4,5,7 idle/free at run time.
- Raw artifacts (pulled from mx `/mnt/storage_pool/liaoyuanjun/dea_gs_m0/room/`, preserved under `dea_gs_m0/out/raw_mx/`):
  - `raw_mx/m0_states.npz` (8,403,474 B) — real Adam state export (theta,m,v,step,k,lr,param_group,beta1/beta2/eps, meta_N/K/C/total_scalars)
  - `raw_mx/run_log.txt` — real training log (GTDataset 311 cams, continuation, per-100-iter n_vis trace, DONE summary)
  - `raw_mx/m0_profile.json` — per-phase timing + param composition + visibility summary
  - `M0_ENV.json`, `M0_BASELINE_PROFILE.json`, `M0_VISIBILITY.json`, `M0_ADAM_ORACLE.json`, `M0_SPEED_BOUND.json`, `M0_LIFECYCLE.json`, `M0_JUDGE.json`
- Training script: `dea_gs_m0/scripts/m0_instrumented_train.py` (real instrumented trainer; no optimizer-semantics modification)
- Pack scripts used verbatim: `_dsh_pack_v1/DEA_GS_M0_DSH_PACK/scripts/{adam_gap_oracle.py,judge_m0.py,speed_bound.py}`
- Reproduction: oracle and judge re-run locally from the pulled npz + JSONs; results identical (m/v p9999 differ only at ~1e-16, float64 reduction-order noise).
- Checkpoint (real theta source): `/mnt/storage_pool/liaoyuanjun/r6_baseline_ckpts/room/checkpoints/iter_15000.pt` (231,533,476 B)

## ASSUMPTION AUDIT
- A1 (appearance does not alter geometric visibility): HOLDS. `_shs` is excluded from all densification predicates (`densify_and_clone/split/prune` use `xyz_gradient_accum` viewspace grad, `scaling`, `opacity`, `max_radii2D`); visibility = `(radii>0).any(-1)` from gsplat `meta["radii"]`, a purely geometric quantity. Geometry/topology params (xyz/scaling/rotation/opacity) are NOT deferred.
- A2 (zero-gradient dense Adam): CONFIRMED. Stock Adam `.step()` updates ALL rows every iteration; invisible rows (radii==0) receive explicit-zero photometric gradient yet m/v/theta still evolve. (Not sparse/absent.)
- A3 (global-step bias correction): single persistent optimizer; `step` is a global counter (2500 at export), bias correction uses `1-beta^t`. Constant-LR shs group admits exact prefix-sum suffix form.
- A4 (topology forces materialization): all observe/copy/remap events can materialize deferred appearance beforehand — `visible_forward` (shs materialized before current-step visible render), `clone_split` (parent shs materialized to current step before `densification_postfix` copies; children get zero state via `cat_tensors_to_optimizer`, last_seen=t), `prune_compact` (discarded rows need no materialization; survivors' deferred timestamps remapped with rows), `checkpoint` (materialize-before-save hook), `evaluation` (visible rows materialized before render), `optimizer_state_remap` (cat/prune remap deferred-timestamp metadata identically). Lifecycle audit = SAFE.
- A5 (numeric underflow): fp64 reference used; v magnitude distribution logged (median 8e-18, max 1.4e-6, 45,654 exact zeros); certificate failures (tiny-v, nonzero-m) fall back to dense and are counted. NaN/Inf = 0 in certified path.

## CONCLUSION
M0 does NOT support proceeding to M1. Two independent, pre-registered failures:
1. Insufficient headroom (binding, overrides): optimizer is only 5.94% of the iteration (immediate-drop threshold < 15%); conservative projected E2E speedup is 0.973x (< 1.25x). Even with 4.04x appearance-work reduction and a safe lifecycle, the optimizer is too small a fraction of the iteration for lazy appearance updates to yield net speedup — under conservative assumptions the method is slower.
2. Numerical fidelity (independent): on real optimizer states the certified closed form exceeds tolerance — theta p99.99 rel 3.88e-06 (> 1e-6) and theta max rel 1.09e-03 (> 1e-5) — and fallback fraction is 11.4% (> 5%), driven by near-zero second-moment (v) with nonzero first-moment (m) states where eps dominates sqrt(vhat).
No thresholds were changed, no geometry/opacity optimizer was touched, no CUDA was written, no long training was run, and no rescue was attempted after the gate failed. Per protocol section 9: STOP. No M1 in this lane without a newly pre-registered hypothesis.
