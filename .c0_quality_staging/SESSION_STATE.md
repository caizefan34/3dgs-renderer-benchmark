# C0-QUALITY-STABILITY0-MX — session state

Date started: 2026-10-08 (round 8)
Goal: judge whether proposed C0 modules preserve full-training quality across seeds (A0 vs C0, same higs-dynamic substrate).

## Host / env (verified)
- MX host: `ssh mx` (config in ~/.ssh/config) → 36.140.146.31:26372, user liaoyuanjun, hostname bms-39468022-001
- Python: /mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python (3.10.21)
- Data: /mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/{room,bicycle,garden}
- GPUs 0-5: FOREIGN jobs (do NOT touch). GPU 6,7: CLEAN (0 MiB). Use 6/7 only.
- /dev/shm: 204G free. /mnt/storage_pool: 99% full (40G) → run under /dev/shm like causal ladder did.

## Frozen causal-ladder campaign (the template + seed42 source)
Path: /mnt/storage_pool/liaoyuanjun/mx_h8/incoming/derived/C0_SAME_PARENT_CAUSAL_LADDER_MX
- Trainer: mx_h8/incoming/derived/mx_h8_trainer.py (--arm a0/c0/a1/a2, --seed, --eps2d, --gpu, --lpips, --checkpoints-lite, --timing-grade PUBLICATION, --final-eval all)
- ONE .so for both arms: C0_SO=9baf8655f859f3e0f456e99a2d8e5b3ddfe414fe17a1a3e9d8072e0afed95b91 ; CORE_SO=361b216bcc11609a0ebb8fb44ad2e0c6170948112b6294e85123df45541c8c98
- worktree git 77ab983f (dirty 25 lines), torch 2.9.1+cu128, cuda 12.8
- arm→env (bootstrap_a/bootstrap_c0): a0={HIGS_DISABLE_F9:1}, c0={HIGS_BWD_SCALAR_ADJOINT:scalar_adjoint, HIGS_BWD_H8_MR:1}; both HIGS_PX_RUNTIME=2, HIGS_BWD_ABSGRAD=1
- scheduler: scripts/causal_scheduler.py (pre-run GPU idle wait + 15s foreign-pid monitor + contamination flag). I adapt this.

## seed42 REUSE (verified intact, all SUCCESS / contaminated=false / PUBLICATION)
raw/{room,bicycle,garden}_s42_{a0,c0}/ : quality.json, results.json, timing.json, training_curve.csv, final_status.json
checkpoints/*.ckpt_lite_30000.pt present
seed42 A0→C0(A3) ΔPSNR: room -0.035, bicycle -0.085, garden -0.496, mean -0.2056 (QUALITY_DEGRADATION)
room wall: A0 1005.5s, C0 995.28s

## Frozen quality bands (from _mx_h8 campaign, fixed before runs — the gate family)
dPSNR_floor=-0.05 dB, dSSIM_floor=-0.0005, dLPIPS_ceil=0.001, catastrophic_pair_dPSNR=-0.5
(goal says: if no frozen LPIPS threshold → LPIPS descriptive only; do not invent)

## C0 matched replication (B1A vs C0, 18 runs, seed42/43/44)
/mnt/storage_pool/liaoyuanjun/MX_C0_MATCHED_REPLICATION — BUT its c0 arm ran on gsplat/H8 substrate, NOT higs-dynamic. Different substrate. Do NOT confuse with A0/C0.

## Plan status
6 prereq gates PASS. Campaign dir + identity/gate/plan/reuse JSONs built locally + pushed to MX. Run plan frozen.

## LAUNCH (2026-10-08 ~02:23Z)
- MX campaign dir: /mnt/storage_pool/liaoyuanjun/C0_QUALITY_STABILITY0_MX
- Local: C:\Users\36570\3dgs-renderer-benchmark\C0_QUALITY_STABILITY0_MX
- Runs under /dev/shm/c0_quality_stability/runs/{scene}_s{seed}_{arm}
- Scheduler1 (GPU6 streams s43_room,s43_garden,s44_bicycle): PID 3671149, log logs/scheduler_seed4344.log, index RUN_INDEX_SEED4344.json
- Scheduler2 (GPU7 streams s43_bicycle,s44_room,s44_garden): PID 3671781, log logs/scheduler_seed4344_gpu7.log, index RUN_INDEX_SEED4344_gpu7.json
- 2 parallel schedulers; both GPUs confirmed training (GPU6 room/s43/a0, GPU7 bicycle/s43/c0 @iter500 loss0.18)
- On MX: QUALITY_SOURCE_IDENTITY/PROTOCOL_IDENTITY/SEED42_REUSE/FROZEN_GATES/RUN_PLAN/RUN_STATUS
- Autonomous watcher PID 3689557 (scripts/watch_and_aggregate.sh, log logs/watch_and_aggregate.log) polls every 120s; when sched+trainers both exit it runs scripts/aggregate_quality.py producing QUALITY_VERDICT.json etc.
- Aggregator scripts/aggregate_quality.py ready (paired bootstrap seed=12345, 2000, matches replication). SYNTAX_OK on MX.
- Progress check scripts: check_progress.sh, check_curves.sh, check_rundir.sh, check_stdout.sh
- Trainer advancing fine: room/s43/a0 @iter15000 PSNR30.45; bicycle/s43/c0 @iter12000 PSNR22.94 (2026-10-08 02:33Z)
- Next: check watch_and_aggregate.log for "VERDICT:" after ~2-3h when 12 runs done

## PIPELINE VERIFIED (02:47Z)
- Full post-run chain works end-to-end on partial data (idempotent):
  run_pipeline.sh = aggregate_quality.py -> first_divergence.py -> build_derived.py -> build_report.py
- Fixed 2 None-format bugs (aggregate print, build_derived H1 f-string).
- KEY FINDING (seed42, complete pairs): first population divergence = **iter 600** (the FIRST densification event) for ALL 3 scenes. A0/C0 trajectories diverge immediately at first densify, not late. Strong support for H2 (FP reassociation flips discrete densify decisions).
- Manifest self-verify 19/19 PASS on partial run; will re-verify on full 9-pair data.
- On completion: watcher auto-runs run_pipeline.sh; then read QUALITY_VERDICT.json + QUALITY_REPORT.md; update KB (D:\学习 not present on this host — use C:\Users\36570\3dgs-renderer-benchmark for KB) + final goal-format response.
