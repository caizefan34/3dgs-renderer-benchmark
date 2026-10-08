# SecantGS S0 — Execution Plan (remote GPU only)

Date: 2026-09-30  |  Host: EPIC-05 (Aliyun, 8x A100-80GB)  |  Env: /root/miniforge3/envs/gsplat/bin/python

## GOAL (hard constraint)
Execute SecantGS S0 per _packet_sg0/S0_PREREG.md (FROZEN). ALL GPU work on remote A100. NEVER local GPU.

## IDENTITY GATE — PASS (verified remote)
- renderer ext  .so sha256 = d26cccf46c7932bfb5fd44ee05e553af27e17c93eaff6dacf3197258a513f11  (= prereg C0 extension) OK
- renderer core .so sha256 = 935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e (= prereg core) OK
- checkpoints present:
    room    /root/epic05_n1/runs/n1snap_c0f_s42_room/ckpt_10000.pt sha256=9aa00f0ddf28b6c097e3dc8f41dbb107cd9bf175750503dc330aa1d6ab579fc4
    room    .../ckpt_15000.pt sha256=b276b338f5a444d8ea1be390d21fd3e67679b7d862652f0babd131db9434e5dd
    bicycle .../n1snap_c0f_s42_bicycle/ckpt_10000.pt sha256=26bf0b959f10f0ce526c496d0dc7ca8e62b0ba93efc83d58b7f9440d58c2f4e9
    bicycle .../ckpt_15000.pt sha256=273971d3867c1d2a08b7246803265539d74756b8eca4e3f01c6a8d149b7fd6f7
- torch 2.9.1+cu128, cuda True, 8 GPUs. a0_lib.bootstrap_c0() OK.
- Runtime activation (non-interactive ssh): 
    export LD_LIBRARY_PATH=/root/miniforge3/envs/gsplat/lib:/root/miniforge3/envs/gsplat/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
    export PYTORCH_ALLOC_CONF=expandable_segments:True

## FROZEN SEMANTICS (matched to campaign authority)
- Objective: L = 0.8*L1 + 0.2*(1-SSIM) fused-ssim padding=same train=True, full-res 1080p. (a0_lib.train_loss)
- Renderer arm c0: rasterize_gaussian_higs_dynamic, eps2d=0.3, absgrad, sh_degree=3. (a0_lib.render_c0)
- Optimizer: native Adam, per-group lr/eps/betas READ FROM FROZEN ckpt optimizer_state_dict. Topology frozen (no densify). (sg0 build_optimizer/native_adam_rollout)
- Displacement = ACTUAL executed Adam param delta theta_t - theta_{t-1} in RAW param tensors.
    groups: xyz[N,3], scaling[N,3](log), rotation[N,4](quat RAW), opacity[N,1], shs[N,16,3] FULL SH.
- Inner product (optimizer-unit), per group g: <u,v>_W = sum_g <u_g/lr_g, v_g/lr_g>.
    lr_g = frozen per-group lr at START checkpoint. Recorded per step (constant over 16-step local rollout; if trainer decays, use per-target-step lr for target+basis and record).
    NO data-derived whitening.
- Candidate state: theta_proj = theta_{t-1} + delta_proj (raw tensors). NO optimizer moments changed. NO backward.
    If native trainer normalizes a param post-Adam, apply same deterministic normalization (C0 does NOT normalize rotation/scales post-step; activated() applies exp/sigmoid at render). -> no extra normalization.
- Camera protocol = CORRECTED true-holdout (CVT0_PREREG §3-4, authority):
    TRUE HOLDOUT = { i : i%8==0 } ; TRAIN POOL = { i : i%8!=0 } ; disjoint.
    8 train panel + 8 holdout panel, each midpoint-stratified from its ordered pool:
      q_k = floor((k+0.5)*N/8); advance on duplicate; camera = pool[q_k].
    Computed panels (floor midpoint quantile per CVT0 sec4, VERIFIED on remote):
      room    train=[20,59,98,137,175,214,253,292] holdout=[16,56,96,136,168,208,248,288]
      bicycle train=[12,36,60,84,109,133,157,181] holdout=[8,32,56,80,112,136,160,184]
- Rollouts: 4 per cell, offsets {0,2,4,6} cyclic over 8 train cams, 16 steps each, from identical deep-copied ckpt+opt.
    After each step record: source cam, src loss before/after, delta_t, peer mean train loss (other 7), true-holdout mean loss/PSNR/SSIM. Eval forwards are no_grad, do NOT touch optimizer.
- Oracle subspaces m in {2,4}. For target step t (t>=m+1): basis D_t=[delta_{t-m}..delta_{t-1}], target delta_t.
    c* = argmin ||D_t c - delta_t||_W^2 + lambda||c||^2, lambda=1e-8*trace(G)/m, G_ab=<d_a,d_b>_W.
    trace(G)=0 -> probe invalid. delta_proj = D_t c*.
- Eligibility: t in [m+1 .. 16]. m=4 -> steps 5..16 (12 targets) x4 rollouts = 48 probes/cell. m=2 -> steps 3..16 (14) x4 = 56.
- Verdict thresholds: analyzer s0_analyze.py (packet). Need >=32 valid source-descending probes/cell per m.

## METRICS
- E_span = ||delta_proj||_W^2 / ||delta_t||_W^2 (report residual energy too).
- P_src: D_native_src = L_src(theta_{t-1})-L_src(theta_t); D_proj_src = L_src(theta_{t-1})-L_src(theta_proj);
    when D_native_src>0: P_src = D_proj_src/D_native_src.
- peer/trueholdout: projected-vs-native mean loss deltas; true-holdout PSNR(proj-native); validity.
- Timing ceiling (descriptive): phase fractions fwd/loss/bwd/opt; T(q)=T_full+(q-1)*T_fwd_val_like. NOT a speedup claim.

## OUTPUTS (13) -> remote /root/epic05_s0/cells + /root/epic05_s0/out
S0_SOURCE_IDENTITY.json, S0_ENVIRONMENT.json, S0_CAMERA_SEMANTICS.json, S0_ROLLOUTS.json,
S0_UPDATE_GRAMS.json, S0_ORACLE_PROJECTIONS.json, S0_LOSS_PROBES.json, S0_TRUE_HOLDOUT.json,
S0_TIMING_CEILING.json, S0_SUMMARY.json, S0_VERDICT.json, S0_REPORT.md, raw logs, SHA256 manifest.

## STEPS
1. Author s0_lib.py (bootstrap, corrected cameras, evaluate, native_adam_rollout_capture, W-inner-product, oracle temporal projection, energy, reconstruct-to-raw).
2. Author s0_cell.py (per scene@ckpt: 4 rollouts, oracle m in {2,4}, metrics, per-cell json).
3. Author s0_finalize.py (aggregate -> S0_SUMMARY in S0_SCHEMA; run packet s0_analyze.py -> S0_VERDICT; S0_REPORT.md; manifest).
4. Push to remote /root/epic05_s0. Smoke test 1 cell tiny (m=2 only, 2 rollouts) to validate identity + pipeline + no NaN.
5. Full run: 4 cells (room/bicycle @10K/15K), m in {2,4}, 4 rollouts each. Parallel across GPUs via launch script.
6. Finalize, pull artifacts, write CLOSED_DIRECTIONS entry if DROP.

## GUARDRAILS (prereg §12)
No m>4 rescue, no per-scene m, no L-BFGS reinterpretation, no learned predictor in S0, no sketch tuning before oracle passes,
no camera-sequence changes, no full training. DROP closes the temporal-span mechanism. If PROMOTE: STOP, return evidence, do NOT auto-implement S1.
Analyzer owns verdict.
