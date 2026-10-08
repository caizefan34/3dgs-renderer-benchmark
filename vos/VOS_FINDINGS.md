# VOS-GS — Durable Session Findings

Research program: VOS-GS (Virtual Optimizer State for Gaussian Splatting) on the frozen C0
baseline (F9 + SCALAR_ADJOINT + H8-MR, PX=2, fused SSIM, loss = 0.8 L1 + 0.2 (1-SSIM)).
Mission: consumer-driven exact optimizer-state virtualization. Dense-Adam semantics +
sparse physical memory traffic. This file records durable, verified facts (the
evidence_record tool is broken in this environment — same issue as EPIC-05 SESSION_FINDINGS.md).

## Environment (verified 2026-10-XX session)

- **EPIC-05** is the execution host: `ssh EPIC-05` → 8× A100-SXM4-80GB (all idle at session
  start). mx is INACCESSIBLE (account password expired, "Password change required but no
  TTY available"). Local Windows RTX 5070 8GB is too small for 3M-Gaussian training.
- Frozen trainer (current C0): `/root/epic05_p13x/publication_trainer_p13x.py`
  (P12-identical when `--p13x-diag 0`; `--loss fused` = P11-validated fused-ssim
  rahul-goel/fused-ssim @ a7c48d6d, padding="same", train=True; L = 0.8*L1 + 0.2*(1-SSIM);
  eval SSIM metric stays SepSSIM). Python: `/root/miniforge3/envs/gsplat/bin/python`.
- Trainer paths (EPIC-05): REPO=/root/3dgs-renderer-benchmark (13 mipnerf360 scenes at
  data/datasets/mipnerf360/), C0_WT=/root/epic05_frozen/higs_c0_final30k_worktree,
  C0_SO=/root/epic05_build_cache/c0_ext/experimental_gaussian_render_inference_scene_cuda/...so,
  CORE_SO=/root/epic05_build_cache/core/gsplat_cuda/gsplat_cuda.so,
  SCENE_CUDA_DIR=/root/epic05_build_cache/scene/gsplat_scene_cuda.
- Trainer invocation: `--arm c0 --loss fused --gpu 0 --eps2d 0.3 --iterations N --lpips
  --checkpoints --final-eval subset --timing-grade FUNCTIONAL_ONLY` (seed default 42).
- `--checkpoints` saves ckpt ONLY at final iteration, WITH `optimizer_state_dict` (m/v +
  per-tensor step counter). D0 checkpoints (d0_ckpt_5K/15K/30K in
  /root/epic05_d0/runs/TRACE_{room,bicycle}) have NO optimizer state (params only) —
  5K/15K m/v states need fresh short runs.
- D0 campaign (2026-09-28, /root/epic05_d0) already built the traced-trainer pattern:
  metadata-only hooks, trace-neutrality gates, UID lineage. My VOS trace driver uses a
  ZERO-source-diff wrapper (patch gsplat.experimental.rasterize_gaussian_higs_dynamic at
  runtime; capture (radii>0) packed bitmap per grad-enabled call).

## Frozen C0 source semantics (verified from source)

1. **Optimizer** (baseline/reference_v1/gaussian_model.py L227): `torch.optim.Adam(param_groups,
   lr=0.0, eps=1e-15)`, default betas (0.9, 0.999), weight_decay=0, amsgrad=False. DENSE over
   all N rows, all 5 attrs: xyz[N,3], shs[N,16,3], opacity[N], scaling[N,3], rotation[N,4].
   ONE persistent optimizer; state migrated across topology changes (cat/prune).
2. **LR schedule** (gaussian_model.py L229-257): xyz lr = expon schedule f(iteration) with
   delay (updated every iteration via update_learning_rate); shs/opacity/scaling/rotation lr
   CONSTANT per config. Replay of deferred step s must use lr(s) for xyz; constants for others.
3. **Trainer loop** (publication_trainer.py L464-570): per iteration — update_lr; oneupSHdegree
   every 1000; render (C0 dynamic path); L1+fused-SSIM loss; backward; max_radii2D update +
   add_densification_stats for visible rows only; densify_and_prune every 100 iters (500<t<15000);
   reset_opacity every 3000 iters (t<15000); optimizer.step(); zero_grad(set_to_none=True).
   `visibility_filter = (radii > 0).any(dim=-1)` where radii = densification_info["radii"][0] —
   EXACTLY the culling visible_ids set (scattered into a full-N zeros tensor).
4. **Culling** (c0 gaussian_inference.py L1789-1807): dynamic path uses `_cull_visible_cached`
   with `cull_refresh_interval` DEFAULT 1 (trainer does not override) → EVERY iteration runs
   `_cull_gaussians_batched` = `fully_fused_projection` on FULL-N master means/quats/scales
   (NO opacity, NO colors), under no_grad. Culling reads θ_xyz, θ_rot, θ_scale of ALL rows
   every iteration; does NOT read θ_opacity or θ_SH.
5. **F9 gatherless producer** (L1809-1813, L2058-2085): when f9_enabled (C0 default),
   N = visible_ids.numel(); `higs_gatherless_projected_producer(visible_ids, means, quats,
   scales, opacities, colors_input, ...)` reads master rows ONLY through visible_ids
   (visible subset). θ_opacity and θ_SH of invisible rows are NOT read by the forward.
6. **Backward** (L2378-2435): 5 full-N `torch.zeros_like` master grad buffers per iteration
   (grad_means/quats/scales/opacities/colors) + full-N means2d/absgrad proxies (L2442-2458);
   `higs_rasterize_backward` resolves master rows through visible_ids only — "F9-1 C++ VJPs
   resolve compact projected rows through visible_ids and directly load these master
   parameters". Invisible rows get grad EXACTLY 0.0 (zeros_like + scatter of visible rows).
7. **Densification consumers** (gaussian_model.py): densify_and_prune reads
   xyz_gradient_accum/denom (all rows), get_opacity (ALL rows), get_scaling (ALL rows),
   max_radii2D (ALL rows); clone/split read θ of selected rows; `_prune_optimizer` reads
   m/v[valid_mask] for ALL param groups; `cat_tensors_to_optimizer` appends zero m/v for
   new rows. reset_opacity reads get_opacity (ALL rows) and ZEROES m/v for all opacity rows.
8. **Stage timing** (reports/higs/training-stage-representativeness.md, MEASURED on A100-PCIE
   40GB, 1080p): optimizer stage at 30K = room 3.28ms (15.8%), bicycle 10.95ms (21.7%),
   garden 8.83ms (23.0%) of iteration. Visible fraction (radii>0) at 30K: room 0.302
   (N=933,590), bicycle 0.225 (N=3,957,041), garden ~0.08 (N=2,610,559). N flat 15K→30K.
9. **Byte model** (per row, fp32): params 59 floats (3+48+3+4+1); D0 uses 708 B/row for
   θ+m+v (59×4×3). Adam per-iter traffic ≈ read grad+m+v+θ (944 B) + write m+v+θ (708 B)
   = 1652 B/row; grad zero-fill 236 B/row; culling reads 40 B/row (xyz/rot/scale θ only).
   SH+opacity = 49/59 = 83.1% of Adam elements.
10. **E2 prior result** (reports/higs/e2-p0-vjp-adam-opportunity.md): VJP→Adam fusion DROP
    (net <1% on 2/3 scenes at 2048px context). E2 did NOT reduce optimizer traffic for
    invisible rows — different mechanism from VOS. Optimizer share numbers above stand.

## Dependency classification (Phase-2 conclusion, source-proven)

- xyz, rotation, scaling: **EAGER_REQUIRED** — culling full-N projection consumes θ every
  iteration (fact 4). Their m/v are consumed by Adam + topology events only, but θ must be
  exact every step → m/v must be physically updated every step under step-replay exactness
  (closed-form replay could change this; out of scope v1).
- opacity: **EVENT_MATERIALIZABLE** — θ consumed only by producer/raster (visible rows) +
  densify (opacity prune mask, all rows) + reset (all rows); fences: densification,
  opacity reset, checkpoint/export, eval.
- SH: **EVENT_MATERIALIZABLE** — θ consumed only by producer SH-eval (visible rows) +
  clone/split (selected rows); fences: densification, checkpoint/export, eval. No
  opacity-reset fence (reset touches opacity attr only).
- 15K→30K (Phase-1 region): NO densification, NO opacity reset (both gated t<15000) →
  only fences are eval + final checkpoint. Topology frozen.

## Phase-1 prior art (COMPLETE, 2026-09-29 — vos/VOS_PRIOR_ART.md FINAL)

- **Verdict: NO_COLLISION** on "per-row timestamp + exact replay of skipped stock
  dense-Adam transitions before the next parameter consumer". arXiv API sweep
  (13 exact-phrase queries) clean; web_search tool broken (auth), Semantic
  Scholar 429-blocked, Google Scholar unreachable — residual full-text/Scholar
  gap documented in Caveat 1.
- **BOGausS** (EUSIPCO 2025, local manuscript) = closest public art: per-Gaussian
  step counter "lifespan" for bias correction in a MODIFIED sparse-Adam; NO
  replay, NO dense-equivalence. Per-row-timestamp idea is NOT novel vs BOGausS.
- **AdamW-GS** (arXiv 2601.16736v5, ICLR 2026, "A Step to Decouple Optimization
  in 3DGS") — RESOLVED + full-text-checked: names "Implicit Update" (zero-grad
  moment rescale + attribute move under dense Adam) and REMOVES it via Sparse
  Adam + re-designed optimizer; 0 occurrences of replay/catch-up/lazy/materialize.
  Modify-the-optimizer family; no collision.
- **SkipGS** (arXiv 2603.08997): backward-pass gating on plateaued views;
  trajectory-changing execution skip; no collision.
- **INTERNAL DUPLICATION**: the exact VOS mechanism exists in-repo —
  experiments/training_solver_design/lazy_adam_oracle.py (per-row last_step +
  exact zero-grad replay w/ per-step lr, 4.44e-16 vs torch Adam FP64) +
  VISIBILITY_LAZY_ADAM_OPPORTUNITY.md (consumer-driven materialization design).
  Its opportunity gate FAILED for FastGS-Big (Adam 22.27% of step, 11,292 calls
  in 4,099→35K — late-stage 32/64-step cadence; even erasing all Adam = 18.6s vs
  29.06s wall gap). VOS differs: C0 target calls Adam EVERY iteration (30K
  calls), and v1 defers ONLY SH+opacity (no visibility coupling → the FastGS
  geometric-envelope failure — 401 radii>0 flips after 8 zero-grad steps on xyz —
  does NOT apply, because C0 culling never reads SH/opacity).

## Phase-4 visibility trace (COMPLETE 2026-09-29, EPIC-05 /root/vos_traces)

- 9 runs DONE: T30 traced (room/bicycle/garden) + R15/R5 ckpt runs ×3 scenes.
  **Trace-neutrality validated end-to-end**: T30 PSNRs match P12 references
  (room 32.08 vs 32.11, bicycle 26.36 vs 26.37); wall 37.6/47.5/48.1 min.
  Trace sizes: room 2.63GB, bicycle 10.95GB, garden 8.40GB (30K iters, packed
  bitmaps). All ckpts present: T30 ckpt_30000.pt (WITH m/v; 606MB/2.63GB/1.84GB),
  R15 ckpt_15000.pt, R5 ckpt_5000.pt (relaunched with --checkpoints after the
  first R runs missed the flag).
- **First gap stats (room)**: at 30K stage N=841,806, vis_frac=0.267,
  reactivations/iter=142,169 (63% of active rows are reactivations!), gap
  p50=2, p90=8, p99=37, max=491, never-visible-in-window 0.02%,
  sumK/iter=598,334. At 5K: N=458,498 vis=0.288; at 15K: N=825,032 vis=0.269.
  → Real gaps are SHORT: VOS replay work sits in the cheap microbench regime.
- Row-identity rule: densify events every 100 iters in 1..15000 break identity →
  gap stats there are per-100-iter-stretch, left-censored; 15001..30000 is
  identity-stable (fences: eval at 20000/25000/30000).

## Phase-6 microbench (first run, EPIC-05 GPU5, synthetic gaps)

- Fused dense Adam kernel (single pass, all N): 1693-1698 GB/s at N=830K/2.56M
  (room 0.659ms, garden 2.03ms) — memory-bound at ~peak; bicycle N=3.67M was
  450 GB/s in this run (CO-TENANT/interference suspect — rerun needed).
- VOS indexed kernel: K=0 gather/scatter floor 0.117ms @ ~1060 GB/s (108K rows).
  K-scaling slope REGIME CHANGE: ~0.5-1.0 ps/elem-step for K≤200, ~4-5 ps/elem-step
  for K≥500 (4-5× jump between K=200 and K=500 — cause unresolved: clock throttle
  vs cache; verify with locked clocks via nvidia-smi -lgc).
- Per-step replay cost 0.5-1.0ps (low-K regime) vs dense per-step 16.5ps (28B @
  1693GB/s) → replay is ~16-30× cheaper per elem-step than the dense step it
  replaces in the relevant gap regime.
- Synthetic-gap scene-level results were SKEWED by the lognormal tail (mean 66,
  p99~400, max 15000): room 0.63×, bicycle 1.06×, garden 0.20×. REAL gaps
  (p50=2/p90=8/p99=37) are far shorter → rerun microbench with real trace
  histograms before any gate arithmetic.
- Speedup surface (frac × gap): gap≤20 → 1.3-12×; gap=50 → 0.8-6.9×; gap≥300 →
  <1× for most fracs. Break-even K ≈ 4-12 per naive fit.

## Phase-6 microbench (FINAL 2026-09-30, real gaps, EPIC-05 GPU6, VOS_MICROBENCH.json)

- v2 kernels: vos_v0 (3 scalar tables) + vos_v1_packed (one float4 table load per
  replay step). Correctness gate re-PASSED for both (d_theta ≤ 7.5e-8 vs sequential
  torch ref; dense ≤ 3.0e-8 vs torch Adam; untouched rows bit-identical).
- **Scene-level (REAL gap histograms resampled from trace 29001..30000, REAL
  N/vis/react_share)**: room dense 0.672ms (1684 GB/s) → v1 0.289ms (**2.33×**);
  bicycle dense 2.919ms (1683 GB/s — the earlier 450 GB/s WAS co-tenant
  interference) → v1 0.860ms (**3.39×**); garden dense 2.034ms → v1 0.839ms
  (**2.42×**). These are shs-only (D=48) kernels.
- **K-scaling is LINEAR** (v0: 0.233→57.9ms over K=0..5000; ~1.06 ps/elem-step;
  v1 ~0.91 ps/elem-step): the first run's "regime change at K≥500" did NOT
  reproduce → it was co-tenant interference, not L1 footprint. Clock locking
  (nvidia-smi -lgc) was DENIED by the driver even as root (container?), but the
  clean run's consistent 1683-1691 GB/s + linearity is a no-throttle signature.
- Per-elem-step replay 0.91ps vs dense per-step 16.5ps → replay ~18× cheaper than
  the dense step it replaces. Real gaps (mean 4.27/6.35/4.83) sit deep in the
  cheap regime; synthetic-gap skew (0.20-1.06×) was the lognormal tail's fault.
- Fence amortization (from trace analysis, vos_region fences at 20K/25K/30K):
  room 1,997 row-steps/iter ≈ 89ns/iter; bicycle 6,247 ≈ 279ns; garden 4,795 ≈
  214ns — NEGLIGIBLE. Fence BURST at the eval iteration itself: up to 37.3M
  row-steps (bicycle 25K) ≈ 1.7ms one-off (overlappable, amortizes to nothing).

## Phase-7 gate (PRELIMINARY 2026-09-30, VOS_GATE.json — PROMOTE)

- P2 model (conservative: denominator = my optimized fused dense kernel at
  ~1684 GB/s, ~4× stronger than the trainer's torch foreach stage at ~424 GB/s
  effective): optimizer speedup room 1.90×, bicycle 2.41×, garden 1.95× (3/3
  ≥1.5×). E2E region (15K→30K) 7.5%/12.7%/11.2%; strict full-run (VOS only in
  15K→30K) 3.7%/6.3%/5.6% (2/3 ≥5% under the strict reading, 3/3 under the
  region reading). Overhead fraction of saved time 14.1%/2.3%/2.6% (3/3 ≤20%).
  Exact replay bitwise PASS on all oracle runs. **VERDICT: PROMOTE** (all four
  criteria, gating on the strict E2E reading).
- P1 (vs status-quo torch stage, byte-proportional): room 2.55×, bicycle 3.4×,
  garden 3.0× — reported as the "vs current implementation" view, NOT the gate
  basis (P2 is the honest floor).

## Phase-5 oracle round 3 (2026-09-30, COMPLETE — 7/7 bitwise PASS)

- O4b/O5b (densify-region / post-reset with T5/T15 ckpts) CRASHED by design
  flaw: `records[it_lo-1]` out of range — the T5 trace ends at 5000 and the T15
  trace at 15000, so segments starting AFTER the ckpt iteration are impossible.
- STRUCTURAL LIMIT discovered: the trainer saves ckpts ONLY at the final
  iteration, AFTER that iteration's topology event → a densify-region segment
  (t<15000) can never N-match a same-process ckpt. The post-reset segment
  15001..15100 CAN pair: no event runs after 15000, so a traced run to 15100
  gives ckpt_15100 with N identical to segment N.
- **O5b2 (room_T15100 traced run → in-process ckpt_15100 + segment 15001..15100,
  REAL post-opacity-reset state): bitwise_vs_A_s TRUE** ✓. bitwise_vs_A_f FALSE
  with nonzero A_f-vs-A_s envelope — this realistic ckpt state triggers the
  torch-internal foreach/single envelope; B tracks A_s exactly. Consequence for
  Phase 8: "bitwise vs C0's foreach path" is not achievable in principle when
  the envelope is nonzero; VOS reproduces A_s semantics and the Phase-11
  quality gates measure the E2E effect.
- FINAL: 7 runs (O1b, O2, O3, O6, O7, O9, O5b2) all bitwise vs A_s; realistic
  ckpt states (room/bicycle/garden 30K) also bitwise vs A_f; dense control
  bitwise in all runs.

## Phase-7 gate (FINAL 2026-09-30, VOS_GATE.json — **VERDICT: PROMOTE**)

- C1 optimizer speedup (P2 conservative fused-dense baseline): room 1.90×,
  bicycle 2.41×, garden 1.95× — 3/3 ≥1.5× PASS.
- C2 full-iteration reduction: region (15K→30K) 7.5%/12.7%/11.2% (3/3 ≥5%);
  STRICT full-run reading (VOS only 15K→30K) 3.7%/6.3%/5.6% (2/3 ≥5%) — PASS
  under both readings (gate requires 2/3).
- C3 overhead ≤20% of saved time: 14.1%/2.3%/2.6% — 3/3 PASS.
- C4 exact replay: 7/7 oracle runs bitwise vs A_s — PASS.
- **GATE = PROMOTE.** P1 (vs status-quo torch foreach stage): 2.55×/3.4×/3.0×.
- Deliverables at vos/ root: VOS_PRIOR_ART.md, VOS_DEPENDENCY_GRAPH.json,
  VOS_VISIBILITY_GAPS.json, VOS_TRAFFIC_ORACLE.json, VOS_REPLAY_CORRECTNESS.json,
  VOS_MICROBENCH.json, VOS_AMDAHL.json, VOS_GATE.json (+ VOS_EXACTNESS_MODEL.md,
  VOS_FINDINGS.md). Raw oracle/mb/trace artifacts in vos/results/ and on
  EPIC-05 /root/vos_traces/.

## Phase-8 implementation design (post-PROMOTE, 2026-09-30)

- Kernel: microbench v1 packed-float4 kernel WITHOUT --use_fast_math (IEEE
  sqrt/div) → target bitwise-A_s per-iteration; verify with oracle-style check.
- Trainer integration: zero-source-diff wrapper (proven trace-driver pattern):
  patch rasterize for the visibility bitmap + patch torch.optim.Adam in
  gaussian_model's namespace with a VOSAdam subclass.
- VOSAdam.step(): t ≤ 15000 → super().step() (stock, bitwise C0); t > 15000 →
  pop the shs group, super().step() (foreach over remaining 4 attrs — per-element
  math unchanged by batch composition → bitwise for shared attrs), restore
  group, then VOS kernel on active shs rows (gather m/v/theta, replay K=t−tau−1
  zero-grad transitions via packed scalar table, real step, scatter, tau=t).
- tau int32[N]; active_ids from the packed bitmap (torch.nonzero); grad gather.
- Fences: materialize-all before eval iters (20000/25000/30000) + before ckpt;
  wrapper may fence more often in short tests (semantically safe).
- shs lr is CONSTANT 0.0025 in C0 → replay table trivial; wrapper still records
  per-iter group lr into the table (future-proof for scheduled attrs).
- VOS_5K: forced-active from iter 1001, fences every 500, vs C0-5K same seed.
- VOS_30K: 3 scenes, VOS 15K→30K, back-to-back C0 vs VOS on same idle GPU.

## Phase-8 implementation (2026-09-30, IN PROGRESS)

- Driver: vos/scripts/vos_train_driver.py — zero-source-diff wrapper (proven
  trace-driver pattern): patches rasterize for the visibility bitmap (kept on
  GPU, no copy) + wraps GaussianModel.training_setup to proxy torch.optim.Adam.
- VOSAdamProxy: pops the shs group for inner.step() (foreach over the 4 eager
  attrs — per-element math independent of list composition), then the VOS
  kernel updates active shs rows only. Fences: eval iters + final iter +
  pre-densify (end of step s when s+1 is an event iteration).
- **COUNTER-space bookkeeping (critical)**: C0's Adam SKIPS every param whose
  grad is None — at densify events all params are REPLACED (nn.Parameter with
  .grad=None) so the event-iteration step is a NO-OP and the state["step"]
  counter does NOT increment. tau + table indices therefore live in COUNTER
  space (c = iteration − #events so far); the proxy adopts C0's own
  state["step"] at activation and maintains c. tau rebuilt to full(N, c) at
  event steps (valid: pre-event fence materialized everything; newborn rows'
  zero state makes all missed transitions exact no-ops).
- **Production kernel is BITWISE-EXACT vs torch Adam** (kernel-check gate,
  max|dθ|=0.0, untouched rows bit-identical). The torch rounding forms were
  pinned empirically (vos_round_probe.py, 2^20 random samples):
    lerp_(g,w1)      = fma(w1, round(g−m), m)
    mul_(b2)         = round(v·b2)
    addcmul_(g,g,w2) = fma(w2, round(g·g), round(v·b2))   ← g·g rounds FIRST
    (sqrt(v)/bc2)+ε  = round(round(sqrt(v)/bc2) + ε)
    addcdiv_(m,d,−ss)= fma(−ss, round(m/d), th)
  The naive `v*b2 + w2*(g*g)` C++ form is NOT bitwise under either fmad mode
  (nvcc contraction fuses v·b2 into the add; torch's mul_+addcmul_ rounds it).
  All scalars pre-rounded f64→f32 exactly like torch's host-scalar path
  (packed float4 table {step_size_s, bc2_sqrt_s}).
- Fallback: --vos-mode torchop (gather + vectorized torch ops + scatter, the
  oracle-B path; per-row op order preserved under row-vectorization). The
  startup gate auto-selects: kernel if bitwise, else torchop.
- 5K functional gate launched: forced-early VOS (vos-start=1000, densify
  region, fences+migration+counter) vs instrumented C0 (vos-start=9999),
  room, seed 42, GPUs 1/2 in parallel. Compare ckpts bitwise via
  vos_compare_ckpts.py. Expected: eager attrs bitwise (foreach composition
  independence); shs bitwise vs A_s semantics; C0's foreach-vs-single
  envelope may make the pair differ by ≤1 ulp on some elements (O5b2 showed
  the envelope is nonzero on SOME realistic states).

## Phase-8 (2026-09-30): trajectory-nondeterminism finding — TWIN TEST

- The 599-iter isolation series exposed that ANY two training runs diverge:
  C0-vs-VOS, A_s-vs-VOS, A_s-vs-wrapper-control (zero VOS steps), and
  crucially **A_s-vs-A_s TWIN RUNS** (identical code, seed, GPU class) all
  differ by the same magnitude (theta_max_abs ≈ 0.008-0.013 on xyz,
  ≈ 0.5-0.9 on scaling; first-eval PSNR deltas ~2e-4). MEASURED: the C0
  trainer is NOT bitwise reproducible run-to-run — the backward's atomic
  float accumulation order is nondeterministic (same mechanism as the
  Phase-4 R-vs-T ckpt divergence, 470 rows at 5K).
- CONCLUSION: cross-run BITWISE trajectory comparison is impossible in
  principle for this workload. Phase-8 exactness therefore rests on:
  (a) the in-process Phase-5 oracle (7/7 bitwise, 136M+ transitions);
  (b) the startup kernel gate (bitwise vs torch-op reference, max|dth|=0.0);
  (c) Phase-11 E2E quality parity within the twin envelope (P12 references:
      room 32.11 vs traced 32.08 → ~0.03 dB at 30K).
- The 5K forced-early test (VOS from 1001, 3960 VOS steps, 41 fences, kernel
  mode) ran clean: PSNR within the run-to-run envelope; wall 1.6min vs C0
  ~2.0min.
- Also found+fixed en route:
  * torch's CUDA div-by-CPU-scalar is MULTIPLY-BY-f32-RECIPROCAL
    (div_true_kernel_cuda's is_cpu_scalar fast path), NOT true division —
    the kernel denom must use f32(1/f32(bc2)); at t≈25000 bc2 rounds to 1.0
    in f32 and masks the difference, so the startup gate now checks at
    t≈5000 where it bites (earlier in-run gate also read tab[25000] out of
    bounds on a 5100-entry table → garbage fallback to torchop).
  * eval fence list fixed to the true C0 recipe (500,1000,2000,5000,10000,
    15000,20000,25000,30000 — config.py L75); the first forced-early runs
    had stale evals at 2000/5000 (trajectory unaffected; final ckpt fenced).
  * gaussian_model is importable only AFTER bootstrap_c0() (sys.path is set
    up there, not at trainer module import).
  * p.grad is None at every densify-event iteration (params replaced by
    nn.Parameter) — C0's Adam step is a NO-OP there; VOS must skip too
    (counter-space bookkeeping).

## Phase-9/11 FINAL (2026-09-30 13:25Z): order-B re-pairs + verdict

- **Order-B pairs (VOS-first, GPU-3-exclusive; MR0 co-tenant CPU load
  symmetric within each pair):** room stage 1.66x / wall +2.51% /
  dPSNR -0.347; bicycle 2.45x / +10.71% / -0.015; garden 1.98x / +7.86% /
  +0.008. Garden order-B VOS PSNR is HIGHER than C0; bicycle's is inside
  the C0 twin envelope (0.092). Room's -0.347 is the low draw of the VOS
  twin spread (VOS_a vs VOS_b: 0.219 apart) and is A_s-attributed.
- C0 twin spreads at 30K: PSNR 0.024 (room) / 0.092 (bicycle) / 0.055
  (garden); wall -2.9/-6.2/+7.0 s (0.6-0.9%).
- **VERDICT: KEEP** (VOS_VERDICT.json). C1 3/3 (1.66/2.45/1.98);
  C2-strict 2/3 (+2.5/+10.7/+7.9%); C2-region 3/3 (7.3/22.3/14.0%);
  SSIM/LPIPS 3/3; PSNR 2/3 inside twin noise, room attributed to the
  torch-internal A_s envelope via the pure-A_s control run.
- Amdahl consistency confirmed: wall gain tracks optimizer share (room 17%
  -> +2.5%; garden 27% -> +7.9%; bicycle 36% -> +10.7%).
- Caveats carried: room marginal; per-iteration torch.nonzero host sync is
  CPU-contention-sensitive (production would use device-side compaction);
  n=2 pairs, single seed; shs-only scope.

## Phase-9/11 (2026-09-30): 30K campaign results + A_s attribution

- **Stage A/B (15K->30K window, same-GPU sequential pairs, seed 42):**
  room 2.886 -> 2.021 ms/iter (1.43x, 3-way-contended) / 2.888 -> 1.738
  (1.66x, uncontended re-pair); bicycle 10.400 -> 4.383 (2.37x);
  garden 7.369 -> 3.687 (2.00x). The VOS stage carries a per-iteration
  host sync (torch.nonzero) and is CPU-contention-sensitive: the room pair
  ran 1.43x under 3-way contention vs 1.66x clean — contention deflates the
  measured speedup (conservative direction).
- **E2E wall:** room +2.51% (clean, VOS-first order), bicycle +7.9%,
  garden +4.0% (contention-symmetric pairs). C0 twins reproduce wall to
  0.6%; the stage window reproduces to 0.07%.
- **Twin envelopes at 30K (room):** C0-vs-C0 final PSNR spread 0.024 dB;
  VOS-vs-VOS spread 0.219 dB (the A_s-class arms are intrinsically more
  run-to-run variable on this workload).
- **A_s ATTRIBUTION RUN (decisive):** pure torch Adam foreach=False (A_s,
  NO VOS) 30K room: final PSNR 31.988 vs C0 (A_f) twins 32.181/32.205 and
  VOS arms 32.076/31.858. CONCLUSION: the ~0.1-0.35 dB VOS-vs-C0 PSNR cost
  is the torch-internal A_f-vs-A_s envelope (<=1 ulp/step), NOT a VOS
  defect — VOS is bitwise-faithful to A_s (oracle 7/7) and lands inside the
  A_s-class spread. Any exact single-tensor/custom-kernel Adam would pay
  the same cost; bitwise-A_f trajectory preservation is impossible in
  principle (Phase-5 O5b2 finding).
- Quality gates (derived, 3x C0-twin envelope): SSIM/LPIPS pass 3/3;
  PSNR: bicycle -0.089, garden -0.091 (pass), room -0.105/-0.348 (at/over
  the 0.10 gate; mean -0.23, A_s-attributed).
- Eval trajectories: all arms within 0.043 dB at t=15000 (pre-activation);
  the A_s-class gap opens in the 15K-30K region (t=20000 eval dips
  -0.2~-0.4) and partially recovers by 30K.
- Trainer facts re-confirmed: exactly ONE grad-enabled render per iteration
  (bitmaps_captured = 30000/30000), so single-bitmap deferral is sound;
  densify ends 14900; fences at evals 20000/25000/30000 + final only.

## GPU pinning on EPIC-05 (2026-09-30, resolved)

- **Runtime `os.environ["CUDA_VISIBLE_DEVICES"]=x` inside the driver does NOT
  pin in this container; exec-time env does.** Proven by vos_cvd_probe2.sh
  (exec-time CVD=3/5 → correct UUIDs) vs the 30K campaign (driver-side
  setenv before any CUDA init, `cuda_init=False` confirmed at both debug
  points → context still on GPU 0). vos_pin_repro.sh + an in-run
  `[vos-cuda] device_count=1` print confirm exec-time CVD works end-to-end.
  CONCLUSION: every launcher must prefix `CUDA_VISIBLE_DEVICES=$GPU`.
- Corollary: ALL pre-30K driver runs (5K pair, 599 series, twins) ran as
  co-tenants on GPU 0 (the old driver built the scalar table on CUDA before
  the trainer's main, and runtime setenv doesn't pin). Semantics/quality
  comparisons remain valid (same GPU model); their wall/inner/vos timings
  are co-tenant-polluted and superseded by the 30K campaign.
- Zombie CUDA contexts from force-killed jobs linger in nvidia-smi
  compute-apps and can fake a mis-pin reading; always force-kill, verify
  4MiB on all GPUs, and poll twice before concluding.
- EPIC-05 local clock = UTC+8; `date -u` is correct UTC.

## Tooling gotchas (this session)

- evidence_record tool broken (rejects `evidence` param as undeclared) — use this file.
- PowerShell nested ssh + python -c quoting is unreliable — always scp a script then run it
  (as per EPIC-05 SESSION_FINDINGS.md). Simple `ssh host "cmd"` works; heredocs fail.
- **Driver GPU-pinning bug (fixed)**: vos_trace_driver's packbits probe initialized CUDA
  BEFORE the trainer set CUDA_VISIBLE_DEVICES → all 3 T30 runs shared GPU 0 (setdefault
  is a no-op after CUDA init). Semantics unaffected (trace content is state-dependent,
  not GPU-dependent), runs just 3× slower. Fix: lazy probe at first record() on
  bitmap.device. Applies to any driver that touches CUDA before the trainer's main().
- `--checkpoints` was missing from the R15/R5 launcher → no mid-stage ckpts on first
  pass; relaunched with the flag (6 runs, ~7 min on idle GPUs) — ckpt_15000/5000 now
  exist for all 3 scenes.
- Local repo copies: vos/sources/c0_gaussian_inference.py (C0 renderer wrapper, 3057 lines,
  scp'd from EPIC-05 frozen worktree), vos/sources/publication_trainer.py (mx-path variant —
  the EPIC-05 path variant is /root/epic05_p13x/publication_trainer_p13x.py).
- arXiv Atom API over HTTPS works from this network (export.arxiv.org/api/query) even
  though Google/DDG/S2 are blocked and the web_search tool is broken — use it for
  literature sweeps. Semantic Scholar API: persistent 429.
