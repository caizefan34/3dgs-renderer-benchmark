# VOS-GS Phase-8 — SH-Only Implementation (post-PROMOTE)

Status: IMPLEMENTED + 30K PROTOCOL MEASURED (VOS_30K.json, VOS_QUALITY.json;
verdict in VOS_VERDICT.json).

All labels per the mission protocol: OBSERVED / DERIVED / HYPOTHESIS / MEASURED / CONCLUSION.

## 1. Scope

CONCLUSION (from the promotion gate, VOS_GATE.json = PROMOTE): the implemented
virtualization targets **shs only, active in iterations 15001..30000** — the
region with NO topology events (last densify at 14900, no opacity reset after
12000, identity stable, fences only at evals 20000/25000/30000 + final ckpt).
xyz/scaling/rotation remain EAGER (culling reads them every iteration);
opacity remains EAGER in v1 (Phase-10 evaluates adding it).

## 2. Architecture (OBSERVED — as implemented)

Driver: `vos/scripts/vos_train_driver.py` — zero source diff on the frozen
trainer (the pattern proven trace-neutral in Phase 4):

1. **Visibility capture**: `gsplat.experimental.rasterize_gaussian_higs_dynamic`
   is wrapped at bootstrap; the last grad-enabled render's bitmap
   `(radii > 0).any(-1)` is kept on GPU (no copy, no packbits, no RNG).
   `evaluate()` renders under `no_grad` and never overwrites it.
2. **Optimizer proxy**: `GaussianModel.training_setup` (reference_v1) is
   wrapped so the constructed `torch.optim.Adam` is enclosed in
   `VOSAdamProxy`. Everything delegates; only `step()` is intercepted.
3. **`VOSAdamProxy.step(t)`**:
   - `t <= vos_start` → stock `inner.step()` (bitwise C0 path).
   - `t > vos_start`:
     a. pop the shs param group; `inner.step()` updates the 4 eager attrs
        (per-element math is independent of foreach list composition);
        restore the group;
     b. VOS shs step on ACTIVE rows only (this iteration's bitmap):
        `k = (c-1) - tau[row]` deferred zero-grad transitions are replayed in
        registers, then the real step-t update is applied, then
        `tau[active] = c`;
     c. fences: at every eval iteration (500,1000,2000,5000,10000,15000,
        20000,25000,30000 — config.py L75), at the final iteration (before
        the ckpt save), and at the end of any step whose NEXT iteration runs
        a densification event (forced-early tests only). A fence materializes
        every stale row through the current step (zero-grad replay only).
4. **Counter-space bookkeeping** (MEASURED quirk of C0): at densify events
   all params are REPLACED (`nn.Parameter` with `.grad=None`), so C0's Adam
   step is a NO-OP there and `state["step"]` does NOT increment. tau and all
   table indices therefore live in C0's step-counter space `c`
   (`c = iteration − #events so far`); the proxy adopts C0's own
   `state["step"]` at activation and maintains `c`. At event steps tau is
   rebuilt to `full(N, c)` — exact because (a) we fenced before the event so
   survivors are materialized through `c`, and (b) newborn rows carry zero
   m/v whose missed transitions are exact no-ops (`m=0 ⇒ Δθ=0`).

## 3. Exactness (MEASURED)

- **Kernel vs torch Adam: BITWISE.** The startup gate builds the kernel
  (fmad=true, then fmad=false) and keeps the first that reproduces a per-row
  torch-op reference bitwise (m, v, θ, untouched rows; measured max|dθ|=0.0).
  The exact torch rounding forms were pinned empirically
  (`vos_round_probe.py`, 2^20 random samples):
    - `lerp_(g, w1)`        = `fma(w1, round(g−m), m)`
    - `mul_(b2)`            = `round(v·b2)`
    - `addcmul_(g,g,w2)`    = `fma(w2, round(g·g), round(v·b2))` (g·g rounds first)
    - `(sqrt(v)/bc2)+ε`     = `round(sqrt(v) · f32(1/f32(bc2)) + ε)` — torch's
      CUDA div-by-CPU-scalar is MULTIPLY-BY-RECIPROCAL
      (`div_true_kernel_cuda` is_cpu_scalar fast path), not true division
    - `addcdiv_(m,d,−ss)`   = `fma(−ss, round(m/d), θ)` (d is a tensor: true division)
  All per-step scalars ({step_size_s, inv_bc2_s}) are pre-rounded
  f64→f32 exactly like torch's host-scalar path (packed float4 table).
- **In-process oracle**: Phase-5 VOS_REPLAY_CORRECTNESS.json — 7/7 runs
  bitwise vs A_s (incl. realistic 30K states on all 3 scenes, the real
  post-opacity-reset state, 136M+ transitions replayed).
- **Trajectory-level (E2E)**: MEASURED via the TWIN TEST — the C0 trainer is
  NOT bitwise reproducible run-to-run (backward atomic accumulation order;
  identical A_s twin runs differ by θ_max ≈ 0.009/0.55 at t=599, and every
  arm pair — including a zero-VOS-step wrapper control — differs by the same
  magnitude). CONCLUSION: cross-run bitwise trajectory comparison is
  impossible in principle; E2E exactness is therefore gated by Phase-11
  quality parity within the twin envelope (P12 reference vs traced run:
  32.11 vs 32.08 dB at 30K → envelope ≈ 0.03 dB converged).
- Fallback: `--vos-mode torchop` (gather + vectorized torch ops + scatter;
  oracle-B path) — 1 ulp off on rare replay elements (tensor scalars cannot
  express torch's fused addcdiv); retained for portability, not used when the
  kernel gate passes (it does on A100/sm_80).

## 4. Functional validation (MEASURED, room, seed 42)

- **599-iter isolation series** (final PSNR, all within ±0.006 dB of each
  other; twin pair alone differs by 0.0037 dB):
  A_s 18.0468 | A_s twin 18.0431 | wrapper control 18.0416 | envelope-free
  VOS 18.0412 | single-VOS-step 18.0463 | C0 18.0465 | VOS 18.0421.
- **5K forced-early run** (VOS from 1001 — inside the densify region:
  exercises pre-densify fences, tau migration, counter machinery, event
  no-op steps): 3960 VOS steps (4000 post-activation iterations minus 40
  event no-ops ✓), 41 fences (40 pre-densify + 1 final ✓), kernel mode,
  PSNR 26.97 vs C0-arm 27.18 (Δ=0.21 dB with a 2.3% population difference
  from nondeterministic event selection — the 5K-scale twin envelope is
  expected to be of this order; the converged 30K protocol comparison is the
  quality gate). Wall 1.6 min vs C0 2.0 min.
- **Optimizer-stage timing preview** (5K run, N≈450-500K): VOS stage
  (4-attr foreach 0.890 ms + VOS shs 0.462 ms) ≈ 1.35 ms vs C0 all-dense
  ≈ 1.54 ms → ≈12% optimizer-stage reduction at 5K scale; the 30K protocol
  runs measure the headline numbers.

## 4b. 30K protocol results (MEASURED — full detail in VOS_30K.json / VOS_QUALITY.json)

- **Optimizer stage (15K→30K window, same-GPU sequential pairs, seed 42):**
  room 2.886→2.021 ms/iter = 1.43× under 3-way co-tenant load, 2.888→1.738 =
  **1.66×** in the uncontended order-swapped re-pair; bicycle 10.400→4.383 =
  **2.37×**; garden 7.369→3.687 = **2.00×**. Contention deflates the measured
  VOS speedup (per-iteration host sync in `torch.nonzero`) — the contended
  numbers are conservative.
- **E2E wall:** room +2.51% (clean re-pair), bicycle +7.9%, garden +4.0%
  (contention-symmetric pairs). Region-share of the saving: room 5.0%,
  bicycle 20.7%, garden 13.4% of region iteration time.
- **Quality (final eval, all cameras):** PSNR deltas VOS−C0: room −0.105 /
  −0.348 (two pairs), bicycle −0.089, garden −0.091; SSIM/LPIPS inside gates
  on 3/3. C0-vs-C0 twin envelope at 30K: 0.024 dB.
- **A_s attribution (decisive):** pure `foreach=False` torch Adam (NO VOS)
  lands at 31.988 dB — inside the VOS spread (32.076/31.858), 0.2 dB below
  the C0 foreach twins (32.181/32.205). CONCLUSION: the PSNR cost is the
  torch-internal A_f-vs-A_s envelope (≤1 ulp/step, chaotically amplified),
  NOT a VOS defect; VOS is bitwise-faithful to A_s. Any exact
  single-tensor/custom-kernel Adam pays the same; bitwise-A_f trajectory
  preservation is impossible in principle (Phase-5 O5b2).

## 5. Files

- `vos/scripts/vos_train_driver.py` — the driver (CUDA kernel + proxy + patches)
- `vos/scripts/vos_round_probe.py` — torch rounding-form probe (addcmul form B, etc.)
- `vos/scripts/vos_compare_ckpts.py` — checkpoint comparator (bitwise + max_abs)
- `vos/scripts/vos_diff_curves.py` — eval-curve divergence locator
- EPIC-05 `vos_runs/` — all run artifacts (results/timing/quality/ckpt per run)
