# VOS-GS — Exactness Model (Phase 3)

Status: FROZEN v1 (per-step replay; closed-form/block replay explicitly out of scope —
may be studied separately only after the per-step ground truth exists).

## 1. Baseline semantics being preserved (all OBSERVED from source)

The frozen C0 trainer executes, per iteration t = 1..T:

1. `model.update_learning_rate(t)` — sets the xyz group lr to `f(t)` (exponential schedule
   with delay; `gaussian_model._make_expon_lr_func`); other groups keep constant lr.
2. One forward render (C0 dynamic path) + L1+fused-SSIM loss + `loss.backward()`.
3. `model.optimizer.step()` — `torch.optim.Adam` (torch 2.9.1+cu128), betas=(0.9, 0.999),
   eps=1e-15, weight_decay=0, amsgrad=False, capturable=False, differentiable=False,
   foreach default (CUDA → multi-tensor path), DENSE over all rows of all 5 tensors.
4. `model.optimizer.zero_grad(set_to_none=True)`.

Gradient buffer fact: the C0 backward allocates five full-N `torch.zeros_like` master grad
buffers and scatters visible-row gradients through `visible_ids`
(c0_gaussian_inference.py L2382-2435). Therefore for any row i not in the visible set C_t:

    grad_a[i] == +0.0   (exactly, fp32, every iteration)

## 2. torch.optim.Adam transition (the function VOS must reproduce)

For a param tensor with state step counter `s` (incremented once per step() call — the
trainer calls step() exactly once per iteration, so s == the iteration number), the torch
2.9.1 non-capturable, non-amsgrad, zero-weight-decay path applies, per element:

    step_t += 1                                        # s = current iteration
    exp_avg.lerp_(grad, 1 - beta1)                     # m <- m + (1-beta1)*(g - m)
    exp_avg_sq.mul_(beta2)                             # v <- v * beta2
    exp_avg_sq.addcmul_(grad, grad, value=1 - beta2)   # v <- v + (1-beta2)*g*g
    (host, double precision:)
    bias_correction1     = 1 - beta1**s
    bias_correction2     = 1 - beta2**s
    step_size            = lr / bias_correction1
    bias_correction2_sqrt = bias_correction2**0.5
    (device, elementwise:)
    denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
    param.addcdiv_(exp_avg, denom, value=-step_size)   # theta <- theta - step_size*m/denom

With g = +0.0 exactly:

    m <- m + (1-beta1)*(0 - m)          # NOT identical rounding to m*beta1 in general
    v <- v*beta2 + (1-beta2)*(+0.0)     # note: -0.0 + +0.0 = +0.0 (sign-of-zero matters)
    theta <- theta - step_size * m / (sqrt(v)/bc2_sqrt + eps)

**theta moves under zero gradient whenever m != 0.** This is why skipping rows
(SparseAdam-style) is NOT dense-equivalent, and why VOS defers-and-replays instead.

## 3. VOS state

Per attribute a ∈ {shs, opacity} (Phase-1/2 scope) and per row i:

    (theta_a[i], m_a[i], v_a[i], tau_a[i])

tau_a[i] ∈ {0..T} is the logical optimizer step of the physical state: the physical tensors
equal the dense baseline's tensors after iteration tau_a[i]. Initial tau = 0 at state birth
(init rows: tau=0 with m=v=0; clone/split children: tau=t_event with m=v=0; after
reset_opacity: tau=t_event for all opacity rows).

## 4. Events

- **CONSUME(i, t)**: row i ∈ C_t (visible set of iteration t, computed by the full-N culling
  projection over xyz/rotation/scaling — all EAGER attributes, always physically exact).
  Before the forward producer reads theta_a[i]: if tau_a[i] < t-1, REPLAY(i, tau_a[i]+1, t-1)
  then set tau_a[i] = t-1. The forward then reads theta. After backward, the VOS optimizer
  applies the real (possibly nonzero-grad) transition at step t to row i: tau_a[i] = t.
- **DEFER(i, t)**: row i ∉ C_t. No physical work. tau_a[i] unchanged. The dense baseline's
  transition at t for row i is the zero-grad transition (Section 2 with g=+0.0); VOS records
  only that steps tau_a[i]+1..t are pending.
- **FENCE(t)**: densify_and_prune, reset_opacity, checkpoint/export, evaluation render.
  Before the event: REPLAY(i, tau_a[i]+1, t-1) for ALL rows i (full materialization),
  tau_a[i] = t-1 for all i. (For the optimizer step at t itself, the event-affected rows
  then transition normally.) Densification reads (mask over all rows of theta_opacity,
  theta_scaling, max_radii2D; survivor m/v via mask-indexing) are then dense-exact.

## 5. REPLAY(i, s0, s1) — exact deferred-transition replay

For s = s0 .. s1 (in increasing order), apply the Section-2 transition to row i with
g = +0.0 and the exact logical-step-s scalars:

    bias_correction1      = 1 - beta1**s      (host double)
    bias_correction2      = 1 - beta2**s      (host double)
    step_size             = lr_a(s) / bias_correction1
    bias_correction2_sqrt = bias_correction2**0.5

where lr_a(s) = f(s) for xyz (not used in Phase-1) and the group constant for shs/opacity.
The device-side op sequence must be THE SAME torch elementwise ops as the baseline
(lerp_ against a +0.0 tensor, mul_, addcmul_ against +0.0, sqrt/div/add_, addcdiv_) so that
per-element rounding matches. No closed-form shortcut is permitted in v1.

**Per-step replay is mandatory in v1** to establish the correctness ground truth.
(Register-resident multi-step replay in the CUDA implementation applies the SAME
per-step op sequence K times in a loop — this is still per-step replay, not closed form.)

## 6. Exactness argument (CONCLUSION, to be validated by the Phase-5 oracle)

Claim: after materialization at any observation point, VOS state == dense baseline state.

1. *Per-row independence*: every Adam op is elementwise; row i's trajectory depends only on
   its own (theta, m, v) and the scalar schedule. No cross-row coupling exists in the
   baseline (OBSERVED: torch Adam elementwise ops; scatter-add gradients touch only
   visible rows, and each visible row is written by ... atomics over pixels — gradients of
   VISIBLE rows are unaffected by VOS since visible rows are never deferred).
2. *Zero-grad identity*: for i ∉ C_t the baseline grad is exactly +0.0 (Section 1), so the
   baseline's transition for row i at t is exactly the Section-5 replay input.
3. *Scalar identity*: the baseline uses step counter s == t and lr == lr_a(t) at iteration
   t; replay at logical step s uses identical host-double scalars → identical device
   scalars after float conversion.
4. *Consumer-set identity*: C_t depends only on EAGER attributes (culling reads
   xyz/rotation/scaling theta only — OBSERVED c0_gaussian_inference.py L697-741), which are
   never deferred, so C_t^{VOS} == C_t^{dense} bit-for-bit, inductively.
5. *Fence completeness*: every all-row consumer (Section 4 FENCE list, from
   VOS_DEPENDENCY_GRAPH.json) triggers full materialization before reading.

Therefore: theta, m, v of every row at every materialization point equal the dense
baseline's values, up to bitwise identity IF the replay op sequence matches the baseline's
kernel arithmetic exactly (expected for elementwise ops with identical scalars; the oracle
measures this).

## 7. Phase-5 oracle protocol (what "PASS" means)

- A = literal `torch.optim.Adam` (same ctor params as the trainer) stepping DENSELY every
  iteration on realistic states (checkpoint-extracted theta/m/v + random inits), with real
  visibility traces supplying per-row zero/real gradients and the real lr schedule.
- B = VOS: per-row tau bookkeeping, deferred transitions, replay-on-consume, fences at
  their trace positions.
- At random observation points force materialization of B and compare against A:
  max_abs, max_rel, rel_L2, cosine on theta, m, v per attribute.
- PASS requires FP32-equivalence: max_abs <= 1e-6-scale OR (if ops match bitwise)
  bit-identical tensors. Expected outcome given Sections 2/5: BITWISE identical.
- If bitwise fails but FP32-equivalent passes: record both, proceed (the claim is
  "floating-point reassociation envelope" at worst).
- If FP32-equivalence fails: STOP — do not enter the renderer (mission §5).

## 8. What VOS v1 deliberately does NOT claim

- No speedup claim for xyz/rotation/scaling (EAGER_REQUIRED under per-step exactness).
- No closed-form or block-compressed replay claim (v1 replays step by step).
- No semantic change to densification, pruning, opacity reset, SH schedule, camera order,
  loss, or renderer. The training trajectory must differ from dense only by allowed
  floating-point reassociation (expected: none, i.e. bitwise, per Section 6).
