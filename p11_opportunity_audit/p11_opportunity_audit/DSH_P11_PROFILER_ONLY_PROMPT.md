# DSH P11 — PROFILER-ONLY OPPORTUNITY GATE

## Mission

Find **one** new exact optimization with enough measured headroom to plausibly survive
30K training.

This phase is measurement only.

DO NOT:
- write a new CUDA optimization kernel;
- tune H8;
- alter the frozen P8/P10/DSH-A binaries;
- launch a 30K campaign;
- use approximate pruning/gating;
- resurrect C25/R6 candidates without new evidence.

Primary source:
- current gsplat main `512d366b67073d77ca099ede742683c165dfc23b`
- DSH-A publication recipe, eps2d=0.3, absgrad=True before 15K
- clean A100-SXM4 policy.

Scenes:
- room
- bicycle
- garden

Use existing frozen checkpoints/states where possible.

---

# Gate 0 — Identity and strong baseline

Record:
- repo SHA
- trainer SHA
- gsplat `.so` SHA
- fused-ssim source/commit and `.so` SHA
- PyTorch/CUDA/driver/GPU UUID
- exact checkpoint identities.

No algorithm change.

---

# Gate L1 — SepSSIM vs existing fused-ssim semantic equivalence

The publication trainer uses:
- 11x11 Gaussian window
- sigma=1.5
- same-padding semantics
- `L = 0.8*L1 + 0.2*(1-SSIM)`.

Use the existing `rahul-goel/fused-ssim` implementation with:
`padding="same"`.

Do NOT use gsplat's convenience `ssim_loss` if it hard-codes/chooses `padding="valid"`.

For each scene on at least 3 representative rendered images:
compare SepSSIM vs fused-ssim:

1. scalar SSIM / total loss
2. `dL/dimage`
3. finite fraction
4. max_abs
5. mean_abs
6. rel_l2
7. cosine

Calibrate normal FP implementation difference; do not invent a threshold before seeing
same-input repeats.

Classify:
- `SEMANTICALLY_MATCHED_WITHIN_FP`
- `MATERIAL_DIFFERENCE`.

If material difference -> fused-ssim is NOT a strong-baseline substitute for this
frozen recipe. Preserve result and continue decomposition with SepSSIM only.

---

# Gate L2 — Loss-only timing decomposition

Use detached rendered images with `requires_grad_(True)` so renderer is not timed.

Benchmark separately:
- L1 forward
- L1 backward to image
- SepSSIM forward
- SepSSIM backward to image
- combined SepSSIM+L1 forward
- combined backward to image
- fused-SSIM forward
- fused-SSIM backward to image
- combined fused-SSIM+L1 forward/backward.

Protocol:
- warmup >=20
- measurements >=100
- reps >=5
- CUDA events
- clean GPU.

Report repetition medians and median-of-medians.

Compute:
`T_loss_sep_total`
`T_loss_fused_total`
`S_loss = T_loss_sep_total / T_loss_fused_total`.

This is **baseline modernization evidence only**, not a contribution.

---

# Gate L3 — Renderer backward with externally supplied image adjoint

Goal: separate loss backward from renderer backward.

For each frozen state/camera:
1. render normally;
2. generate the exact `dL/dimage` using the loss path;
3. benchmark renderer backward driven by a precomputed detached image adjoint;
4. benchmark full loss.backward path.

Do not reuse a consumed autograd graph. Rebuild the graph correctly per timed sample.

Report:
- renderer forward
- renderer backward given external `dL/dimage`
- loss-to-image backward
- total F+B
- full backward.

Check additive timing closure within measurement noise.

---

# Gate L4 — Dynamic kernel profile

Collect short clean Nsight Systems/CUPTI profiles for:

A. publication SepSSIM path
B. same-semantics fused-SSIM path (if L1 passes)

Categorize:
- loss forward kernels
- loss backward kernels
- image-adjoint writes
- raster backward reads
- raster backward kernel
- projection/SH backward
- launches / host gaps.

For the fused-SSIM baseline calculate a conservative physically removable bound for a
hypothetical exact loss-adjoint→raster fusion.

Count only:
- full-image `dL/dimage` write/read traffic that would actually disappear;
- loss-backward kernels/launch work that can be subsumed without recomputation;
- avoidable intermediate tensors.

Do NOT count:
- renderer work that remains;
- SSIM arithmetic that a fused design still must execute;
- impossible peak-bandwidth savings twice.

Give:
- conservative bound
- moderate bound
- optimistic bound
all as `% T_iter`.

### L-GATE

`ADVANCE_LOSS_FUSION` only if:
- semantic gate passes (or exact SepSSIM formulation can be fused directly),
- conservative physically removable bound >= 2.0% E2E,
- moderate bound >= 3.0% E2E,
- no audited implementation already performs the same loss-adjoint→raster fusion.

Otherwise DROP.

---

# Gate D1 — Densification-stat timing decomposition

Only pre-15K states matter.

Use checkpoints/states around:
- ~2K
- ~5K
- ~10K
- ~14K

on room/bicycle/garden where available.

Measure separately:
1. raster backward `absgrad=False`
2. raster backward `absgrad=True`
3. `means2d.absgrad` clone/read
4. visibility mask / gather
5. pixel-space x/y scaling
6. L2 norm
7. `xyz_gradient_accum` / `denom` updates
8. max-radii update
9. total per-step densification-stat update.

Use current publication trainer semantics, not gsplat DefaultStrategy semantics if they differ.

Also verify exact formula:
- abs component accumulation location
- width/2 and height/2 scaling
- visible filter
- norm
- persistent accum/count.

### D-GATE

Compute the **joint** removable budget:
- any avoidable absgrad auxiliary output traffic in renderer backward
- plus post-backward tensor-op chain.

Advance only if conservative total >=2.0% E2E.

If <2%, DROP.

---

# Gate T — Current-main active-interaction sanity (measurement only)

This gate exists only to verify that old C28 conclusions still hold on `512d366b`.

Using existing metadata / contribution-count utilities, measure:
- tile candidate intersections
- pixels' `last_ids`
- estimated interactions actually traversed by current backward
- exact contributing interactions if obtainable without changing renderer
- memory lower bound for a compact `(gaussian_id,pixel_id)` worklist.

Do not build the worklist as an optimization.

If current `last_ids`/warp termination already removes most tail work OR compact-list memory/scan
cost is comparable to predicted savings, keep candidate DROP.

Only flag `REOPEN_TAIL` if there is a newly measured >=3% E2E conservative opportunity not
explained by existing mechanisms.

---

# Gate O — Optimizer sanity / no-go check

Measure optimizer phase by parameter group if cheap.

Audit exact dependency:
- next iteration projection reads means/scales/quats/opacities/SH;
- standard Adam updates may move a parameter even with zero current gradient due to momentum.

Do not implement cross-iteration double-buffer overlap.

Only reopen optimizer overlap if you can prove that every next-forward read observes exactly
the standard updated parameter value.

Existing fused/sparse Adam implementations are prior art; generic fused Adam is not a paper
contribution.

---

# Candidate decision table

Produce:

`P11_CANDIDATE_MATRIX.json`

For each:
- measured_current_e2e_fraction
- conservative_removable_pct
- moderate_removable_pct
- semantic_risk
- prior_art_risk
- implementation_complexity
- exactness_status
- decision

Candidates:
- LOSS_ADJOINT_RASTER_FUSION
- DENSIFICATION_STAT_SINK
- ACTIVE_INTERACTION_COMPACTION
- OPTIMIZER_OVERLAP
- ATOMIC_AGGREGATION (historical control)
- ZERO_INIT (historical control)
- BACKWARD_OPTIMIZER_FUSION (historical control)

---

# Final selection rule

At most ONE candidate may receive:

`ADVANCE_TO_MINIMAL_PROTOTYPE`.

Requirements:
1. exact semantics plausible and testable;
2. conservative E2E headroom >=2%;
3. moderate headroom >=3%;
4. not already subsumed by public prior art;
5. mechanism measurable without a 30K run;
6. expected gain is not merely a microkernel arithmetic reduction.

If none pass, return:
`NO_NEW_OPTIMIZATION_JUSTIFIED`.

Do not lower thresholds to force a candidate.

---

# Required artifacts

- `P11_SOURCE_IDENTITY.json`
- `P11_LOSS_EQUIVALENCE.json`
- `P11_LOSS_TIMING.json`
- `P11_LOSS_KERNEL_PROFILE.json`
- `P11_DENSIFICATION_TIMING.json`
- `P11_ACTIVE_INTERACTION_AUDIT.json`
- `P11_OPTIMIZER_DEPENDENCY_AUDIT.md`
- `P11_CANDIDATE_MATRIX.json`
- `P11_VERDICT.json`
- `P11_REPORT.md`

Preserve raw timing/profiler traces and commands.

---

# Final return format

```text
P11 opportunity gate

LOSS:
SepSSIM total =
fused-SSIM total =
semantic class =
conservative fusion headroom =
moderate fusion headroom =
decision =

DENSIFICATION:
absgrad auxiliary cost =
post-bwd stat cost =
combined conservative headroom =
decision =

TAIL:
current-main residual opportunity =
decision =

OPTIMIZER:
exact overlap feasible =
decision =

SELECTED CANDIDATE:
<name | NONE>

Reason:
...

Next action:
<MINIMAL_PROTOTYPE | STOP_NEW_OPTIMIZATION>
```
