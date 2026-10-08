# RF0B_POLICY_DIFF — FastGS Density-Control Policy Port into Frozen C0

Task: DSH-3 ReproFast0 **Lane B**. Isolate FastGS's *algorithmic* density-control
contribution by porting ONLY its public densification/pruning policy into the
frozen C0 trainer. Renderer, loss, optimizer, iteration budget, tile size,
holdout split, seed, and black background are all held FROZEN.

- Date: 2026-09-30
- FastGS source: `fastgs/FastGS` @ `44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f`
  (detached HEAD, "update fast-pgsr"). Frozen independently by Lane B (Lane A
  had not published a commit at start); **reconcile identities before any
  publication claim.** Local path: `/root/rf0a_laneA/FastGS`.
- C0 trainer: `/root/epic05_p12/publication_trainer_p12_holdout8.py`
  (sha256 `17d4034bd2b20b75d1e95f9d61d57edca681130f07ceb2eaf03f89d2dbbfe761`).
- C0 renderer: `gsplat.experimental.rasterize_gaussian_higs_dynamic`
  (ext `.so` `d26cccf4…`, core `.so` `935ea6c0…`).

---

## 1. Every file changed

| File | Role | Change |
|---|---|---|
| `rf0b_fastgs_model.py` (NEW) | FastGS policy subclass + scoring helpers | `FastGSGaussianModel(GaussianModel)` overriding ONLY policy decisions; module fns `compute_fastgs_scores`, `sample_score_cameras`, projection helper, `FGS_*` constants. |
| `rf0b_trainer.py` (DERIVED from frozen C0) | Training driver | Added `c0fgs` arm branch: dual-accumulator densification stats, FastGS score computation, `densify_and_prune_fastgs` call, `final_prune_fastgs` block, counters. NO change to renderer/loss/optimizer/iterations/split/seed. |

The frozen C0 renderer binaries, loss backend (`SepSSIM`, `0.8*L1+0.2*(1-SSIM)`),
optimizer/LR groups, 30K iteration budget, tile16, black background, corrected
true-holdout (`i % 8 == 0`), and seed42 are UNTOUCHED. `c0fgs` reuses the exact
C0 render path (`bootstrap_c0`, `rasterize_gaussian_higs_dynamic`, eps2d 0.3,
H8_MR + absgrad env) — only the density-control decisions differ from `c0`.

---

## 2. Every C0 semantic preserved (byte-level where practical)

- **Renderer**: `rasterize_gaussian_higs_dynamic` via `bootstrap_c0()`; identical `.so` shims, `HIGS_PX_RUNTIME=2`, `HIGS_BWD_SCALAR_ADJOINT=scalar_adjoint`, `HIGS_BWD_H8_MR=1`, `HIGS_BWD_ABSGRAD=1`, eps2d=0.3, tile_size=16, packed=False.
- **Loss**: `(1-0.2)*L1 + 0.2*(1-SSIM)`, SepSSIM training loss (default `--loss sep`); eval SSIM metric = SepSSIM. Unchanged.
- **Background**: black (`background="black"`).
- **Optimizer / LR groups**: inherited UNCHANGED from frozen C0 `training_setup` (xyz/shs/opacity/scaling/rotation groups, exponential xyz schedule). The subclass only ADDs a zeroed `xyz_gradient_accum_abs` buffer; it does not alter any LR or group.
- **Iteration budget**: 30,000. `densify_from_iter=500`, `densify_until_iter=15,000`, `densification_interval=100`, `opacity_reset_interval=3000`.
- **Camera sequence / holdout**: identical rng (`random.Random(seed)`), `viewpoint_stack = [i for i in range(n) if i%8!=0]`, pop-shuffle. True-holdout = every 8th camera (`i%8==0`), never trained.
- **Seed**: 42 (`torch/np/random` seeded; camera-sequence rng seeded).
- **SH progression**: `oneupSHdegree()` every 1000 iters (unchanged).
- **Gaussian-ID bookkeeping** (`_gaussian_ids`/`_next_gaussian_id`): inherited unchanged from frozen C0 `densification_postfix`/`prune_points`.
- **Opacity reset**: inherited `reset_opacity` (min(opacity, 0.01) cap) — matches FastGS `reset_opacity` (0.01 cap).

---

## 3. Every FastGS policy component ported

| FastGS (@44e02a5) | Ported to | Notes |
|---|---|---|
| `utils/fast_utils.py:15` `sampling_cameras` | `rf0b_fastgs_model.sample_score_cameras` | 10 distinct train cams (FastGS pops from a fresh copy = without replacement; we use `rng.sample`). |
| `utils/fast_utils.py:62` `compute_gaussian_score_fastgs` | `rf0b_fastgs_model.compute_fastgs_scores` | Multi-view incidence (`importance_score`) + min-max normalized `sum_v photometric_loss_v * footprint_count_v` (`pruning_score`). |
| `scene/gaussian_model.py:528` `add_densification_stats` | `FastGSGaussianModel.add_densification_stats` (override) | Dual accumulators: signed→`xyz_gradient_accum`, abs→`xyz_gradient_accum_abs`. |
| `scene/gaussian_model.py:468` `densify_and_prune_fastgs` | `FastGSGaussianModel.densify_and_prune_fastgs` | grad-qualified clone/split, metric_mask filter, opacity+screen+world prune, budget prune, 0.8 opacity cap. |
| `scene/gaussian_model.py:431` `densify_and_split_fastgs` | `FastGSGaussianModel.densify_and_split_fastgs` | N=2 children, parent removed. |
| `scene/gaussian_model.py:455` `densify_and_clone_fastgs` | `FastGSGaussianModel.densify_and_clone_fastgs` | exact copy, parent kept. |
| `scene/gaussian_model.py:533` `final_prune_fastgs` | `FastGSGaussianModel.final_prune_fastgs` | opacity<0.1 OR pruning_score>0.9. |
| `scene/gaussian_model.py:279` `reset_opacity` | inherited (0.01 cap) | identical to FastGS. |
| train.py scheduling | trainer `c0fgs` branch | densify 500–15K int100; opacity reset every 3K; final_prune every 3K in (15K,30K). |
| Policy args | `FGS_*` constants | grad_thresh 0.0002, grad_abs_thresh 0.0012, dense 0.001, min_opacity 0.005, final min_opacity 0.1, loss_thresh 0.1, metric_mask>5, prune_score>0.9, remove_budget 0.5, opacity_cap 0.8, n_score_views 10. |

### Faithfulness fixes applied during port (vs. first draft)
1. **Dispatch fix**: the trainer calls `model.add_densification_stats(...)`; the override is now named `add_densification_stats` (not `…_fastgs`) so the dual accumulators are actually populated. Without this, `xyz_gradient_accum_abs` stays zero and all splits are silently disabled.
2. **Budget-prune fix**: port now prunes ONLY inside `if remove_budget:` (matching FastGS exactly — FastGS prunes nothing when `remove_budget==0`). First draft over-pruned the full mask in that case.
3. **Sampling**: without-replacement (matches FastGS pop semantics).
4. **Photometric weight**: uses FastGS loss form `(1-0.2)*L1 + 0.2*(1-SSIM)` per view (was L1-only).

---

## 4. Every unavoidable semantic mismatch

These are forced by holding the C0 renderer/loss fixed while porting a policy
that was designed against the original 3DGS rasterizer + graphdeco camera. Each
is documented; none are tunable post-hoc.

1. **`accum_metric_counts` screen-footprint reconstruction (renderer-side count).**
   FastGS obtains `accum_metric_counts` from a modified CUDA rasterizer that,
   with `get_flag=True` + a binary `metric_map`, returns a per-Gaussian count of
   high-error **covered** pixels. The frozen C0 renderer does NOT return this
   count and exposes `means2d` only as an adjoint buffer (zero in forward;
   verified signed nonzero = 0). We therefore reconstruct the equivalent
   per-Gaussian count with **minimum bookkeeping**: project world means to pixel
   coords via the frozen camera's `world_view_transform` + `K`
   (`_project_to_pixel`), take the per-Gaussian screen radius `r` from the C0
   render's `radii`, and count high-error pixels inside the footprint with an
   **integral-image (summed-area table) box-sum** over the binary metric map
   (`compute_fastgs_scores`). This reproduces a covered-pixel count (can exceed
   1 per view, exactly like FastGS), so `importance_score > 5` is a
   magnitude-threshold, not a view-majority vote. Residual approximation: we
   sum the axis-aligned bounding box `[px±r, py±r]` (area `(2r)²)`)
   rather than the exact elliptical footprint (`≈πr²`), inflating counts
   by ≤ ~27% uniformly; the box is used because the C0 render exposes no
   per-Gaussian footprint mask. See item #3 for the count-unit consequence.

2. **Signed→absgrad for the clone qualifier (+ pixel→NDC grad scaling).**
   FastGS `add_densification_stats` reads `viewspace_point_tensor.grad[:, :2]`
   (signed, pixel-space) for `xyz_gradient_accum` (clone qualifier,
   `grad_thresh=0.0002`) and `[:, 2:]` (abs) for `xyz_gradient_accum_abs` (split
   qualifier, `grad_abs_thresh=0.0012`). Under C0 H8_MR the signed `.grad` is
   structurally zero (verified), so BOTH accumulators read `means2d.absgrad`.
   Consequence: the clone qualifier threshold (0.0002) is applied to an absgrad
   signal. This is forced — the C0 renderer does not expose a contracted
   signed screen gradient. The split qualifier is unaffected (already abs).
   The C0 `means2d` gradient is pixel-space; we apply a fixed `width/2`,
   `height/2` factor before the norm to normalize to the NDC convention that
   FastGS's `grad_thresh`/`grad_abs_thresh` are calibrated against. This is
   deterministic bookkeeping (not tuned) and preserves the gradient direction
   and cross-Gaussian ordering.

3. **`importance_score` count unit (box vs ellipse).**
   FastGS `importance_score = floor(sum_v accum_metric_counts_v / n_views)` where
   `accum_metric_counts_v` is a true covered-pixel count. Our reconstruction
   (item #1) is also a covered-pixel count but over the footprint bounding box
   rather than the ellipse, so the per-Gaussian magnitude is inflated by a
   near-constant factor (`≈ (2r)² / πr² ≈ 1.27`). Because the
   factor is approximately scale-invariant across Gaussians of similar radius,
   the `importance_score > 5` threshold retains its relative ordering; the
   absolute cutoff is, however, not unit-identical to FastGS. This is a
   documented approximation forced by the frozen renderer (no footprint mask
   exposed).

4. **`photometric_loss` weight.**
   FastGS `compute_photometric_loss` = `(1-0.2)*L1 + 0.2*(1-fused_ssim)` per
   view. We use the identical functional form with the frozen SepSSIM (the
   `--loss sep` default). Only the SSIM backend differs from FastGS's fused_ssim;
   the loss *form* and weight (0.2) match.

5. **`sampling_cameras` RNG.**
   FastGS uses the global `random` module (process-seeded). We use a dedicated
   `random.Random(seed*1000003 + iteration)` for reproducibility and to avoid
   perturbing the frozen camera-sequence RNG. Distribution (10 distinct train
   cams, without replacement) is identical.

6. **Score-camera render cost (algorithmic, not semantic).**
   FastGS renders 10 extra views per densification event and per final-prune
   event. This is intrinsic to FastGS's multi-view-consistent policy and is
   retained (it is part of the algorithm's cost, not a backend substitution). It
   is included in wall-clock timing.

---

## 5. Exact source line / function mapping

FastGS `@44e02a5` → port:
- `utils/fast_utils.py:15 sampling_cameras` → `rf0b_fastgs_model.py:sample_score_cameras`
- `utils/fast_utils.py:62 compute_gaussian_score_fastgs` → `rf0b_fastgs_model.py:compute_fastgs_scores`
- `utils/fast_utils.py get_loss/_ndc_error_map` → `rf0b_fastgs_model.py:_ndc_error_map`
- `scene/gaussian_model.py:279 reset_opacity` → inherited `baseline/reference_v1/gaussian_model.py:614 reset_opacity`
- `scene/gaussian_model.py:342 _prune_optimizer` → inherited (C0)
- `scene/gaussian_model.py:364 prune_points` → inherited (C0) + `xyz_gradient_accum_abs` migration in subclass
- `scene/gaussian_model.py:383 cat_tensors_to_optimizer` → inherited (C0)
- `scene/gaussian_model.py:409 densification_postfix` → inherited (C0) + `xyz_gradient_accum_abs` reset in subclass
- `scene/gaussian_model.py:431 densify_and_split_fastgs` → `FastGSGaussianModel.densify_and_split_fastgs`
- `scene/gaussian_model.py:455 densify_and_clone_fastgs` → `FastGSGaussianModel.densify_and_clone_fastgs`
- `scene/gaussian_model.py:468 densify_and_prune_fastgs` → `FastGSGaussianModel.densify_and_prune_fastgs`
- `scene/gaussian_model.py:528 add_densification_stats` → `FastGSGaussianModel.add_densification_stats`
- `scene/gaussian_model.py:533 final_prune_fastgs` → `FastGSGaussianModel.final_prune_fastgs`
- `train.py:127-145` densify schedule → trainer `c0fgs` densify branch (`rf0b_trainer.py`)
- `train.py:147-158` opacity reset + final_prune → trainer `c0fgs` reset + final-prune branch

### Ported-vs-frozen decision table
| Aspect | Frozen C0 (`c0`) | Ported (`c0fgs`) | Source of truth |
|---|---|---|---|
| Renderer | `rasterize_gaussian_higs_dynamic` | SAME | C0 (frozen) |
| Loss | 0.8*L1+0.2*(1-SSIM) SepSSIM | SAME | C0 (frozen) |
| Optimizer/LR | C0 groups | SAME (inherited) | C0 (frozen) |
| Iterations | 30K | SAME | C0 (frozen) |
| Tile | 16 | SAME | C0 (frozen) |
| Holdout | i%8==0 | SAME | C0 (frozen) |
| Seed | 42 | SAME | C0 (frozen) |
| Densification statistic | absgrad norm ≥ 0.0008 | FastGS dual grad_thresh/grad_abs_thresh + metric_mask | FastGS (ported) |
| Prune policy | min_opacity + screen/world | FastGS opacity+screen+world + budget prune + 0.8 cap | FastGS (ported) |
| Final prune | none | every 3K in (15K,30K), opacity<0.1 OR score>0.9 | FastGS (ported) |
| Opacity reset | every 3K, 0.01 cap | SAME (identical in both) | both |