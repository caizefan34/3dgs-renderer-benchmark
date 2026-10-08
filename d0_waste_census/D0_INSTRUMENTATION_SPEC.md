# D0 — Instrumentation & Statistics Spec (frozen before any run)

Date: 2026-09-28. Status: FROZEN (no edits after first GPU run).
Bundle: `D0_3DGS_Training_Waste_Census_Bundle.zip` (extracted `D0_bundle_extract/`,
all 6 sha256 verified against `MANIFEST.json`).

This document freezes every definition D0 uses. It exists so that Stage 3/4
analysis cannot re-interpret measures after seeing numbers.

---

## 1. Identity

| artifact | sha256 |
|---|---|
| trainer `publication_trainer_p12.py` | `675b0267f3a1a6a0fa63fadb37c8536481c1c7a6de37a4b7f6ffa98a449050c5` |
| C0 ext `.so` | `d26cccf46c7932bfb5fd44ee05e553af27e17c93eaff6dacf3197258a513f11` |
| C0 core `.so` | `935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e` |
| fused-ssim `.so` | `dd6a4c799b98ff9fa82b4203acce3b36b1529c97155c0ef4cd50fd0f583d8ebb` |

All four re-verified on EPIC-05 2026-09-28 (paths in `D0_SOURCE_IDENTITY.json`).

**Hash reconciliation (recorded, not resolved by re-choice):** the P12
certification lane (`p12_support_audit/certification/P12_SOURCE_CERTIFICATION_REPORT.md`)
calls the mx-prebuilt pair `9baf8655…/361b216b…` the historical frozen C0 binary
and warns "do not mix". The actual P12A/P12B runs and `P12_SOURCE_IDENTITY.json`
used the EPIC rebuild pair `d26cccf4…/935ea6c0…` — which is exactly what the D0
prereg pins. D0 follows the prereg (the binaries the P12 numbers were produced
with). Both the prereg pair and the certification note are recorded in
`D0_SOURCE_IDENTITY.json`.

Frozen recipe: seed 42, 30000 iters, densify 500..15000 every 100,
opacity reset every 3000 (<15000), SH oneup every 1000, tile16, `--loss fused`
(`--eps2d 0.3`), eval SepSSIM (frozen evaluator), LPIPS(vgg) enabled,
`--final-eval all`.

Reference (clean) runs used for calibration & comparison:
`/root/epic05_p12/runs/p12a/p12a_c0f_s42_{room,bicycle}` (PUBLICATION grade,
same recipe). Frozen reference numbers:

| | room | bicycle |
|---|---|---|
| final_N | 855694 | 3663539 |
| final PSNR / SSIM / LPIPS | 32.11283 / 0.927911 / 0.300718 | 26.368364 / 0.835533 / 0.247354 |
| eval cameras | 311 | 194 |
| `camera_sequence.npy` sha256 | `a80127c542dc2859c0806b7fbf90b8c475a8e0536133be785ff38010630b98e9` | `8debe1c2244aca0994e9bd27ddfdd440fcc60f51341351d6c899b3c637c78e2d` |
| peak VRAM (reference) | 5.87 GB | 8.49 GB |

Init counts (SfM): room 112627, bicycle 54275.

---

## 2. Run matrix

| id | run | purpose |
|---|---|---|
| C3-room / C3-bicycle | clean trainer (frozen, byte-identical), 3000 iters | determinism arm (C3 = second clean 3000 for room only) |
| T3-room / T3-bicycle | traced trainer, 3000 iters | trace-neutrality smoke (Stage 0c) |
| CLEAN-room / CLEAN-bicycle | clean trainer, 30K, `--lpips --final-eval all` | "clean same-recipe replay" for trace validation |
| TRACE-room / TRACE-bicycle | traced trainer, 30K, `--lpips --final-eval all` | the instrumented trace (Stage 1/2) |

All runs on physical GPU 2 (training GPU), clean-GPU gate before launch,
`CUDA_VISIBLE_DEVICES` pinned. Smoke runs use `--iterations 3000`.

Trace-validity gates (Stage 0c and Stage 1/2), all must pass:
1. `camera_sequence.npy` of traced run byte-identical to clean replay
   (and its 30K prefix equals the historical reference sequence).
2. Traced vs clean 3000-step loss trace bit-identical (if the renderer is
   deterministic in this env; if C3-room vs C3-bicycle differ, the renderer is
   nondeterministic and the gate is "traced loss trace inside the clean-vs-clean
   spread" with the spread reported verbatim).
3. `final_N(trace)` within 0.5% of `final_N(clean)`.
4. Final quality inside the frozen C0 parity envelope (reference numbers above)
   and `all_losses_finite`.
5. Source diff limited to metadata/logging hooks (no semantic tensor touched).

If any gate fails → `D0_TRACE_INVALID`, STOP.

---

## 3. Instrumentation (metadata-only)

Files (local → EPIC `/root/epic05_d0/`):
- `d0_trainer.py` — copy of frozen trainer + D0 hooks (diff in `D0_SOURCE_DIFF.diff`)
- `d0_gaussian_model.py` — copy of `baseline/reference_v1/gaussian_model.py` + D0 hooks
- `c0_worktree_d0/` — copy of `/root/epic05_frozen/higs_c0_final30k_worktree`
  with 5 insertions in `gsplat/experimental/render/functional/gaussian_inference.py`
  exposing (a) `tiles_per_gauss` (already computed, previously discarded),
  (b) visible-row pixel-space `means2d`, (c) `visible_ids`, (d) visible-row
  `depths`, (e) `f9_enabled`. `.so` files unchanged.
  Sites (frozen-file line numbers in parentheses, all metadata-only):
  R1 `ctx.d0_tiles_per_gauss`/`ctx.d0_depths_vis` at the `intersect_tile`
  result (2178, inside `_native_forward_capture`); R2 `ctx.d0_means2d_px =
  captured[0]` before `return render_colors, ...` (2273, same function);
  R3 the four `d0_*` keys in the main metadata dict (2028); R4 the no-visible
  early-return dict (1911); R5 the four `d0_*` keys in `_higs_dynamic_forward`'s
  metadata dict (2984, the entry the C0 recipe calls) — note
  `_higs_frozen_forward`'s metadata dict is textually near-identical there and
  is deliberately NOT patched (not on the dynamic path); the R5 anchor carries
  the `"scene_version": scene_version,` line to remain unique.
  The tracer hard-asserts `tiles_per_gauss`/`means2d_px` presence on the first
  visible iteration, so a missing hook fails fast instead of silently
  under-accounting.

Rules honored: no change to parameter tensors, optimizer updates, camera order,
densification/prune decisions, or training RNG consumption. UID counter is a
plain Python int (no RNG). No `torch.rand*`/`np.random`/`random` call is added
anywhere on the training path. All added tensors are `detach()`ed metadata.

Renderer hook semantics (verified by reading the frozen source):
- `tiles_per_gauss[k]` ↔ `visible_ids[k]` (visible-row space; gatherless F9 path).
- `metadata["n_isects"]` = exact scene intersection count `I_t` of the
  differentiable render (sampled_tile_ratio == 1.0 in the frozen recipe).
- cross-check every iteration: `Σ tiles_per_gauss == n_isects`.

### 3.1 UID / lineage records

Per Gaussian: `uid` (0..U−1 creation order; init = 0..N_init−1), `parent_uid`
(−1 for init), `birth_iter` (0 for init), `birth_event_id`, `birth_type`
(0 init, 1 clone, 2 split_child), `death_iter` (−1 alive), `terminal_reason`
(0 alive, 1 opacity, 2 screen, 3 world, 4 split_parent_removed).

Split children ordering: `repeat(N=2)` order → children `(2j, 2j+1)` of parent
`j`; split parents die with reason 4 at the same iteration.

Iteration semantics (frozen): a row born in the densify event of iteration `t`
**is** updated by the optimizer at `t`; a row removed (pruned / split parent)
at iteration `d` is **not** updated at `d`.

### 3.2 Per-uid persistent state (GPU, growable)

| name | dtype/shape | semantics |
|---|---|---|
| `vis_stamp` | uint16 `[U,128]`, init `0xFF00` | observation stamp at ring slot `t % 128`; visible at `t` iff slot valid |
| `cam_bits` | uint64 `[U, ceil(ncam/64)]` | bit `c` set iff ever visible from camera `c` (pre-current-iteration history) |
| `cam_count` | uint32 `[U]` | distinct cameras seen (incremental) |
| `hist_n` | uint32 `[U]` | count of visible observations (slot = `hist_n % 32`) |
| `res_mean_ring` | float32 `[U,32]` | residual channel-mean at parent center tile (px) |
| `res_p90_ring` | float32 `[U,32]` | residual channel-p90 at parent center tile (px) |
| `tiles_ring` | float32 `[U,32]` | tiles_per_gauss per visible observation (shared slot with res rings) |
| `dir_ring` | float32 `[U,32,2]` | unit 2-D signed view-space gradient direction per visible observation |
| `tiles_cum` | int64 `[U]` | cumulative tiles_per_gauss over **all** iterations |
| `last_tiles` | float32 `[U]` | tiles_per_gauss at the most recent visible observation (NaN dense-init) |
| `last_depth` | float32 `[U]` | view-space depth from the renderer's `depths` at the most recent visible observation |
| `birth_feat` | float32 `[F=30, U]` | pre-birth features (filled only at birth) |

Ring slots are **written every visible observation** (NaN when a component has no
value), so stale values can never leak into a later window. Weight-32 windows are
`nanmean` over the ≤32 stored slots (`hist_n` counts observations).

Visibility at iteration `t` = `visible_ids` of the training render (the render's
participating set; `n_visible` in the trajectory is `visible_ids.numel()`), and
the renderer's visible-row arrays (`tiles_per_gauss`, `depths`, pixel-space
`means2d`) are aligned to it index-for-index. At iteration 1 the tracer asserts
the visible-space `means2d` is pixel-space (median |x| > 10 px).
Residual stats: `R = |image−gt|.mean(channel)`, tile grid 16×16 (68×120):
`tile_mean = avg_pool2d(R,16)`, `tile_p90 = kthvalue` at k=⌈0.9·256⌉ of the 256
pixel values per tile. A visible uid's value = the tile containing its
pixel-space `means2d`. If `means2d.grad` is `None` the direction features are
NaN for that iteration (logged once, `grad_missing_iters`).

### 3.3 Pre-birth features (frozen list, F=30)

Captured **strictly before materialization** at the densify selection.

Parent state (10): `parent_age` (child_iter − parent_iter), `parent_opacity`,
`parent_xyz_norm`, `depth_last_visible` (renderer view-space depth at the most
recent visible observation), `scale_x/y/z` (activated),
`scale_volume`, `radius_last_px`, `sh_degree_active`.
(never-visible → NaN; init parents get `birth_iter=0`)

Densification signals (7): `grad_accum_row`, `grad_denom_row`,
`grad_mean_norm` (`accum/denom`), `threshold_margin` (mean_norm − 0.0008),
`max_radii2D_row`, `last_absgrad_norm_px` (the exact per-row value
`add_densification_stats` accumulates: `‖absgrad · [W/2, H/2]‖` at the current
iteration), `branch` (1 clone / 2 split).

Historical visibility (6): `vis_count_32` (stamp-ring count over
`(t−31..t)`), `vis_count_128`, `vis_frac_32`, `vis_frac_128`,
`n_distinct_cameras` (pre-update value), `view_dir_spread_32`
(`1 − ‖Σ d̂‖/count` over the last ≤32 visible observations; 0 = one direction,
1 = isotropic; NaN if none).

Historical reconstruction (7): `res_mean_current_tile`,
`res_p90_current_tile` (both at the current iteration's camera; NaN if the
parent is not visible in it), `res_roll_mean_32`, `res_roll_var_32`
(over the last ≤32 visible observations), `grad_dir_stability_32`
(`‖Σ d̂‖ / count` over the last ≤32 visible observations; 1 = consistent,
0 = cancelling; NaN if none), `footprint_tiles_last_obs`
(tiles_per_gauss at the most recent visible observation),
`footprint_tiles_last32_mean` (rolling mean over the last ≤32 visible
observations). F = 10 + 7 + 6 + 7 = **30**.

Order of feature columns is the order above, frozen in
`D0_BIRTH_TABLE.parquet` (`f00..f29`).

### 3.4 Population trajectory (every iteration, exact)

`iter, N_t, n_visible, I_t (=n_isects), I_t_sumchecked (=Σ tiles_per_gauss),
births_clone, births_split, prunes_total, prunes_opacity/screen/world,
split_parent_removed, opacity_reset (0/1), loss, phase_ms{forward,loss,backward,
densify,optimizer}, total_iter_ms`.
Note: deaths are counted in the iteration they occur (death_iter = that iter).

### 3.5 Sampled renderer accounting

- Sample set: `S = {t ∈ [1,30000] : t % 20 == offset}`, weights w = 20.
- `offset = 1 + int(sha256("D0:"+scene), 16) % 20` → **room 11, bicycle 15**
  (both give 1500 samples). Derived from the scene name only — not training RNG.
- At each `t ∈ S` dump per-visible-uid arrays `(uid, tiles)` (nonzero tiles only)
  to `d0_cost_samples/sample_{t:05d}.npz` plus scalars
  `(t, I_t, sum_tiles, N_t, n_visible)`.
- `D0_RENDER_COST_SAMPLES.parquet` is assembled from these shards in Stage 3
  (kept on EPIC-05 if > ~200 MB; hash + path recorded in the return package).
- Reconstruction check (gate): for the union of all sampled iterations,
  `1 − Σ_s sum_tiles(s) / Σ_s I_t(s) ≤ 3%` must hold, plus the every-iteration
  self-check `Σ tiles_per_gauss == n_isects` at every iteration (mismatch count
  must be 0). If violated → accounting invalid, STOP before verdict.

---

## 4. Accounted quantities (Stage 3)

Definitions (all restricted to post-init birth roots unless stated; see 4.5):

**Lineage** (per prereg §3): root = a birth (clone child or split child); the
lineage = root + all descendants born from it (clone children and split
children of any member). Init Gaussians are **their own root class** (`init`),
handled in the sensitivity variant 4.5.

- `has_final_descendant`, `n_final_descendants`, `lineage_death_iter`
  (last member death; −1 if alive), `lineage_lifetime` (last member death −
  root birth), `lineage_population_integral` = Σ over members of
  `max(0, min(last_alive_iter, 30000) − birth_iter + 1)` (Gaussian-iterations,
  exact — equals parameter-row update count since every extant row is updated
  every iteration). Optimizer-state byte proxy = count × 708 B (per-row:
  (3+48+3+4+1) params × 4 B × 3 tensors (value + 2 Adam moments)).
- `RETROSPECTIVELY_DEAD` iff no member alive at 30K.

**r_I_dead** (pre-t_star): over sampled iterations `t ≤ t_star`,
`Σ_{s} Σ_{uid ∈ dead} tiles(s,uid) / Σ_s Σ_uid tiles(s,uid)` (weight 20 cancels).

**r_N_dead** (pre-t_star): `Σ_{dead roots} pop_integral(≤ t_star) / Σ_{all roots}
pop_integral(≤ t_star)` where `pop_integral(≤T)` per root = Σ over members of
`max(0, min(last_alive, T) − birth + 1)`. Exact (no sampling).

**Headroom** (prereg §8), with `f_FB = 13.915/19.369`, `f_OPT = 4.140/19.369`:

```
R_upper = [ (f_FB·r_I_dead + f_OPT·r_N_dead) · t_star
            + (f_FB + f_OPT) · (30000 − t_star) ] / [ (f_FB + f_OPT) · 30000 ]
S_upper = 1 / (1 − R_upper)
```

The tail term treats **all** calibrated work after `t_star` as removable
(optimistic), the pre-tail term only the dead-lineage fraction — no double
counting by construction (disjoint iteration ranges).

**Bootstrap** (CI of R_upper): 2000 resamples of the pre-tail sampled
iterations with replacement (`numpy.random.default_rng(20260928)`), ratio
estimator recomputed per resample, `r_N_dead` held exact; CI = [2.5%, 97.5%]
percentiles.

**t_star**: earliest checkpoint in {5000, 10000, 15000, 20000, 25000} with
`PSNR ≥ PSNR_final − 0.10`, `SSIM ≥ SSIM_final − 0.003`,
`LPIPS ≤ LPIPS_final + 0.005` (final = same run's 30K full-camera eval); else
`t_star = 30000`.

Gates: `R_upper_ci95_low ≥ 0.70` on both scenes → headroom;
`< 0.60` on either scene → `D0_NO_3X_HEADROOM`; else borderline.

### 4.5 Init-root sensitivity (documented interpretation)

Prereg §3 scopes the per-root table to "post-init birth roots"; the main
accounting therefore uses post-init roots only (this is the **conservative**
choice: init roots' work is excluded from dead work and stays in the
denominator). A sensitivity table additionally treats init Gaussians as roots
(same extinction rule) and reports its `r_*_dead`, `R_upper`, `S_upper`.
The **verdict uses the post-init partition**; the sensitivity variant is
reported but never substituted. If the two partitions straddle a gate
threshold, the verdict stays on the post-init number and the borderline is
recorded verbatim.

---

## 5. Predictability (Stage 4, only if headroom on both scenes)

- Label: `RETROSPECTIVELY_DEAD` (post-init birth roots).
- Features: the F=30 pre-birth features (Section 3.3), no others.
  Rows with NaN features: NaN → training-set column mean (computed on the
  training scene only); constant columns dropped (training scene only).
- Model 1: standardized (z-score from training scene) L2 logistic regression,
  `sklearn.linear_model.LogisticRegression(C=1.0, solver="lbfgs",
  max_iter=5000, class_weight=None)`, no search.
- Model 2: best single scalar = argmax training-scene AUROC over the F features
  (ties → lowest feature index); direction fixed by training scene.
- Cross-scene only: room→bicycle and bicycle→room. No test-scene tuning.
- Threshold (per model, per direction): among candidate thresholds = unique
  training-scene scores, choose the **highest recall** with survivor-lineage
  FPR ≤ 1% on the training scene; ties → higher threshold.
- Test metrics: AUROC, AUPRC, survivor FPR, dead-lineage birth-count recall,
  cost-weighted dead recall with weight `w_l = f_FB·20·Σ_s tiles(l) +
  f_OPT·pop_integral(l)` over the full run (same calibration as R_upper).
- STRONG (per direction, per model): AUROC ≥ 0.85, test survivor FPR ≤ 2%,
  cost-weighted dead recall ≥ 50%. STRONG for a direction = STRONG for the
  frozen model 1 (logreg); model 2 is diagnostic.

---

## 6. Final-state audit (Stage 5, descriptive)

16 deterministic audit cameras: `round(k·(n_cam−1)/15)`, k=0..15, distinct,
ascending (room: n_cam=311, bicycle: 194). On the 30K state, per surviving
lineage:
`visibility_count` (# audit cameras visible), `tile_intersections_total`,
`alpha_footprint_proxy` = Σ_cam `σ(opacity_i)·min(π·r_max_px², W·H)`.
RGB contribution proxy: not cheaply exposed by the C0 renderer → recorded as
unavailable (with the reason) rather than approximated.
No claim of "safe to remove" is made.

---

## 7. Ops / hygiene

- EPIC-05 `/root/epic05_d0/` (scripts, worktree copy, runs, artifacts, logs).
- Deploy: local → scp → `ssh EPIC-05 'bash /root/epic05_d0/scripts/<x>.sh'`
  (CRLF stripped locally; `bash -n` on remote before each run).
- Clean-GPU gate: 10 × 30 s checks on GPU 2 before every run (no compute PIDs,
  ≤ 4 MiB used, 0% util).
- Contamination: GPU UUID + foreign-PID snapshot at start/end (PID-namespace
  ghost PIDs from `nvidia-smi` ignored; ownership judged by GPU UUID),
  `/root/08_watchdog.py` jsonl evidence, peak VRAM, start/end times,
  environment → `D0_CONTAMINATION.json`.
- Timing: instrumented wall time is **not** evidence; phase times recorded but
  labeled FUNCTIONAL_ONLY. `f_FB`/`f_OPT` are the frozen P12 calibration values.
- No git commit (campaign protocol).
