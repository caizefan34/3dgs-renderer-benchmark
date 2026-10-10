# R19 Trainer Contract — TRAINABLE-HIGS-FULLTRAIN-R19-MX

## Objective
Prove whether the R18 Renderer FB speedup of HiGS translates into real 3DGS
training benefit. This is a **training-level** study, not a fixed-checkpoint
renderer microbenchmark. The trainer is plugged into the real 3DGS optimization
loop; only the renderer backend is swapped.

## Hard constraint: frozen R18 binaries
Both arms use the **exact frozen R18 CUDA binaries** (sha256-verified at runtime):

| Arm | Binary | sha256 | Entry call |
|-----|--------|--------|------------|
| A `std_ll` | `gsplat/csrc.so` | `40a009837e75620454059cfdbb5088cc08423f281e4c734da3053b727d39df50` | `gsplat.rasterization(absgrad=True, tile_size=16, packed=False, radius_clip=0.0, eps2d=0.1, render_mode="RGB")` |
| B `higs` | `gsplat/experimental/render/kernels/csrc.so` | `b64d02c1cdad51bfc9ff0c2c435f9a4c1d1603466700194fb30f1c2348bcf2c3` | `rasterize_gaussian_higs_frozen(freeze_topology=False, enable_culling=True, use_higs_culling=True, render_mode="RGB")` |

The `final30k_trainer.py` custom arm `.so` files (B1A_SO/C0_SO) are **NOT** used.

## Training semantics held identical (by construction)
The R19 trainer reuses the shared `baseline/reference_v1/` recipe verbatim — only
the renderer call differs. Identical across both arms:

- **Init + seed**: `create_from_pcd` from the real SfM sparse PLY (xyz + DC color);
  `torch.manual_seed(42); np.random.seed(42); random.seed(42)`.
- **Camera sequence**: fixed via `random.Random(config.seed)` shuffled stack;
  saved to `camera_sequence.npy` (identical for both arms).
- **Train/test split**: same `eval_indices` (every `n_cameras//10`-th camera).
- **Gaussian activation**: sigmoid (opacity), exp (scales), normalize (rotations)
  via `GaussianModel` getters.
- **Camera + normalization**: same `GTDataset`, same `scene_extent`.
- **Optimizer + LR schedule**: Adam, position_lr_init=1.6e-4, final=1.6e-6,
  delay_mult=0.01, max_steps=30000; feature_lr=2.5e-3, opacity_lr=2.5e-2,
  scaling_lr=5e-3, rotation_lr=1e-3, eps=1e-15.
- **SH degree schedule**: progressive, `oneupSHdegree` every 1000 iters.
- **Loss**: `(1-0.2)*L1 + 0.2*(1-SSIM)`, `lambda_dssim=0.2`.
- **Densification/pruning**: `densify_grad_threshold=0.0008` (absgrad),
  `percent_dense=0.01`, `min_opacity=0.005`, `opacity_reset_interval=3000`,
  `max_screen_size=20`, densify 500→15000, interval 100.
- **State update**: `optimizer.step(); zero_grad(set_to_none=True)`.

## The only difference: renderer backend
- `std_ll`: `gsplat.rasterization` (frozen `csrc.so` `40a00983…`).
- `higs`: `rasterize_gaussian_higs_frozen` (frozen HiGS `csrc.so` `b64d02c1…`).

## freeze_topology=False semantics (proven)
In R18, `freeze_topology=False` was a benchmark workaround to allow mixed
Gaussian counts across configs. In **training**, densification legitimately
changes the Gaussian count every 100 iters (500→15000), so `freeze_topology=False`
is **required** for correct semantics — it only disables the cross-call
Gaussian-count assertion. It does NOT change:
- culling behavior (`enable_culling=True, use_higs_culling=True` identical),
- gradient flow (both arms backprop the same loss),
- densification rules (driven entirely by shared `GaussianModel` code).

## The absgrad densification bridge (key adapter)
The reference_v1 recipe densifies on the **absolute** 2D-mean gradient
(`grow_grad2d=0.0008`). `gsplat.rasterization(absgrad=True)` natively populates
`meta["means2d"].absgrad`. The frozen HiGS function returns `means2d_proxy`
carrying the **signed** view-space gradient in `.grad` but does **NOT** expose
`.absgrad` (absgrad is in its unsupported-features set).

To keep densification **semantically identical** across both arms, the R19
trainer registers a hook on the HiGS `means2d_proxy` that populates
`means2d.absgrad = |grad|` from the signed accumulated gradient. Since
`means2d.grad` is already the sum of per-pixel contributions, `|grad|` is the
faithful absolute-gradient densification signal. Both arms therefore feed the
same `means2d.absgrad` into `add_densification_stats`, which prefers
`means2d.absgrad` and falls back to `means2d.grad`.

This is the minimal, faithful adapter: it changes **no** densification rule,
threshold, resolution, batch size, or loss. It only makes the HiGS arm expose the
same gradient signal the std_ll arm exposes natively.

## Forbidden (not done)
- Disabling densification, lowering resolution, changing batch size, or reducing
  the loss to make HiGS faster. None of these were performed.

## Component timing (P2)
Per-iteration CUDA-event phase timing, one sync/iter:
`forward, loss, backward, densify, optimizer`. Renderer fwd/bwd time =
`forward + backward + densify` (the renderer-owned portion); end-to-end =
total wall-clock. `S_renderer = T_std_renderer / T_higs_renderer` and
`S_e2e = T_std_training / T_higs_training` are reported separately.
Evaluation (PSNR/SSIM/LPIPS) time is recorded separately, NOT mixed into
training time.
