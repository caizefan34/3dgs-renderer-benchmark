# Source code inspection record — GitHub master only, **not** MX runtime

Inspected 2026-10-08 through GitHub connector:

1. `benchmark/run_higs_train_benchmark.py` Git blob SHA: `93c983cedfd5648b5fbc6ceb0c1756e5e60486bb`. URL: https://github.com/caizefan34/3dgs-renderer-benchmark/blob/master/benchmark/run_higs_train_benchmark.py
    - `make_forward_fn(backend,...,tile_sampling_ratio,...sampling_mode)` supports explicit `tile_mask` for `rasterize_gaussian_higs_dynamic` when backend is `higs_dynamic_ts`.
    - `_error_guided_mask()` uses `torch.multinomial(p, k, replacement=True)` with repeat count `m`; `_importance_l1_loss()` uses weights `m/(k*p)`.
    - `_masked_l1_loss()` uses mean over selected pixels, which is not valid HT for unequal inclusion p or uneven border under general sampling designs.
    - `std_ll` is low-level **dense** conventional gsplat; don't pretend it is native sparse-tile control.
2. `scripts/higs/run_round57_sparse_px.sh` Git blob SHA: `25a4f281316c98c61c8fdea72de3216e7126b1df`. It uses 1920x1080, 3k, n_train4/n_eval3, R57 style LPIPS anchor; this is a historical lead, not permission to pool with fresh experiment or treat initial N as high-N state.
3. `paper/higs/tables/confirmatory-accel15-summary.json` Git blob SHA: `3dda30decf560a1c52584c5ec2cbcb303483dea8`. Separate 11-scene full-training gsplat reference; no direct attribution to cost-aware tile scheduling.

**Identity boundary:** The Git blob hash is a repository-file identity, not the active MX Git commit or CUDA binary SHA256. DSH must record current MX identities separately.
