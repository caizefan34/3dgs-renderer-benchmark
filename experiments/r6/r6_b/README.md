# R6-B — rasterizer persistent-gradient prototype

This prototype touches only the unpacked 3DGS rasterizer backward outputs:
`v_means2d`, `v_conics`, `v_colors`, `v_opacities`, and `v_means2d_abs`.
Projection, SH, parameter `.grad`, optimizer, and training math are untouched.

## Modes

| Mode | `r6b_set_mode` | Description |
|------|---------------|-------------|
| baseline | 0 | Original gsplat `torch::zeros_like` allocation each backward |
| B0 | 1 | Persistent capacity-managed buffers + full active-range zero-fill |
| B1-v1 (SUPERSEDED) | 2 | Intersection-list clearing: stored `flatten_ids [n_isects]` and cleared one row per intersection |
| B1-v2 | 2 | Touched-mask clearing: scatter `flatten_ids` into `[C*N]` bool mask, scan mask, clear only touched rows |

B1-v1 and B1-v2 share `r6b_set_mode(2)`; the difference is in the
`r6b_persistent_buffers.cuh` implementation.

### B1-v1 flaw (SUPERSEDED)

B1-v1 stored `prev_touched_ = flatten_ids`, where `flatten_ids` is `[n_isects]`
— a tile–Gaussian intersection list with duplicates, NOT a unique touched
Gaussian-row list.  This caused:

- One clear kernel thread per intersection (O(n_isects) gradient writes)
- A Gaussian in multiple tiles had its gradient row cleared repeatedly
  (correctness-safe but performance-invalid)
- The entire intersection tensor was retained until the next backward

### B1-v2 repair

B1-v2 replaces the intersection-list clearing with a two-phase touched-mask
approach:

1. **Scatter phase** (in `finish()`): scatter `flatten_ids` into a compact
   `[C*N]` bool touched mask via a scatter kernel.  O(n_isects) one-byte
   writes, no unique/sort/cumsum.
2. **Clear phase** (in `prepare()`): scan the `[C*N]` mask once and clear
   the 11 rasterizer-gradient floats for a row ONLY if that row was
   previously touched.  O(C*N predicate reads + N_touched gradient writes).

Target complexity: `O(N_total predicate reads + N_touched gradient writes)`,
NOT `O(N_intersections gradient writes)`.

## Build

Build an isolated source tree; do not patch the baseline installation:

```bash
python experiments/r6/r6_b/prepare_r6b_source.py \
  --source ~/.local/lib/python3.10/site-packages/gsplat \
  --output /tmp/r6b/gsplat
PYTHONPATH=/tmp/r6b:$PYTHONPATH python experiments/r6/r6_b/benchmark_r6b.py ...
```

`r6b_set_mode(1)` enables B0.  `r6b_set_mode(2)` enables B1-v2.
`r6b_invalidate()` must be called after clone, split, or prune; it makes the
following backward perform a full clear.  Packed rendering remains baseline by
construction.

## Tests

| Script | Scope |
|--------|-------|
| `test_r6b.py` | Rasterizer-only: B0/B1 equivalence + basic stale-row |
| `test_sequential_r6b.py` | Sequential stale-gradient: camera A→bwd, camera B→bwd (no invalidate), verify A-touched/B-untouched rows are zero |
| `test_topology_r6b.py` | Topology transition: invalidate→bwd full-clear fallback, capacity growth, no stale rows |
| `correctness_r6b.py` | Checkpoint replay: full gradient comparison (mean2d, absgrad, xyz, SH, scaling, rotation, opacity) |
| `benchmark_r6b.py` | CUDA-event benchmark: T_bwd, T_iter, T_prepare, T_scatter, T_clear, metadata |
