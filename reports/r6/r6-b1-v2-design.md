# R6-B1-v2 — Touched-Mask Selective Clear Design

## Status: B1-v1 SUPERSEDED

B1-v1 (commit `b50afa7`) stored `prev_touched_ = flatten_ids` (an intersection
list with duplicates) and cleared one gradient row per intersection.  This was
O(n_isects) gradient writes — potentially MORE than B0's full clear.

B1-v2 replaces this with a two-phase touched-mask approach.

## Design

### Phase 1: Scatter (in `finish()`)

After each raster backward, scatter `flatten_ids [n_isects]` into a compact
`[C*N]` bool touched mask:

```cuda
__global__ void build_touched_mask_kernel(
    const int32_t* flatten_ids, int64_t n_isects, bool* mask
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_isects) return;
    mask[flatten_ids[i]] = true;  // idempotent — duplicates safe
}
```

Complexity: O(n_isects) one-byte writes.  No `torch.unique`, sort, cumsum,
or CPU synchronization.

### Phase 2: Selective clear (in `prepare()`)

Before the next backward, scan the `[C*N]` mask once and clear only touched
gradient rows:

```cuda
__global__ void clear_touched_rows_kernel(
    const bool* touched_mask, int64_t n_rows,
    float* means2d, float* conics, float* colors,
    float* opacities, float* means2d_abs, bool absgrad
) {
    int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    if (!touched_mask[row]) return;  // skip untouched rows
    // zero 11 gradient floats for this row
}
```

Complexity: O(C*N predicate reads + N_touched × 11 gradient writes).

### Target complexity

```
O(N_total predicate reads + N_touched gradient writes)
```

NOT:

```
O(N_intersections gradient writes)   [B1-v1]
O(N_total × 11 gradient writes)      [B0 / baseline]
```

### Metadata overhead

| Component | B1-v1 | B1-v2 |
|-----------|-------|-------|
| Retained tensor | `flatten_ids [n_isects]` int32 | `touched_mask [C*N]` bool |
| Bytes (room 15K) | ~97 MB (25M × 4B) | ~1.5 MB (1.5M × 1B) |
| Lifetime | until next backward | until next backward |
| Preparation | none | scatter kernel (O(n_isects) 1B writes) |
| Clear | O(n_isects × 11) float writes | O(C*N reads + N_touched × 11) float writes |

### Invariant

The touched mask is a conservative superset of actually-touched rows: an
intersection rejected by alpha or mask may still set `mask[row] = true`, so
its gradient row is cleared even though it was zero.  This preserves
exactness (clearing a zero row is harmless) while avoiding the cost of
precise touched-set computation.

### Topology handling

- **Growth / shape change**: `invalidate()` sets `force_full_clear_ = true`
  and clears `prev_touched_`.  The next backward performs a full clear.
- **Explicit `r6b_invalidate()` after clone/split/prune**: same as above.
- **`runtime.install_topology_guard()`**: wraps `densification_postfix` and
  `prune_points` to call `invalidate()` automatically.
- **Packed rendering**: falls back to baseline because `enabled_for()`
  returns false when `means2d.dim() != 3`.

### No hidden preprocessing

B1-v2 does NOT use:
- `torch.unique(flatten_ids)` — O(n_isects log n_isects) sort + unique
- `torch.sort` / `torch.cumsum` — prefix-sum over intersection list
- CPU synchronization — `.item()`, `.cpu()`, `.tolist()`
- Any intersection-sized temporary beyond the scatter kernel itself

The scatter kernel is a single-pass O(n_isects) write with no atomics
(idempotent stores of `true`).

## Implementation

| File | Role |
|------|------|
| `r6b_persistent_buffers.cuh` | Declarations only (safe for host compiler) |
| `r6b_persistent_buffers.cu` | CUDA kernels + RasterBufferManager implementations (nvcc) |
| `prepare_r6b_source.py` | Patches gsplat source (auto-detects legacy/modern layout) |
| `runtime.py` | Python runtime: configure, invalidate, timing/metadata accessors |
| `benchmark_r6b.py` | CUDA-event benchmark: T_bwd, T_iter, T_prepare, T_scatter, T_clear, metadata |
| `test_sequential_r6b.py` | Sequential stale-gradient correctness test |
| `test_topology_r6b.py` | Topology-transition correctness test |

## Compilation note

The CUDA kernels and kernel-launching member functions are in a separate
`.cu` file compiled by `nvcc`.  The `.cuh` header contains only declarations
and is safe to include from `.cpp` files compiled by the host `c++` compiler.
This avoids the "blockIdx not declared" error that occurs when CUDA syntax
appears in a host-compiled translation unit.
