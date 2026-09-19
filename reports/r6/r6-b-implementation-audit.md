# R6-B Implementation Audit

## Scope

Audit of the Codex implementation commits:

- B0: `14849f6` — persistent rasterizer gradient buffers
- B1-v1: `b50afa7` — selectively clear prior raster rows
- reports: `4afcf97` — prototype gate status

## B0 Audit

### Implementation

`r6b_persistent_buffers.cuh` defines a `RasterBufferManager` that owns
capacity-managed flat storage for the five rasterizer backward outputs:

- `v_means2d`  [C, N, 2]
- `v_conics`   [C, N, 3]
- `v_colors`   [C, N, 3]
- `v_opacities` [C, N]
- `v_means2d_abs` [C, N, 2]

On each backward, `prepare()` returns active-shaped views into the persistent
storage.  If the storage is too small, it grows by 50% and forces a full clear.
B0 (`r6b_set_mode(1)`) always performs a full active-range zero-fill before
launching the unchanged raster backward kernel.

### Correctness

B0 is semantically equivalent to the baseline `torch::zeros_like` allocation:
the persistent buffer is zero-filled before each backward, so the raster
kernel sees the same initial state.  Capacity growth forces a full clear, so
no stale data from a smaller buffer can leak into a larger one.

### Verdict

B0 is correct and ready for A100 measurement.  It separates allocation/
reconstruction cost from zero-fill cost.

---

## B1-v1 Audit — CRITICAL FLAW CONFIRMED

### The flaw

B1-v1 stores the previous iteration's `flatten_ids` as `prev_touched_`:

```cpp
void finish(const torch::Tensor& flatten_ids) {
    prev_touched_ = flatten_ids;
}
```

In gsplat, `flatten_ids` is `[n_isects]` — a **tile–Gaussian intersection
list** (with duplicates), NOT a unique touched Gaussian-row list.  A Gaussian
that appears in multiple tiles has multiple entries in `flatten_ids`.

The selective clear then launches one clear thread per intersection:

```cpp
void selective_clear() {
    ...
    const int blocks = (prev_touched_.numel() + threads - 1) / threads;
    clear_previous_rows_kernel<<<blocks, threads>>>(prev_touched_...);
}
```

This means:

1. **O(n_isects) gradient writes** — one clear per intersection, not per
   unique touched Gaussian.  Since `n_isects >> N_touched` (typically 5-20x),
   this launches far more clear threads than necessary.

2. **Repeated clearing of the same row** — a Gaussian in K tiles has its
   gradient row cleared K times.  This is correctness-safe (every store
   writes zero) but wastes memory bandwidth and SM cycles.

3. **Full intersection tensor retained** — the entire `[n_isects]` int32
   tensor (4 × n_isects bytes) is retained until the next backward.  For the
   room scene at 15K, this is ~97 MB (25M intersections × 4 bytes).

### Why this is performance-invalid

The original observation is that dense zero-init of `[C, N, 11]` gradient
buffers is costly.  B1-v1 replaces this with clearing `[n_isects]` rows, but
`n_isects > C*N` in typical workloads (each Gaussian touches multiple tiles),
so B1-v1 may clear MORE gradient floats than B0's full clear.

Example: room 15K, N ≈ 1.5M Gaussians, C=1:
- B0 full clear: C×N×11 = 16.5M float writes
- B1-v1 intersection clear: n_isects × 11 ≈ 25M × 11 = 275M float writes
  (with duplicates, each clearing 11 floats)

B1-v1 is **worse than B0** in this regime, not better.

### Verdict

`B1-v1 = DO_NOT FORMALLY BENCHMARK`

The flaw is confirmed in the source code at commit `b50afa7`.

---

## B1-v1 → B1-v2 Repair

B1-v2 replaces the intersection-list clearing with a touched-mask approach:

1. **Scatter phase** (in `finish()`): scatter `flatten_ids` into a `[C*N]`
   bool touched mask.  O(n_isects) one-byte writes.  No `unique`, sort,
   or cumsum.

2. **Clear phase** (in `prepare()`): scan the `[C*N]` mask once and clear
   only touched gradient rows.  O(C*N predicate reads + N_touched × 11
   gradient writes).

Target complexity: `O(N_total predicate reads + N_touched gradient writes)`,
NOT `O(N_intersections gradient writes)`.

Extra metadata: `[C*N]` bool mask = C×N bytes (e.g. 1.5 MB for room 15K vs.
97 MB for B1-v1's intersection tensor).

### Status

B1-v2 implemented in commit `1be8f82` (and refined in `3329d4e`, `c7d05bc`).
Structural correctness tests and A100 benchmark pending.

---

## B1-v1 History

B1-v1 is labeled **SUPERSEDED**.  Its source is preserved in git history at
commit `b50afa7`.  The original report `r6-b1-selective-zero.md` described
the B1-v1 design and is retained for provenance.
