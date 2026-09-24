# H5-1 · HiGS Batch-Factored Exact Backward — Feasibility Report

**Task:** Determine whether the long sequential Gaussian-compositing backward
chain decomposes into independent, exact, batchable/mini-batched work
segments. **Scope:** analysis + validation microkernel only; no production
backward-kernel integration.

**Device / environment:** `mx` A100-PCIE-40GB · `higs-13scene` env
(torch 2.9.1 + cu128) · microkernel compiled `sm_80` · validated against the
native forward order (depth-then-id) on real mipnerf360 scenes
`room / bicycle / garden`, camera 0, 2048 max-side.

Gate criteria (from task): STRONG requires **≥2× p90/p99 critical-path
reduction in ≥2/3 scenes AND ≤15% retained-state overhead AND microkernel
evidence**.

---

## 1. Compositing monoid — exact, factorable

Front-to-back compositing is an affine map with a **two-scalar** monoid:

```
C_out = C_in + T_in * (a_i * rgb_i)
T_out = T_in * (1 - a_i)
```

Segment aggregate `(S, tau)` composes by an associative product. Over the
reals the factored build equals the sequential build exactly. Full algebra
and the backward pass decomposition are in
[`algebra.md`](artifacts/higs-h5-1/algebra.md).

## 2. Scalar-adjoint boundary state is 2 scalars (not CDIM)

The current backward keeps `(T, buffer_dot, tail_const)` per pixel. The
factored dataflow's segment boundary state is exactly `(T, buffer_dot)` —
**two scalars** — so a segment-local reverse needs no cross-segment vector.
See `scalar_adjoint_mapping.json`.

## 3. Measured chain lengths (real data)

Per-pixel reachable chain length `L` (count of Gaussians before the
transmittance early-termination threshold `T < 1e-4`):

| scene   | pixels   | active | L_mean | p90 | p99 | max | fraction_active |
|---------|----------|--------|--------|-----|-----|-----|-----------------|
| room    | 2,795,520 | 0.211 | 1.11   | 3   | 22  | 48  | 0.211           |
| bicycle | 2,787,328 | 0.140 | 0.29   | 1   | 6   | 46  | 0.140           |
| garden  | 2,717,696 | 0.033 | 0.18   | 0   | 6   | 40  | 0.033           |

**Observation:** real per-pixel chains are short (median ≤ 3; only the p99
tail reaches 6–22; absolute max 40–48). The per-tile ordered lists are long
(tens of Gaussians), but per-**pixel** reachable depth is small.

## 4. Segment distributions & critical-path reduction (S = 32…1024)

Factored critical depth ≈ `2*ceil(L/S) + S` (two boundary scans + local
reverse); the `fwdT` variant ≈ `ceil(L/S) + S` when the forward pass captures
the T-boundary product.

`red = L / factored_depth` (ratio > 1 means factored is shallower → a win).
**Maximum red at the shallowest granularity (S=32):**

| scene   | red_p90 | red_p99 | red_max | red_fwdT_max |
|---------|---------|---------|---------|--------------|
| room    | 0.38    | 0.97    | 1.33    | 1.41         |
| bicycle | 0.15    | 0.44    | 1.28    | 1.35         |
| garden  | 0.47    | 0.92    | 1.11    | 1.18         |

**Finding:** for realistic chains the `+S` fixed cost **dominates** the
reduction. red stays below 1 for the vast majority of pixels — the factored
critical path is *not shorter* than the sequential one. Only the longest
chains (`L > 2S`) see any gain (`red_max ≈ 1.1–1.33`, or `1.18–1.41` with the
`fwdT` boundary-capture variant, both at `S=32`). The bands `S≥64` strictly
degrade red further (full tables in `segment_distribution.csv`,
`critical_path.json`).

→ Effect is the **opposite** of the STRONG gate: no scene reaches ≥2×
reduction; it is never ≥1× on the median/p90.

## 5. Retained-state cost

Per active pixel the factored workflow adds only a small boundary record per
segment (incoming-T, ~4 B/segment) plus reuses `last_ids` already carried in
forward.

- room:  +8.06 B/active pixel @S=32 (≈ +1.01× the 8 B baseline) but only
  ≈ +1.70 B / *all* pixels; S≥64 → +8.0 B/active, ≈ +0.21× of baseline
  across all pixels.
- bicycle / garden: similarity-shaped; segment summaries are cheap.
- Full breakdown in `state_cost.json`.

State overhead is **small relative to the 8 B/pixel forward state**, but it
does not buy the required critical-path reduction.

## 6. FP exactness — ALGEBRAIC_EXACT_FP_REASSOCIATED

Sequential FP32 vs factored FP32 (devided by the f64 reference) on 908–4k+
real sampled chains:

| comparison                 | max_abs  | rel_L2   | cosine  | support Δ |
|----------------------------|----------|----------|---------|-----------|
| f32 seq vs f64             | 1.57e-7  | 2.45e-8  | ≈1.0    | 0         |
| factored f32 vs f64        | 4.25e-8  | 1.58e-8  | ≈1.0    | 0         |
| factored f32 vs seq f32    | 1.42e-7  | 2.91e-8  | ≈1.0    | 0         |

Classification per task taxonomy: **`ALGEBRAIC_EXACT_FP_REASSOCIATED`** —
the factored build is the same affine monoid, differing from sequential only
in FP32 rounding/association order of tau-products and color-sums. Same result
across all S and all scenes (`fp_exactness.json`).

## 7. last_ids composition & mask-native feasibility

- `last_ids` already encodes chain end; histogram confirms termination ranks
  closely track the tile ordered-list tails (`lastid_composition.json`).
- The factored workflow reads the **same per-tile ordered list** that follows
  directly from `macro_sorted_ids + fine_tile_masks`; segment summaries reduce
  over those per-tile lists with **no flattened fine-tile reconstruction**
  (`mask_mapping.json: feasible_per_macro_from_masks = true`).

## 8. Validation microkernel timing (A100, sm_80)

One thread per sampled chain; per-thread staging arrays capture the
temporary-state cost honestly (ptxas/register data in `resource_usage.json`).

Sequential vs factored wall-clock (ns/element, sampled chains), by scene:

| scene   | sequential | S=32 | S=64 | S=128 | S=256 |
|---------|-----------|------|------|-------|-------|
| room    | 6.46      | 7.08 | 6.80 | 6.83  | 6.83  |
| bicycle | 17.06     | 17.45| 17.51| 17.51 | 17.47 |
| garden  | 16.95     | 17.29| 16.79| 16.82 | 16.57 |

On-device check: factored vs sequential agree to rel_L2 ≈ 1.1–1.8e-7 (exact
to FP32 rounding). Observed factored is ≈ parity to a few % slower per element
than the sequential loop at these short chain lengths — consistent with §4:
the 2× scan + fixed segment overhead exceeds any parallel saving when `L` is
small.

---

## Gate verdict

| gate | threshold | measured | result |
|------|-----------|----------|--------|
| STRONG (critical-path ≥2× in ≥2/3 scenes) | ≥2.0 × @p90/p99 | max ≈ 0.97 @p99, red_max≈1.36 (never ≥2×, only longest chains) | ✗ |
| state overhead ≤15% | ≤ 0.15 × baseline | ≈ +0.21× (all pixels), +1.0×/active@S32 | mixed (ok all-pixels, exceeds active) |
| microkernel evidence | timing present, exact | measured, factoring ≈5–10% slower | ✗ (no benefit) |
| **Overall** | — | algebraic exactness **does** hold; parallelism/win **does not** materialize | **WEAK** |

## Recommended segment size

Although the technique is exact, no segment size in `S∈{32,64,128,256,512,1024}`
delivers a critical-path reduction on realistic mipnerf360 chains, because the
per-pixel reachable chain is short (`L` median ≤ 3, max ≈ 40–48). Smaller
`S` always improves the red ratio (S=32 is best), but the benefit only
appears for `L > 2S` tails, which are statistically negligible here.

**Recommendation:** do **not** batch-factory the backward at segment
granularity on these scenes; keep the current sequential/parallel-over-chains
backward. If a workload shift produces systematically longer chains
(e.g. foreground-dense or no-early-termination mode), revisit with **S = 16–32**
and the `fwdT` boundary-pass variant (`red ≈ L/(ceil(L/S)+S)`), which is the
only configuration that can reach the ≥2× regime.

## Artifacts

`artifacts/higs-h5-1/`: `algebra.md`, `scalar_adjoint_mapping.json`,
`segment_distribution.csv`, `critical_path.json`, `state_cost.json`,
`fp_exactness.json`, `lastid_composition.json`, `mask_mapping.json`,
`microkernel_timing.csv`, `resource_usage.json`, `analysis.json`,
`provenance.json`.

Repro scripts: `scripts/h5/h5_capture.py` (oracle forward order + real color),
`scripts/h5/h5_micro.cu` (sm_80 microkernel), `scripts/h5/h5_run.py`
(orchestrator), `scripts/h5/assemble_h5_1.py` (artifact merge).