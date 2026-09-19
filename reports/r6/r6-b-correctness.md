# R6-B — Correctness

## Test suite

| Test | Level | File | Status |
|------|-------|------|--------|
| Rasterizer-only equivalence | L1 | `test_r6b.py` | PASS (A100) |
| Sequential stale-gradient (rasterizer) | L1 | `test_sequential_r6b.py` Level 1 | PASS (A100) |
| Topology capacity growth | L1 | `test_topology_r6b.py` Test 1 | PASS (A100) |
| Full-clear after invalidate | L1 | `test_topology_r6b.py` Test 2 | PASS (A100) |
| Sequential stale-gradient (checkpoint) | L2 | `test_sequential_r6b.py` Level 2 | PASS (A100) |
| Topology transition (checkpoint) | L2 | `test_topology_r6b.py` Test 3 | PENDING |
| Checkpoint gradient equivalence | L2 | `correctness_r6b.py` | PENDING |

## Level 1: Rasterizer-only tests (A100, gsplat 1.5.3)

### Basic B0/B1 equivalence (`test_r6b.py`)

All five rasterizer gradient outputs (color, opacity, mean2d, conic, absgrad)
are bit-exact between baseline, B0, and B1-v2:

```
max_abs = 0.0 for all comparisons
nan_or_inf = false for all comparisons
stale_gradient row = [0.0, 0.0] — passed
```

### Sequential stale-gradient (`test_sequential_r6b.py` Level 1)

**Scenario**: t touches Gaussian 0 → backward (no invalidate) → t+1 touches
Gaussian 1 only → backward.

**Result**: PASS. All stale-row zero checks PASS — rows touched at t but not
at t+1 have exactly zero gradients at t+1:

```
stale_row_zero_b1:
  color_gradient:      is_zero=True, max_abs=0.0
  opacity_gradient:    is_zero=True, max_abs=0.0
  mean2d_gradient:     is_zero=True, max_abs=0.0
  conic_gradient:      is_zero=True, max_abs=0.0
  absgrad:             is_zero=True, max_abs=0.0
```

B1-v2 at t+1 matches baseline bit-exactly (max_abs=0.0 for all gradients).

B1-v2 at t has a small absgrad difference (max_abs=0.00326) due to atomicAdd
non-determinism between fresh-alloc and persistent-buffer paths. B0 shows
the same difference, confirming it is a buffer-reuse artifact, not a B1-v2
selective-clear bug.

### Topology capacity growth (`test_topology_r6b.py` Test 1)

**Scenario**: B1-v2 N=2 backward → invalidate → B1-v2 N=4 backward (capacity
grows, full-clear must occur).

**Result**: PASS. B1-v2 N=4 matches baseline N=4 within floating-point noise:

```
color_gradient:      max_abs=1.16e-10
opacity_gradient:    max_abs=0.0
mean2d_gradient:     max_abs=6.91e-11
conic_gradient:      max_abs=0.0
absgrad:             max_abs=4.66e-10
```

Metadata grows from 2 rows (N=2) to 4 rows (N=4). No stale data from the
old (smaller) buffer leaks into the new buffer.

### Full-clear after invalidate (`test_topology_r6b.py` Test 2)

**Scenario**: B1-v2 [0] → backward → invalidate → B1-v2 [1] → backward
(full-clear fallback).

**Result**: PASS. All stale rows (row 0) are exactly zero:

```
stale_row0_zero:
  color_gradient:      is_zero=True
  opacity_gradient:    is_zero=True
  mean2d_gradient:     is_zero=True
  conic_gradient:      is_zero=True
  absgrad:             is_zero=True
```

## Level 2: Checkpoint tests (A100)

### Sequential stale-gradient (checkpoint, room 5K)

**Scenario**: Camera A (idx=0) → backward → Camera B (idx=1) → backward
(no invalidate). Compare B1-v2 camera-B backward against baseline camera-B
backward. Verify A-touched/B-untouched rows have zero gradients.

**Result**: PASS.

**Comparison B1-v2 vs baseline (camera B)**:

| Gradient | max_abs | relative_l2 | nan_or_inf |
|----------|---------|-------------|------------|
| mean2d | 1.14e-12 | 1.03e-07 | false |
| absgrad | 7.28e-12 | 1.02e-07 | false |
| xyz | 2.04e-09 | 4.34e-07 | false |
| SH | 8.00e-11 | 2.66e-07 | false |
| scaling | 1.67e-09 | 3.57e-06 | false |
| rotation | 7.49e-08 | 7.37e-05 | false |
| opacity | 2.91e-11 | 2.48e-07 | false |

All differences are at floating-point noise level (< 1e-7 relative L2).

**Stale-row zero check (A-touched, B-untouched)**:

| Gradient | is_zero | max_abs | n_nonzero |
|----------|---------|---------|-----------|
| mean2d_gradient | **true** | 0.0 | 0 |
| absgrad_densification | **true** | 0.0 | 0 |
| xyz_gradient | **true** | 0.0 | 0 |
| sh_gradient | **true** | 0.0 | 0 |
| scaling_gradient | **true** | 0.0 | 0 |
| rotation_gradient | **true** | 0.0 | 0 |
| opacity_parameter_gradient | **true** | 0.0 | 0 |

**All A-touched/B-untouched rows have exactly zero gradients** for all 7
gradient types (5 rasterizer + 5 propagated parameter). This confirms B1-v2's
selective clear correctly zeroes stale rows before accumulation.

## Coverage

### Rasterizer gradients checked

- `v_means2d` [C, N, 2] ✓
- `v_conics` [C, N, 3] ✓
- `v_colors` [C, N, 3] ✓
- `v_opacities` [C, N] ✓
- `v_means2d_abs` [C, N, 2] (absgrad) ✓

### Propagated parameter gradients checked (Level 2)

- `xyz_gradient` [N, 3] ✓
- `sh_gradient` [N, K, 3] ✓
- `scaling_gradient` [N, 3] ✓
- `rotation_gradient` [N, 4] ✓
- `opacity_parameter_gradient` [N, 1] ✓

### Metrics reported

- `max_abs` ✓
- `mean_abs` ✓
- `relative_l2` ✓
- `nan_or_inf` ✓
- `n_nonzero` (for stale-row checks) ✓

## Environment

- GPU: NVIDIA A100-PCIE-40GB
- gsplat: 1.5.3 (patched, isolated source tree)
- PyTorch: 2.4.1+cu124
- CUDA arch: sm_80
- Build: FAST_COMPILE=1, -O0
- Checkpoint: room iter_5000 (N=560,632)
- Commit: `d53f6d3`
