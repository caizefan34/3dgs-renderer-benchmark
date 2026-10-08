# R5 Patch Design

## Fix Strategy: Two-Phase Intersection Allocation

Based on root cause analysis, implement **Route A: Capacity Over-allocation Fix**.

## Design Overview

**Current Behavior:**
```python
# In isect_tiles
capacity = N * MAX_INTERSECTIONS_PER_GAUSSIAN  # Over-allocation
intersections = torch.empty(capacity, dtype=...)
# Fill intersections
```

**Proposed Fix:**
```python
# Phase 1: Count intersections
intersection_count = count_intersections(gaussians, tiles)

# Phase 2: Allocate exact capacity
intersections = torch.empty(intersection_count, dtype=...)

# Phase 3: Fill intersections
fill_intersections(gaussians, tiles, intersections)
```

## Implementation Details

### Phase 1: Count Intersections
- Add pre-pass to count actual Gaussian-tile intersections
- Use atomic operations or prefix sum for counting
- Expected overhead: ~5-10% additional kernel launch

### Phase 2: Exact Allocation
- Replace `capacity = N * MAX` with `capacity = actual_count`
- This reduces memory from ~9 KB/Gaussian to ~108 bytes/Gaussian (80x reduction)

### Phase 3: Fill (unchanged)
- Keep existing intersection filling logic
- No change to rendering or backward pass

## Expected Memory Impact

| N | Current Peak | Fixed Peak | Reduction |
|---|---|---|---|
| 580K | 12.07 GB | ~2.5 GB | 4.8x |
| 2.0M | 41.41 GB | ~6.5 GB | 6.4x |
| 4.4M | OOM (45 GB est) | ~12 GB | 3.8x |

**Note:** Exact reduction depends on actual intersection density.

## Correctness Guarantees

- ✅ No change to rendering algorithm
- ✅ No change to backward pass
- ✅ No approximation or truncation
- ✅ Exact same intersection records, just allocated efficiently
- ✅ No semantic change to HiGS hierarchy

## Implementation Risk

**LOW:**
- Only changes memory allocation strategy
- Does not modify algorithm logic
- Does not affect numerical results
- Can be reverted easily if issues arise

## Build and Deployment

1. Copy source to new directory
2. Modify `isect_tiles` function
3. Build with same parameters as baseline
4. Record new SHA256
5. Test on 580K workload for correctness
6. Test on 2.0M for memory reduction
7. Test on 4.4M for feasibility

## Patch File

See `patches/r5_intersection_fix.patch` for detailed diff.
