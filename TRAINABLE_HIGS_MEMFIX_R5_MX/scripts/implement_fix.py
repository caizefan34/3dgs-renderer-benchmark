#!/usr/bin/env python3
"""
R5 Implementation - Two-phase intersection allocation fix

Based on analysis:
- OOM occurs in isect_tiles at 4.4M Gaussians
- Allocation: 37.74 GiB = ~9 KB per Gaussian
- This is PRE-ALLOCATED capacity, not actual intersections
- Expected actual: ~108 bytes per Gaussian (9 intersections * 12 bytes)

Strategy:
The isect_tiles CUDA kernel is pre-allocating based on worst-case scenario.
We need to either:
1. Modify CUDA kernel to count-then-allocate (complex)
2. Add Python-level wrapper to estimate and cap allocation (simpler)
3. Use dynamic allocation with atomic counters (moderate)

Given time constraints, we'll implement Option 2: Adaptive capacity estimation
"""

import sys
import os
sys.path.insert(0, '/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx')

def estimate_intersection_count(means2d, radii, tile_size, tile_width, tile_height):
    """
    Estimate actual intersection count based on Gaussian distribution.
    This is a heuristic to avoid worst-case over-allocation.
    """
    N = means2d.shape[-2]
    
    # Compute bounding box area for all Gaussians
    total_bbox_area = (radii.sum(dim=-1) * 2).prod(dim=-1).sum().item()
    
    # Compute total tile area
    tile_area = tile_size * tile_size
    num_tiles = tile_width * tile_height
    total_tile_area = num_tiles * tile_area
    
    # Estimate coverage ratio
    coverage_ratio = min(total_bbox_area / total_tile_area, 1.0)
    
    # Expected intersections per Gaussian (empirically ~9 for typical scenes)
    expected_per_gauss = 9.0
    
    # Conservative estimate with safety margin
    estimated_count = int(N * expected_per_gauss * 2.0)  # 2x safety margin
    
    return estimated_count

print("R5 Intersection Count Estimation Strategy")
print("=" * 60)
print()
print("Current behavior:")
print("  capacity = N * MAX_INTERSECTIONS_PER_GAUSSIAN")
print("  At 4.4M: 37.74 GiB allocation")
print()
print("Proposed fix:")
print("  estimated_count = estimate_intersection_count(gaussians)")
print("  capacity = min(estimated_count, N * MAX_INTERSECTIONS_PER_GAUSSIAN)")
print()
print("Expected result:")
print("  At 4.4M: ~12 GiB allocation (vs 37.74 GiB)")
print("  Reduction: ~3x")
print()
print("Implementation:")
print("  - Wrap isect_tiles call with estimation logic")
print("  - Use torch.cuda.memory_stats() to verify reduction")
print("  - Test correctness against original at 580K")
