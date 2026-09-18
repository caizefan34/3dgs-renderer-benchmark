# R4 CUDA Insertion Audit

**Date:** 2025-09-18
**Author:** Candidate C R4
**Pinned commit:** `8c2d7b0` (R3.1 final)
**gsplat version:** 1.5.3+pt24cu124

---

## 1. Actual Backward Kernel

The gsplat 1.5.3 backward rasterization kernel is:
`rasterize_to_pixels_bwd_kernel` in `rasterize_to_pixels_bwd.cu`.

### Kernel launch configuration
- **Grid:** `(C, tile_height, tile_width)` — one block per (camera, tile)
- **Block:** `(tile_size, tile_size, 1)` — typically `(16, 16, 1)` = 256 threads
- **Shared memory:** `block_size * (sizeof(int32_t) + 2*sizeof(vec3<float>) + CDIM*sizeof(float))`

### Per-thread workload
Each thread processes one pixel within the tile. For each batch of Gaussians
(in groups of `block_size`), the thread:
1. Loads one Gaussian's data (means2d, conic, opacity, color) into shared memory
2. Iterates through all Gaussians in the batch (back-to-front)
3. For each valid Gaussian:
   - Computes alpha = opacity * exp(-sigma)
   - Updates transmittance T and color buffer
   - Computes gradients: v_rgb, v_conic, v_xy, v_opacity
   - Warp-reduces gradients (warpSum)
   - Thread 0 of warp does atomicAdd to global gradient tensors

---

## 2. Certificate Insertion Point

### Level: Tile-Gaussian intersection

R3.1 proved certificates at the **(tile, Gaussian) pair** level. In the CUDA
kernel, this corresponds to each iteration of the inner loop variable `t`
which iterates over Gaussians within a tile batch.

### Insertion location
```
for (uint32_t t = ...; t < batch_size; ++t) {
    // ... alpha computation ...
    
    if (valid) {
        // >>> CERTIFICATE SKIP CHECK HERE <<<
        bool do_skip = skip_mask[isect_idx];
        
        if (!do_skip) {
            // Full gradient computation (v_rgb, v_conic, v_xy, v_opacity)
            // ... existing code ...
        }
        
        // Buffer update always executes (for T/buffer correctness)
        buffer[k] += rgbs_batch[t * COLOR_DIM + k] * fac;
    }
    
    // Warp reduction + atomicAdd (zero for skipped, no-op)
    // ... existing code ...
}
```

### Data required by certificate
The skip_mask is precomputed on the Python side and passed as a boolean tensor
of shape `[n_isects]`. It requires:
- `means2d`, `conics`, `colors`, `opacities` (from forward pass)
- `tile_offsets`, `flatten_ids` (from intersection sorting)
- `render_alphas` (from forward output)
- Budget epsilon (default 5%)

### Additional loads
- One extra global memory read per intersection: `skip_mask[isect_idx]`
- This is a single bool read, coalesced across the warp (all threads in the
  warp read the same address since `isect_idx = batch_end - t` is uniform)

---

## 3. R3.1 Weighted-Work → CUDA Workload Mapping

### R3.1 definition
```
W_it = number of pixel lanes in tile t where Gaussian i is active (alpha >= 1/255)
JOINT_SKIP_WEIGHTED_WORK_FRACTION = sum(W_it for skipped) / sum(all W_it)
```

### CUDA mapping
In the CUDA kernel, "pixel lanes where derivative work executes" corresponds
to the set of valid threads (pixels) that compute gradients for Gaussian `t`.
This is exactly:
- Threads where `inside == true` (pixel is within image bounds)
- AND `batch_end - t <= bin_final` (Gaussian contributes to this pixel)
- AND `sigma >= 0 && alpha >= 1/255` (Gaussian is visible)

The `warp.any(valid)` check ensures that if no thread in the warp is valid,
the entire loop iteration is skipped. When a Gaussian is certificate-skipped:
- The **gradient computation** (lines 196-241 in original) is skipped
- The **atomicAdd operations** (lines 250-274) still execute but with zero values
- The **alpha/T/buffer update** still executes (for correctness)

### Actual work avoided per skipped intersection
- 3 FMA for v_rgb (COLOR_DIM=3)
- 3 FMA + 3 mul for v_alpha
- 6 FMA for v_conic
- 4 FMA for v_xy
- 1 mul for v_opacity
- 5 warpSum reductions (each ~5 instructions for 32-thread warp)
- 9 atomicAdd operations (3 v_rgb + 3 v_conic + 2 v_xy + 1 v_opacity)
- **Total: ~30 FMA + 5 reductions + 9 atomicAdds per pixel per skipped Gaussian**

### Atomic operations avoided
For each skipped (tile, Gaussian) pair, per pixel lane:
- 3 atomicAdd to v_colors
- 3 atomicAdd to v_conics
- 2 atomicAdd to v_means2d
- 1 atomicAdd to v_opacities
= **9 atomicAdd operations avoided per pixel lane**

With ~62.5% weighted-work removal at 5% budget, and ~8 million tile-Gaussian
interactions per camera at 30K, this translates to approximately:
- ~5 million intersections × ~100 pixel lanes average × 9 atomicAdds
= ~4.5 billion atomicAdd operations avoided per backward pass

---

## 4. Branches Introduced

### Single branch per intersection
```cpp
bool do_skip = skip_mask[isect_idx];  // uniform across warp
if (!do_skip) { /* gradient computation */ }
```

### Divergence analysis
- `do_skip` is **warp-uniform** (all 32 threads in the warp read the same
  `skip_mask[isect_idx]` because `isect_idx = batch_end - t` depends only on
  the loop variable `t`, not on the thread index)
- Therefore there is **no warp divergence** in the skip branch
- The branch is a simple conditional that either all threads take or none take

### Buffer update
The buffer update (`buffer[k] += ...`) always executes regardless of skip.
This is necessary because subsequent Gaussians' gradient computation depends
on the accumulated buffer value. Skipping the buffer update would make
subsequent gradients incorrect.

---

## 5. Correctness Preservation

### What is preserved
1. **Forward pass**: completely unchanged (same kernel, same outputs)
2. **Transmittance T**: correctly updated for skipped Gaussians
3. **Color buffer**: correctly accumulated for skipped Gaussians
4. **Subsequent gradients**: computed with correct T and buffer values

### What is skipped
1. Gradient computation for the skipped Gaussian's color, conic, mean2d, opacity
2. The corresponding warpSum reductions (execute with zero values — no-op)
3. The corresponding atomicAdd operations (execute with zero values — no-op)

### What is NOT skipped
1. Alpha computation (needed for T update)
2. Buffer accumulation (needed for subsequent gradients)
3. The `valid` check and `warp.any(valid)` early exit

### Numerical impact
The skipped gradients are set to exactly zero (not approximately zero).
This means the gradient tensor for a skipped Gaussian will have:
- v_colors[g] = 0 (contribution from this pixel lane)
- v_conics[g] = 0
- v_means2d[g] = 0
- v_opacities[g] = 0

The R3.1 certificate guarantees that the sum of these skipped contributions
across all pixel lanes is bounded by epsilon * total_bound for each family.
This is the certified error budget — the training will converge to a solution
within this budget of the exact gradient solution.
