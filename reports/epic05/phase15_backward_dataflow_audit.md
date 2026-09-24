# Phase 15 — Backward Dataflow Audit

> Detailed analysis of gradient flow through the gsplat differentiable rasterizer.

---

## 1. Backward Call Chain

```
render_colors, render_alphas = rasterize_to_pixels(...)
loss = loss_fn(render_colors, gt_images)
loss.backward()
  │
  ├─── _RasterizeToPixels.backward()
  │     │
  │     ├── v_render_colors [I, H, W, CDIM]  (∂loss/∂render_colors)
  │     ├── v_render_alphas [I, H, W, 1]
  │     │
  │     └── rasterize_to_pixels_3dgs_bwd_kernel
  │           Thread blocks: I × tile_height × tile_width
  │           Threads/block: tile_size × tile_size
  │           
  │           Re-reads from global memory:
  │             means2d   [nnz, 2]    — projected means
  │             conics    [nnz, 3]    — inverse covariances
  │             colors    [nnz, CDIM] — colors (SH-evaluated)
  │             opacities [nnz]       — opacities
  │             tile_offsets [I, th, tw]
  │             flatten_ids [n_isects]
  │             render_alphas [I, H, W]
  │             last_ids   [I, H, W]
  │             v_render_colors [I, H, W, CDIM]
  │             v_render_alphas [I, H, W]
  │
  │           Approximately 10 tensor reads from global memory
  │           ALL of these were also read by the forward kernel
  │
  │           Writes (via gpuAtomicAdd):
  │             v_means2d  [nnz, 2]
  │             v_conics   [nnz, 3]
  │             v_colors   [nnz, CDIM]
  │             v_opacities [nnz]
  │
  │   Returns:
  │     v_means2d, v_conics, v_colors, v_opacities
  │
  ├─── (if SH degree was used) _SphericalHarmonics.backward()
  │     └── sh_coeffs_to_color_fast_vjp
  │         Reads: dirs, coeffs, v_colors (from rasterize_to_pixels backward)
  │         Output: v_coeffs, optionally v_dirs
  │
  └─── _FullyFusedProjectionPacked.backward()
        └── projection_ewa_3dgs_packed_bwd_kernel
            Reads: means, quats, scales, viewmats, Ks,
                   radii (from forward), conics (from forward),
                   v_means2d, v_depths, v_conics
            Output: v_means, v_quats, v_scales, v_viewmats
```

---

## 2. Backward Kernel: rasterize_to_pixels_3dgs_bwd_kernel

### 2.1 Traversal Pattern

The backward kernel traverses gaussians **back to front** — opposite to the forward kernel.

This means the kernel accesses `flatten_ids[n_isects]` in **descending order**, which is:
- NOT a sequential read of flatten_ids (it reads index n_isects-1, n_isects-2, ...)
- But flatten_ids is accessed via computed indices (batch_start + offset), so it IS sequential in decreasing order

### 2.2 Shared Memory Usage

Same pattern as forward: each thread batch-loads gaussians into shared memory.

Shared memory layout per block (`tile_size × tile_size` threads):
```
id_batch[block_size]         — int32: gaussian flatten index
xy_opacity_batch[block_size] — vec3: {x, y, opacity}
conic_batch[block_size]      — vec3: {conic[0], conic[1], conic[2]}
rgbs_batch[block_size * CDIM] — float[CDIM]: colors
```

For CDIM=3, tile_size=16: shared memory = 256 × (4 + 12 + 12 + 12) = 256 × 40 = 10,240 bytes.

Note: The backward kernel's shared memory is **larger** than the forward kernel (which lacks `rgbs_batch`).

### 2.3 gpuAtomicAdd Contention

The backward kernel accumulates gradients via `gpuAtomicAdd` (lines 256-274):

```cuda
if (warp.thread_rank() == 0) {
    gpuAtomicAdd(v_rgb_ptr + k, v_rgb_local[k]);      // per color channel
    gpuAtomicAdd(v_conic_ptr + i, v_conic_local[i]);   // 3 conic values
    gpuAtomicAdd(v_xy_ptr + j, v_xy_local[j]);         // 2 mean values
    gpuAtomicAdd(v_opacities + g, v_opacity_local);    // scalar opacity
}
```

Each warp reduces first, then **one thread per warp** writes. This reduces contention by 32× (from per-thread to per-warp writes).

Contention risk: A large Gaussian (e.g., radius=200px) may be visible in many tiles. If ~10 tiles each have 32 warps writing to the same Gaussian's gradient, that's still 10 simultaneous atomicAdd calls to the same address.

**Observation**: The atomicAdd is on float32 values — this is correct but does not merge writes from overlapping tiles covering the same Gaussian. In a dense tile scenario, a Gaussian could receive atomicAdd contributions from many pixel-warps.

### 2.4 Last ID Optimization

The `last_ids` tensor tracks the last contributing intersection for each pixel. This allows the backward kernel to skip gaussians before `last_ids[pixel]` (they couldn't contribute to that pixel's alpha due to transmittance cutoff). This is an **existing optimization** that eliminates work for terminated pixels.

The warp-level `cg::greater<int>()` reduce on `bin_final` (line 127) extracts the maximum `last_ids` for all threads in a warp, enabling the warp-level skip at line 156:
```cuda
for (uint32_t t = max(0, batch_end - warp_bin_final); t < batch_size; ++t)
```

### 2.5 Redundant Global Memory Reads

The backward kernel re-reads ALL forward input tensors from global memory:
- `means2d`, `conics`, `colors`, `opacities` — these were written by projection/SH in the forward pass
- `tile_offsets`, `flatten_ids` — computed during forward sort/offset
- `render_alphas`, `last_ids` — forward rasterization outputs

**Total read traffic in backward ≈ all forward output tensors read again.**

---

## 3. Backward: Projection VJP

### 3.1 Kernel Behavior

`projection_ewa_3dgs_packed_bwd_kernel` (or fused_bwd for dense mode):

- Parallelizes over valid (radius > 0) gaussians
- **Recomputes** geometry: reads `quats`, `scales`, `viewmats`, `Ks` — recalculates covar matrices and projection
- Applies VJP through: perspective projection, covariance transformation, quat→covar conversion
- No intermediate values from forward projection are available — everything is recomputed

### 3.2 Recompute vs Store Trade-off

The projection backward kernel **recomputes geometry** rather than storing intermediate 3D results:

| Item | Forward computes | Backward needs | Backward action |
|------|-----------------|----------------|-----------------|
| `covar` (3D) | yes, internally | needed for VJP | **recomputed** from quats/scales |
| `mean_c` (camera space) | yes, internally | needed for VJP | **recomputed** from means/viewmats |
| `covar_c` (camera space) | yes, internally | needed for VJP | **recomputed** from covar/R |
| `covar2d` (projected) | yes, internally | needed for VJP | **recomputed** from covar_c |
| `radii` | yes → stored | needed | **stored** (from forward output) |
| `conics` | yes → stored | needed | **stored** (from forward output) |
| `means2d` | yes → stored | needed | **stored** (from forward output) |
| `depths` | yes → stored | needed | **stored** (from forward output) |

The storage cost for intermediate 3D matrices would be ≈B×C×N×(9+3+9+6) = ~27 × B×C×N floats ≫ the 8 floats per gaussian currently stored (means2d[2]+conics[3]+radii[2]+depth[1]).

**Current design is memory-optimal for projection backward** — recompute is the right trade-off for 3D matrices.

---

## 4. Memory Traffic Summary

| Stage | Read (GB) | Write (GB) | AtomicWrites | Notes |
|-------|-----------|------------|--------------|-------|
| Projection fwd | ~36B×B×C×N | ~8B×B×C×N | No | Geometry computation |
| Intersect 1st pass | ~5B×nnz | ~4B×nnz | No | Counting only |
| Intersect 2nd pass | ~7B×nnz | 12×n_isects B | No | Main data expansion |
| Radix sort | 12×2×n_isects B | 12×2×n_isects B | No | Largest memory traffic |
| Offset encode | 8×n_isects B | 4×I×th×tw B | No | Re-scan of sorted data |
| Rasterize fwd | big (all params) | ~4×I×H×W×(CDIM+1) B | No | Alpha compositing |
| **Rasterize bwd** | **big (all fwd params)** | ~4×nnz×(2+3+CDIM+1) B | **Yes** | **Re-reads everything + atomic accumulation** |
| Projection bwd | ~40B×B×C×N | ~10B×B×N | No | Geometry recompute |

---

## 5. Key Backward Observations

### O1: The backward rasterization kernel re-reads the full intersection structure

The backward kernel accesses `flatten_ids[n_isects]` and `tile_offsets[I×th×tw]` — the same sorted intersection data that the forward sort produced. This means the sorted intersection data **must be preserved** from forward to backward.

**Impact**: No gradient computation can begin until the forward sort + offset encode are complete. All n_isects data must live in GPU memory through the backward pass.

### O2: Forward tensors are live through backward

`means2d[nnz,2]`, `conics[nnz,3]`, `colors[nnz,CDIM]`, `opacities[nnz]` are needed by both forward and backward. They cannot be freed until backward completes.

### O3: The `render_alphas` and `last_ids` tensors are specific to backward

These are not needed after forward output but are saved via `ctx.save_for_backward()` — adding ~4×I×H×W bytes of persistent memory.

### O4: No gradient for isect_offsets or flatten_ids

The `intersect_tiles` and `intersect_offset_encode` are `@torch.no_grad()` — they produce integer tensors that are not in the autograd graph. This is correct as tile assignment and sorting are non-differentiable operations.

### O5: AtomicAdd for gradient accumulation is the only contention point

In the entire backward pass, `rasterize_to_pixels_3dgs_bwd_kernel` is the **only kernel using atomic operations**. Every other backward kernel has 1:1 thread-to-gaussian mapping.

### O6: Projection backward recomputes geometry

The projection backward kernel reads quats/scales/viewmats/Ks again and recomputes the full geometry chain (covar → cam space covar → projected 2D → VJP). No intermediate 3D matrices are stored from forward.
