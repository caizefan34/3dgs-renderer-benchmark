# R6-0 — Backward Code-Path Audit

## 1. Frozen baseline (B2)

| Item | Value |
|------|-------|
| Repo | `~/3dgs-renderer-benchmark` on mx (36.140.146.31:26372) |
| Commit | `32ab80e773f74f4d8e40ff4e338c86b29c7957f5` |
| Branch | master (dirty: colmap_reader.py, config.py, scripts/phase-r0.1) |
| trainer.py sha256 | `ca4f9744daa0edb14d94d43012f6d509b4db3b87d960a081bade3e49f1f452f3` |
| config.py sha256 | `b440df61f49b00af5a29cc266147b7fdeb706c0386bb6ede798c1fb10998dce4` |
| gaussian_model.py sha256 | `b8cb0e251bfb0822291cf788911c2ca38d5977fea591f54f1c740052e8fc5f5f` |
| gsplat version | 1.5.3 (`~/.local/lib/python3.10/site-packages/gsplat/`) |
| PyTorch | 2.4.1+cu124 |
| CUDA runtime | 12.4 |
| Python | 3.10.19 (conda env `anysplat`) |
| GPU | 8× A100-PCIE-40GB, driver 595.71.05 |
| ncu | `/usr/bin/ncu` v2021.3.1.0 |
| nsys | NOT installed |
| Training config | ReferenceV1Config: seed=0, 30K iters, SH degree 3, 1080p, tile_size=16, packed=False, absgrad=True, densify_grad_threshold=0.0008, opacity_reset=3000, densify 500–15000 every 100, loss=0.2×L1+0.8×DSSIM |
| Optimizer | torch.optim.Adam, 5 param groups, eps=1e-15, zero_grad(set_to_none=True) |
| AMP | none (full fp32) |
| Exact rasterizer path | gsplat.cuda._wrapper._RasterizeToPixels → rasterize_to_pixels_3dgs_bwd (Rasterization.cpp:118) |

## 2. Backward call chain (Python → CUDA)

```
trainer.py: run_training loop (iter 1..30000)
│
├─ render_with_meta(model, cam, sh_degree)
│   └─ gsplat.rasterization(means=xyz, quats=rotations, scales=scaling,
│       opacities=opacity, colors=shs, viewmats, Ks, width, height,
│       tile_size=16, packed=False, sh_degree=active, absgrad=True)
│       │
│       ├─ [forward] fully_fused_projection(means, covars=None, quats, scales, ...)
│       │   → radii, means2d, depths, conics, compensations
│       │   (fused: quats+scales → covars → EWA projection in one kernel)
│       │
│       ├─ [forward] spherical_harmonics(sh_degree, dirs, shs, masks)
│       │   → colors [N, 3]  (post-SH RGB, clamped ≥0)
│       │
│       └─ [forward] _RasterizeToPixels.apply(means2d, conics, colors,
│               opacities, backgrounds, masks, width, height, tile_size,
│               isect_offsets, flatten_ids, absgrad)
│           → render_colors [H,W,3], render_alphas [H,W,1]
│
├─ loss = 0.2*L1 + 0.8*DSSIM(image, gt)
│
├─ loss.backward()  ← THE BACKWARD ENTRY POINT
│   │
│   ├─ [bwd] _RasterizeToPixels.backward(v_render_colors, v_render_alphas)
│   │   └─ rasterize_to_pixels_3dgs_bwd (Rasterization.cpp:118)
│   │       ├─ ALLOCATES + ZEROS: v_means2d, v_conics, v_colors, v_opacities, v_means2d_abs
│   │       └─ KERNEL: rasterize_to_pixels_3dgs_bwd_kernel (ATOMIC gpuAtomicAdd)
│   │       → returns v_means2d_abs, v_means2d, v_conics, v_colors, v_opacities
│   │
│   ├─ [bwd] spherical_harmonics.backward(v_colors)
│   │   └─ ALLOCATES + ZEROS: v_coefficients [N, K_active, 3]
│   │   └─ KERNEL: elementwise (one thread per visible Gaussian, NO atomics)
│   │       → v_coefficients accumulates into _shs.grad
│   │
│   ├─ [bwd] _FullyFusedProjection.backward(v_means2d, v_conics, ...)
│   │   └─ projection_ewa_3dgs_fused_bwd (Projection.cpp:192)
│   │       ├─ ALLOCATES + ZEROS: v_means [N,3], v_quats [N,4], v_scales [N,3]
│   │       │   (v_covars NOT allocated — covars=None input)
│   │       └─ KERNEL: elementwise (one thread per visible Gaussian, NO atomics)
│   │           → v_means → _xyz.grad, v_quats → _rotation.grad, v_scales → _scaling.grad
│   │
│   └─ [bwd] sigmoid.backward(v_opacities)
│       └─ elementwise autograd → _opacity.grad
│
├─ model.add_densification_stats(means2d, visibility_filter)
│   └─ reads means2d.absgrad (the absgrad path)
│
├─ [densification/prune every 100 iters if 500<iter<15000]
│   └─ reads xyz_gradient_accum / denom
│
├─ model.optimizer.step()  ← reads .grad on all 5 parameters
│
└─ model.optimizer.zero_grad(set_to_none=True)  ← frees .grad tensors
```

## 3. The seven mandatory audit questions

### Q1: Where are gradient buffers allocated?

Three C++ host launchers allocate gradient output tensors:

| Location | Buffers | Shape | Method |
|----------|---------|-------|--------|
| Rasterization.cpp:162-168 | v_means2d | [N, 2] | `at::zeros_like(means2d)` |
| | v_conics | [N, 3] | `at::zeros_like(conics)` |
| | v_colors | [N, 3] | `at::zeros_like(colors)` (CDIM=3, post-SH) |
| | v_opacities | [N] | `at::zeros_like(opacities)` |
| | v_means2d_abs | [N, 2] | `at::zeros_like(means2d)` (absgrad=True, always) |
| Projection.cpp:239-249 | v_means | [N, 3] | `at::zeros_like(means)` |
| | v_quats | [N, 4] | `at::zeros_like(quats)` |
| | v_scales | [N, 3] | `at::zeros_like(scales)` |
| | v_covars | — | NOT allocated (covars=None input) |
| SphericalHarmonics.cpp:59 | v_coefficients | [N, K_active, 3] | `at::zeros_like(coeffs)` |

PyTorch autograd also allocates `.grad` tensors for the 5 `nn.Parameter`s
(xyz, shs, scaling, rotation, opacity) during backward accumulation.
With `zero_grad(set_to_none=True)`, these are freed after `optimizer.step()`.

### Q2: Where are gradient buffers zero-initialized?

**All** gradient buffers listed above use `at::zeros_like`, which dispatches to
`cudaMemsetAsync(ptr, 0, nbytes)`. Zero-init is explicit and unconditional.

Key observation: zero-init scales with **TOTAL N** (all Gaussians), not with
the number of Gaussians actually touched (visible/intersecting) in the current
frame. This is the R6-B target.

### Q3: Which gradients use atomicAdd?

**Only** the rasterizer backward kernel uses atomic operations:
`rasterize_to_pixels_3dgs_bwd_kernel` (RasterizeToPixels3DGSBwd.cu:256-274).

All other backward kernels (SH, projection fused, sigmoid) are **elementwise**
(one thread per Gaussian, direct write, no contention).

### Q4: Theoretical atomic count per contribution

The rasterizer backward kernel organizes as one thread-block per tile
(16×16 = 256 threads = 8 warps). Within each warp, all 32 lanes process the
SAME Gaussian `t` per loop iteration (batch-loaded into shared memory).

The kernel **already performs warp-level aggregation**:
1. Each lane computes its per-pixel gradient contribution to Gaussian `t`.
2. `warpSum<CDIM>(v_rgb_local, warp)` reduces across the 32 lanes.
3. Only `warp.thread_rank() == 0` issues `gpuAtomicAdd`.

Atomic ops per (warp, Gaussian):
| Buffer | gpuAtomicAdd calls |
|--------|--------------------|
| v_colors (CDIM=3) | 3 |
| v_conics (3) | 3 |
| v_means2d (2) | 2 |
| v_means2d_abs (2) | 2 |
| v_opacities (1) | 1 |
| **Total** | **11** |

Theoretical total atomics per backward = 11 × (number of (warp, Gaussian) pairs
that contribute). Each contributing Gaussian in a tile generates 1 atomic per
warp that has at least one active pixel for that Gaussian. With 8 warps per
tile, a Gaussian covering the entire tile generates up to 8 atomic ops (one
per warp). A Gaussian covering 1 pixel generates 1 atomic op.

**Implication for R6-A**: The within-warp aggregation is already done.
Remaining atomic-reduction potential is **block-level** (across the 8 warps
in a tile writing to the same Gaussian) and **cross-tile** (same Gaussian
appearing in multiple tiles).

### Q5: What parameter gradients does the backward output?

The backward produces `.grad` on 5 `nn.Parameter`s:

| Parameter | Shape | Gradient source | Via |
|-----------|-------|-----------------|-----|
| `_xyz` | [N, 3] | v_means [N, 3] | projection fused backward |
| `_shs` | [N, K_active, 3] | v_coefficients [N, K_active, 3] | SH backward |
| `_scaling` | [N, 3] | v_scales [N, 3] | projection fused backward |
| `_rotation` | [N, 4] | v_quats [N, 4] | projection fused backward |
| `_opacity` | [N] | v_opacities [N] | sigmoid backward |

Note: `_shs.grad` is [N, K_active, 3] where K_active = (active_sh_degree+1)².
The full tensor is [N, 16, 3] but only the active coefficients get gradients.
The SH forward slices `get_features = _shs[:, :n_active, :].contiguous()`,
so the backward gradient is only for the active slice.

### Q6: Which gradient buffers does the optimizer read?

`torch.optim.Adam` reads `.grad` on all 5 parameters via its param groups:
```
group "xyz":      [self._xyz]       lr = position_lr_init * spatial_lr_scale (exponential decay)
group "shs":      [self._shs]       lr = feature_lr = 0.0025
group "opacity":  [self._opacity]   lr = opacity_lr = 0.025
group "scaling":  [self._scaling]   lr = scaling_lr = 0.005
group "rotation": [self._rotation]  lr = rotation_lr = 0.001
```

Adam also reads/writes moment states `exp_avg` and `exp_avg_sq` for each
parameter (allocated lazily on first step, migrated across topology changes).

### Q7: Which intermediate buffers have lifetime only across backward→optimizer?

| Buffer | Allocated in | Consumed by | Lifetime |
|--------|-------------|-------------|----------|
| v_means2d [N,2] | rasterizer bwd | projection bwd (as v_means2d input) | backward only |
| v_conics [N,3] | rasterizer bwd | projection bwd (as v_conics input) | backward only |
| v_colors [N,3] | rasterizer bwd | SH bwd (as v_colors input) | backward only |
| v_opacities [N] | rasterizer bwd | sigmoid bwd | backward only |
| v_means2d_abs [N,2] | rasterizer bwd | add_densification_stats (reads means2d.absgrad) | backward→densification |
| v_means [N,3] | projection bwd | accumulates into _xyz.grad | backward only |
| v_quats [N,4] | projection bwd | accumulates into _rotation.grad | backward only |
| v_scales [N,3] | projection bwd | accumulates into _scaling.grad | backward only |
| v_coefficients [N,K,3] | SH bwd | accumulates into _shs.grad | backward only |
| _xyz.grad [N,3] | autograd | optimizer.step() | backward→optimizer (freed by zero_grad) |
| _shs.grad [N,K,3] | autograd | optimizer.step() | backward→optimizer (freed by zero_grad) |
| _scaling.grad [N,3] | autograd | optimizer.step() | backward→optimizer (freed by zero_grad) |
| _rotation.grad [N,4] | autograd | optimizer.step() | backward→optimizer (freed by zero_grad) |
| _opacity.grad [N] | autograd | optimizer.step() | backward→optimizer (freed by zero_grad) |

The rasterizer's 5 gradient buffers (v_means2d, v_conics, v_colors, v_opacities,
v_means2d_abs) are intermediate — they exist only during backward, consumed by
the VJP chain. This is the R6-C fusion target: if the rasterizer could directly
accumulate into the parameter .grad tensors (bypassing the intermediate
v_means2d → v_means → _xyz.grad chain), the intermediate global writes/reads
could be eliminated.

## 4. Total zero-init budget per backward call

| Component | Floats per Gaussian | Bytes per Gaussian | At SH degree 0 | At SH degree 3 |
|-----------|--------------------:|-------------------:|----------------:|---------------:|
| Rasterizer (5 buffers) | 11 | 44 | 44N | 44N |
| Projection fused (3 buffers) | 10 | 40 | 40N | 40N |
| SH backward (1 buffer) | 3×(d+1)² | 12×(d+1)² | 12N | 192N |
| **Total** | | | **96N bytes** | **276N bytes** |

For garden late training (N≈3M, SH degree 3):
- Total zero-init ≈ 276 × 3M = 828 MB per backward
- At A100 memory bandwidth (1.55 TB/s), cudaMemset ≈ 0.53 ms

## 5. Implications for R6 candidates

### R6-B (zero-init / fixed-cost oracle)
- Target: 3 launchers zero-init **24N–69N floats** (96N–276N bytes) per backward
- Zero-init scales with TOTAL N, not touched count
- If N is large and touched fraction is small, zero-init is wasteful
- **Must measure actual T_zero vs T_bwd and T_iter**

### R6-A (atomic accumulation)
- gsplat 1.5.3 ALREADY does warp-level aggregation (warpSum + warp-leader atomicAdd)
- Remaining potential: block-level (8 warps/tile → same Gaussian) and cross-tile
- The duplicate structure must be measured from the intersection list (isect_offsets + flatten_ids)
- **The "same warp, same Gaussian" assumption from the protocol's §7 is already handled —
  the real question is cross-warp and cross-tile duplicates**

### R6-C (backward-optimizer fusion)
- 9 intermediate gradient buffers (rasterizer + projection + SH) exist only during backward
- They are consumed by the VJP chain, then freed
- The parameter .grad tensors persist until optimizer.step()
- Fusion could bypass intermediate buffers but must preserve the full VJP math
- **Must measure T_optimizer and grad-buffer traffic**

## 6. Source provenance

All source files read from the installed gsplat 1.5.3 package on mx:
- `~/.local/lib/python3.10/site-packages/gsplat/cuda/csrc/Rasterization.cpp` (host launcher)
- `~/.local/lib/python3.10/site-packages/gsplat/cuda/csrc/RasterizeToPixels3DGSBwd.cu` (kernel)
- `~/.local/lib/python3.10/site-packages/gsplat/cuda/csrc/Projection.cpp` (projection binding)
- `~/.local/lib/python3.10/site-packages/gsplat/cuda/csrc/SphericalHarmonics.cpp` (SH binding)
- `~/.local/lib/python3.10/site-packages/gsplat/cuda/_wrapper.py` (Python autograd Functions)
- `~/.local/lib/python3.10/site-packages/gsplat/rendering.py` (rasterization() orchestrator)

Baseline trainer/model read from repo at commit 32ab80e on mx:
- `baseline/reference_v1/trainer.py`
- `baseline/reference_v1/gaussian_model.py`
- `baseline/reference_v1/config.py`
- `experiments/r4/r4_train_wrapper.py`

Local copies stored in `experiments/r6/_mxsrc/`.
