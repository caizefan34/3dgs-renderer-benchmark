"""
R2.1 Patch Script — Adds three independent per-branch masks to rasterize_to_pixels_3dgs_bwd.

Modifies:
  1. RasterizeToPixels3DGSBwd.cu — kernel + launch function
  2. Rasterization.cpp — C++ dispatch
  3. _wrapper.py — Python autograd Function + rasterize_to_pixels()
  4. rendering.py — rasterization() top-level

Saves backups as *.orig_r21.
"""
import os, sys, re

GSPLAT_DIR = "/tmp/gsplat_baseline/gsplat-1.5.3/gsplat"
CUDA_CSRC = f"{GSPLAT_DIR}/cuda/csrc"
WRAPPER = f"{GSPLAT_DIR}/cuda/_wrapper.py"
RENDERING = f"{GSPLAT_DIR}/rendering.py"


def backup(filepath):
    bak = filepath + ".orig_r21"
    if not os.path.exists(bak):
        import shutil
        shutil.copy2(filepath, bak)
        print(f"  Backed up to {bak}")


def patch_file(filepath, replacements, description):
    print(f"\n=== Patching {os.path.basename(filepath)}: {description} ===")
    backup(filepath)
    with open(filepath, 'r') as f:
        content = f.read()
    for old, new, desc in replacements:
        if old not in content:
            print(f"  WARNING: Could not find pattern for: {desc}")
            continue
        content = content.replace(old, new, 1)
        print(f"  Applied: {desc}")
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"  Written {filepath}")


# ============================================================
# 1. Patch RasterizeToPixels3DGSBwd.cu — CUDA kernel
# ============================================================
def patch_cuda_kernel():
    filepath = f"{CUDA_CSRC}/RasterizeToPixels3DGSBwd.cu"
    print(f"\n=== Patching {os.path.basename(filepath)} ===")
    backup(filepath)

    with open(filepath, 'r') as f:
        content = f.read()

    # 1a. Add three mask parameters to kernel signature (after importance_mask)
    old_sig = """    // sparse backward: importance mask
    const uint8_t *__restrict__ importance_mask, // [N] or [nnz], 1=compute grad, 0=skip
    const bool compute_densify_grad,             // B2: if true, compute v_means2d for masked
    const uint32_t image_width,"""

    new_sig = """    // sparse backward: importance mask (legacy single-mask mode)
    const uint8_t *__restrict__ importance_mask, // [N] or [nnz], 1=compute grad, 0=skip
    const bool compute_densify_grad,             // B2: if true, compute v_means2d for masked
    // R2.1: per-branch masks (when all three are non-null, per-branch mode is used)
    const uint8_t *__restrict__ r2_geo_mask,     // [N] or [nnz], 1=compute v_means2d+v_conics, 0=skip
    const uint8_t *__restrict__ r2_app_mask,     // [N] or [nnz], 1=compute v_colors, 0=skip
    const uint8_t *__restrict__ r2_opacity_mask, // [N] or [nnz], 1=compute v_opacities, 0=skip
    const uint32_t image_width,"""

    content = content.replace(old_sig, new_sig, 1)
    print("  Applied: kernel signature — added 3 mask params")

    # 1b. Add three bool shared memory arrays after mask_batch
    old_shmem = """    bool *mask_batch =
        (bool *)&rgbs_batch[block_size * CDIM]; // [block_size] — importance mask"""

    new_shmem = """    bool *mask_batch =
        (bool *)&rgbs_batch[block_size * CDIM]; // [block_size] — importance mask (legacy)
    bool *app_mask_batch =
        (bool *)&mask_batch[block_size]; // [block_size] — R2.1 app mask
    bool *geo_mask_batch =
        (bool *)&app_mask_batch[block_size]; // [block_size] — R2.1 geo mask
    bool *opacity_mask_batch =
        (bool *)&geo_mask_batch[block_size]; // [block_size] — R2.1 opacity mask"""

    content = content.replace(old_shmem, new_shmem, 1)
    print("  Applied: shared memory — added 3 mask arrays")

    # 1c. Replace mask loading logic
    old_load = """            // Load importance mask: 1=compute gradient, 0=skip
            if (importance_mask != nullptr) {
                int32_t mask_idx = packed ? g : (g - (int32_t)(image_id * N));
                mask_batch[tr] = importance_mask[mask_idx] != 0;
            } else {
                mask_batch[tr] = true; // no mask = compute all
            }
        } else {
            mask_batch[tr] = true; // default for out-of-range threads
        }"""

    new_load = """            // Load importance mask: 1=compute gradient, 0=skip
            int32_t mask_idx = packed ? g : (g - (int32_t)(image_id * N));
            if (importance_mask != nullptr) {
                mask_batch[tr] = importance_mask[mask_idx] != 0;
            } else {
                mask_batch[tr] = true; // no mask = compute all
            }
            // R2.1: load per-branch masks
            if (r2_app_mask != nullptr) {
                app_mask_batch[tr] = r2_app_mask[mask_idx] != 0;
            } else {
                app_mask_batch[tr] = true;
            }
            if (r2_geo_mask != nullptr) {
                geo_mask_batch[tr] = r2_geo_mask[mask_idx] != 0;
            } else {
                geo_mask_batch[tr] = true;
            }
            if (r2_opacity_mask != nullptr) {
                opacity_mask_batch[tr] = r2_opacity_mask[mask_idx] != 0;
            } else {
                opacity_mask_batch[tr] = true;
            }
        } else {
            mask_batch[tr] = true; // default for out-of-range threads
            app_mask_batch[tr] = true;
            geo_mask_batch[tr] = true;
            opacity_mask_batch[tr] = true;
        }"""

    content = content.replace(old_load, new_load, 1)
    print("  Applied: mask loading — per-branch")

    # 1d. Replace the gradient computation block
    # This is the big one — replace the entire do_full_grad / do_densify_only logic
    old_grad = """            // Determine gradient computation mode for this Gaussian.
            // mask_batch[t] is uniform across all threads in the warp (same Gaussian).
            bool g_masked = !mask_batch[t];  // true if this Gaussian is skipped
            bool do_full_grad = valid && !g_masked;
            bool do_densify_only = valid && g_masked && compute_densify_grad;
            // initialize everything to 0, only set if the lane is valid
            if (valid) {
                // compute the current T for this gaussian — MUST ALWAYS DO
                float ra = 1.0f / (1.0f - alpha);
                T *= ra;
                const float fac = alpha * T;

                // Buffer update — MUST ALWAYS DO (for T/buffer correctness)
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    buffer[k] += rgbs_batch[t * CDIM + k] * fac;
                }

                if (do_full_grad) {
                    // === Full gradient computation (selected Gaussians) ===
#pragma unroll
                    for (uint32_t k = 0; k < CDIM; ++k) {
                        v_rgb_local[k] = fac * v_render_c[k];
                    }
                    float v_alpha = 0.f;
#pragma unroll
                    for (uint32_t k = 0; k < CDIM; ++k) {
                        v_alpha += (rgbs_batch[t * CDIM + k] * T - buffer[k] * ra) *
                                   v_render_c[k];
                    }
                    v_alpha += T_final * ra * v_render_a;
                    if (backgrounds != nullptr) {
                        float accum = 0.f;
#pragma unroll
                        for (uint32_t k = 0; k < CDIM; ++k) {
                            accum += backgrounds[k] * v_render_c[k];
                        }
                        v_alpha += -T_final * ra * accum;
                    }
                    if (opac * vis <= 0.999f) {
                        const float v_sigma = -opac * vis * v_alpha;
                        v_conic_local = {
                            0.5f * v_sigma * delta.x * delta.x,
                            v_sigma * delta.x * delta.y,
                            0.5f * v_sigma * delta.y * delta.y
                        };
                        v_xy_local = {
                            v_sigma * (conic.x * delta.x + conic.y * delta.y),
                            v_sigma * (conic.y * delta.x + conic.z * delta.y)
                        };
                        if (v_means2d_abs != nullptr) {
                            v_xy_abs_local = {abs(v_xy_local.x), abs(v_xy_local.y)};
                        }
                        v_opacity_local = vis * v_alpha;
                    }
                } else if (do_densify_only) {
                    // === B2: compute only v_xy_local for densification ===
                    // Need v_alpha and v_sigma to get v_xy_local.
                    // Skip v_rgb, v_conic, v_opacity computation.
                    float v_alpha = 0.f;
#pragma unroll
                    for (uint32_t k = 0; k < CDIM; ++k) {
                        v_alpha += (rgbs_batch[t * CDIM + k] * T - buffer[k] * ra) *
                                   v_render_c[k];
                    }
                    v_alpha += T_final * ra * v_render_a;
                    if (backgrounds != nullptr) {
                        float accum = 0.f;
#pragma unroll
                        for (uint32_t k = 0; k < CDIM; ++k) {
                            accum += backgrounds[k] * v_render_c[k];
                        }
                        v_alpha += -T_final * ra * accum;
                    }
                    if (opac * vis <= 0.999f) {
                        const float v_sigma = -opac * vis * v_alpha;
                        v_xy_local = {
                            v_sigma * (conic.x * delta.x + conic.y * delta.y),
                            v_sigma * (conic.y * delta.x + conic.z * delta.y)
                        };
                        if (v_means2d_abs != nullptr) {
                            v_xy_abs_local = {abs(v_xy_local.x), abs(v_xy_local.y)};
                        }
                        // v_rgb_local, v_conic_local, v_opacity_local stay 0
                    }
                }
                // else: B1/B3 mode, all gradient vars stay 0
            }"""

    new_grad = """            // Determine gradient computation mode for this Gaussian.
            // mask_batch[t] is uniform across all threads in the warp (same Gaussian).
            bool g_masked = !mask_batch[t];  // true if this Gaussian is skipped (legacy)
            bool do_full_grad = valid && !g_masked;
            bool do_densify_only = valid && g_masked && compute_densify_grad;
            // R2.1: per-branch masks (uniform across warp)
            bool r2_mode = (r2_geo_mask != nullptr) || (r2_app_mask != nullptr) || (r2_opacity_mask != nullptr);
            bool do_app = valid && (r2_mode ? app_mask_batch[t] : !g_masked);
            bool do_geo = valid && (r2_mode ? geo_mask_batch[t] : (!g_masked || compute_densify_grad));
            bool do_opacity = valid && (r2_mode ? opacity_mask_batch[t] : !g_masked);
            bool do_geo_full = valid && (r2_mode ? geo_mask_batch[t] : !g_masked);
            // initialize everything to 0, only set if the lane is valid
            if (valid) {
                // compute the current T for this gaussian — MUST ALWAYS DO
                float ra = 1.0f / (1.0f - alpha);
                T *= ra;
                const float fac = alpha * T;

                // Buffer update — MUST ALWAYS DO (for T/buffer correctness)
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    buffer[k] += rgbs_batch[t * CDIM + k] * fac;
                }

                // === R2.1: per-branch gradient computation ===
                // v_rgb only needs fac + v_render_c (NOT v_alpha)
                if (do_app) {
#pragma unroll
                    for (uint32_t k = 0; k < CDIM; ++k) {
                        v_rgb_local[k] = fac * v_render_c[k];
                    }
                }

                // v_alpha is needed for opacity, conic, AND xy
                // Only compute if at least one of those branches is active
                bool need_v_alpha = do_opacity || do_geo;
                float v_alpha = 0.f;
                if (need_v_alpha) {
#pragma unroll
                    for (uint32_t k = 0; k < CDIM; ++k) {
                        v_alpha += (rgbs_batch[t * CDIM + k] * T - buffer[k] * ra) *
                                   v_render_c[k];
                    }
                    v_alpha += T_final * ra * v_render_a;
                    if (backgrounds != nullptr) {
                        float accum = 0.f;
#pragma unroll
                        for (uint32_t k = 0; k < CDIM; ++k) {
                            accum += backgrounds[k] * v_render_c[k];
                        }
                        v_alpha += -T_final * ra * accum;
                    }
                }

                // Opacity gradient
                if (do_opacity && opac * vis <= 0.999f) {
                    v_opacity_local = vis * v_alpha;
                }

                // Geometry gradients (v_conic + v_xy)
                if (do_geo_full && opac * vis <= 0.999f) {
                    const float v_sigma = -opac * vis * v_alpha;
                    v_conic_local = {
                        0.5f * v_sigma * delta.x * delta.x,
                        v_sigma * delta.x * delta.y,
                        0.5f * v_sigma * delta.y * delta.y
                    };
                    v_xy_local = {
                        v_sigma * (conic.x * delta.x + conic.y * delta.y),
                        v_sigma * (conic.y * delta.x + conic.z * delta.y)
                    };
                    if (v_means2d_abs != nullptr) {
                        v_xy_abs_local = {abs(v_xy_local.x), abs(v_xy_local.y)};
                    }
                } else if (do_geo && !do_geo_full && opac * vis <= 0.999f) {
                    // B2 legacy: compute only v_xy for densification
                    const float v_sigma = -opac * vis * v_alpha;
                    v_xy_local = {
                        v_sigma * (conic.x * delta.x + conic.y * delta.y),
                        v_sigma * (conic.y * delta.x + conic.z * delta.y)
                    };
                    if (v_means2d_abs != nullptr) {
                        v_xy_abs_local = {abs(v_xy_local.x), abs(v_xy_local.y)};
                    }
                }
            }"""

    content = content.replace(old_grad, new_grad, 1)
    print("  Applied: gradient computation — per-branch")

    # 1e. Replace warp reduction + atomicAdd block
    old_atomic = """            // Warp reduction — uniform branch (mask_batch[t] same for all threads in warp)
            if (!g_masked) {
                // Full warpSum for selected Gaussians
                warpSum<CDIM>(v_rgb_local, warp);
                warpSum(v_conic_local, warp);
                warpSum(v_xy_local, warp);
                if (v_means2d_abs != nullptr) {
                    warpSum(v_xy_abs_local, warp);
                }
                warpSum(v_opacity_local, warp);
            } else if (compute_densify_grad) {
                // B2: only warpSum for v_xy (densification)
                warpSum(v_xy_local, warp);
                if (v_means2d_abs != nullptr) {
                    warpSum(v_xy_abs_local, warp);
                }
            }
            // else: B1/B3 — no warpSum needed (all zeros)
            if (warp.thread_rank() == 0) {
                int32_t g = id_batch[t]; // flatten index in [I * N] or [nnz]
                if (!g_masked) {
                    // Full atomicAdd for selected Gaussians
                    float *v_rgb_ptr = (float *)(v_colors) + CDIM * g;
#pragma unroll
                    for (uint32_t k = 0; k < CDIM; ++k) {
                        gpuAtomicAdd(v_rgb_ptr + k, v_rgb_local[k]);
                    }
                    float *v_conic_ptr = (float *)(v_conics) + 3 * g;
                    gpuAtomicAdd(v_conic_ptr, v_conic_local.x);
                    gpuAtomicAdd(v_conic_ptr + 1, v_conic_local.y);
                    gpuAtomicAdd(v_conic_ptr + 2, v_conic_local.z);
                    float *v_xy_ptr = (float *)(v_means2d) + 2 * g;
                    gpuAtomicAdd(v_xy_ptr, v_xy_local.x);
                    gpuAtomicAdd(v_xy_ptr + 1, v_xy_local.y);
                    if (v_means2d_abs != nullptr) {
                        float *v_xy_abs_ptr = (float *)(v_means2d_abs) + 2 * g;
                        gpuAtomicAdd(v_xy_abs_ptr, v_xy_abs_local.x);
                        gpuAtomicAdd(v_xy_abs_ptr + 1, v_xy_abs_local.y);
                    }
                    gpuAtomicAdd(v_opacities + g, v_opacity_local);
                } else if (compute_densify_grad) {
                    // B2: only v_means2d atomicAdd for densification
                    float *v_xy_ptr = (float *)(v_means2d) + 2 * g;
                    gpuAtomicAdd(v_xy_ptr, v_xy_local.x);
                    gpuAtomicAdd(v_xy_ptr + 1, v_xy_local.y);
                    if (v_means2d_abs != nullptr) {
                        float *v_xy_abs_ptr = (float *)(v_means2d_abs) + 2 * g;
                        gpuAtomicAdd(v_xy_abs_ptr, v_xy_abs_local.x);
                        gpuAtomicAdd(v_xy_abs_ptr + 1, v_xy_abs_local.y);
                    }
                }
                // else: B1/B3 — no atomicAdd
            }"""

    new_atomic = """            // Warp reduction — per-branch conditional (uniform across warp)
            if (r2_mode) {
                // R2.1: per-branch warp reduction + atomicAdd
                if (app_mask_batch[t]) {
                    warpSum<CDIM>(v_rgb_local, warp);
                }
                if (opacity_mask_batch[t]) {
                    warpSum(v_opacity_local, warp);
                }
                if (geo_mask_batch[t]) {
                    warpSum(v_conic_local, warp);
                    warpSum(v_xy_local, warp);
                    if (v_means2d_abs != nullptr) {
                        warpSum(v_xy_abs_local, warp);
                    }
                }
                if (warp.thread_rank() == 0) {
                    int32_t g = id_batch[t];
                    if (app_mask_batch[t]) {
                        float *v_rgb_ptr = (float *)(v_colors) + CDIM * g;
#pragma unroll
                        for (uint32_t k = 0; k < CDIM; ++k) {
                            gpuAtomicAdd(v_rgb_ptr + k, v_rgb_local[k]);
                        }
                    }
                    if (opacity_mask_batch[t]) {
                        gpuAtomicAdd(v_opacities + g, v_opacity_local);
                    }
                    if (geo_mask_batch[t]) {
                        float *v_conic_ptr = (float *)(v_conics) + 3 * g;
                        gpuAtomicAdd(v_conic_ptr, v_conic_local.x);
                        gpuAtomicAdd(v_conic_ptr + 1, v_conic_local.y);
                        gpuAtomicAdd(v_conic_ptr + 2, v_conic_local.z);
                        float *v_xy_ptr = (float *)(v_means2d) + 2 * g;
                        gpuAtomicAdd(v_xy_ptr, v_xy_local.x);
                        gpuAtomicAdd(v_xy_ptr + 1, v_xy_local.y);
                        if (v_means2d_abs != nullptr) {
                            float *v_xy_abs_ptr = (float *)(v_means2d_abs) + 2 * g;
                            gpuAtomicAdd(v_xy_abs_ptr, v_xy_abs_local.x);
                            gpuAtomicAdd(v_xy_abs_ptr + 1, v_xy_abs_local.y);
                        }
                    }
                }
            } else if (!g_masked) {
                // Legacy: Full warpSum for selected Gaussians
                warpSum<CDIM>(v_rgb_local, warp);
                warpSum(v_conic_local, warp);
                warpSum(v_xy_local, warp);
                if (v_means2d_abs != nullptr) {
                    warpSum(v_xy_abs_local, warp);
                }
                warpSum(v_opacity_local, warp);
                if (warp.thread_rank() == 0) {
                    int32_t g = id_batch[t];
                    float *v_rgb_ptr = (float *)(v_colors) + CDIM * g;
#pragma unroll
                    for (uint32_t k = 0; k < CDIM; ++k) {
                        gpuAtomicAdd(v_rgb_ptr + k, v_rgb_local[k]);
                    }
                    float *v_conic_ptr = (float *)(v_conics) + 3 * g;
                    gpuAtomicAdd(v_conic_ptr, v_conic_local.x);
                    gpuAtomicAdd(v_conic_ptr + 1, v_conic_local.y);
                    gpuAtomicAdd(v_conic_ptr + 2, v_conic_local.z);
                    float *v_xy_ptr = (float *)(v_means2d) + 2 * g;
                    gpuAtomicAdd(v_xy_ptr, v_xy_local.x);
                    gpuAtomicAdd(v_xy_ptr + 1, v_xy_local.y);
                    if (v_means2d_abs != nullptr) {
                        float *v_xy_abs_ptr = (float *)(v_means2d_abs) + 2 * g;
                        gpuAtomicAdd(v_xy_abs_ptr, v_xy_abs_local.x);
                        gpuAtomicAdd(v_xy_abs_ptr + 1, v_xy_abs_local.y);
                    }
                    gpuAtomicAdd(v_opacities + g, v_opacity_local);
                }
            } else if (compute_densify_grad) {
                // Legacy B2: only warpSum for v_xy (densification)
                warpSum(v_xy_local, warp);
                if (v_means2d_abs != nullptr) {
                    warpSum(v_xy_abs_local, warp);
                }
                if (warp.thread_rank() == 0) {
                    int32_t g = id_batch[t];
                    float *v_xy_ptr = (float *)(v_means2d) + 2 * g;
                    gpuAtomicAdd(v_xy_ptr, v_xy_local.x);
                    gpuAtomicAdd(v_xy_ptr + 1, v_xy_local.y);
                    if (v_means2d_abs != nullptr) {
                        float *v_xy_abs_ptr = (float *)(v_means2d_abs) + 2 * g;
                        gpuAtomicAdd(v_xy_abs_ptr, v_xy_abs_local.x);
                        gpuAtomicAdd(v_xy_abs_ptr + 1, v_xy_abs_local.y);
                    }
                }
            }"""

    content = content.replace(old_atomic, new_atomic, 1)
    print("  Applied: warp reduction + atomicAdd — per-branch")

    # 1f. Update shmem_size in launch function
    old_shmem_size = """    int64_t shmem_size =
        tile_size * tile_size *
        (sizeof(int32_t) + sizeof(vec3) + sizeof(vec3) + sizeof(float) * CDIM +
         sizeof(bool));"""

    new_shmem_size = """    int64_t shmem_size =
        tile_size * tile_size *
        (sizeof(int32_t) + sizeof(vec3) + sizeof(vec3) + sizeof(float) * CDIM +
         sizeof(bool) + sizeof(bool) + sizeof(bool) + sizeof(bool));"""

    content = content.replace(old_shmem_size, new_shmem_size, 1)
    print("  Applied: shmem_size — +3 bool arrays")

    # 1g. Update launch function signature
    old_launch_sig = """    // sparse backward
    const at::optional<at::Tensor> importance_mask, // [N] or [nnz], uint8
    const bool compute_densify_grad                 // B2 mode flag
) {"""

    new_launch_sig = """    // sparse backward
    const at::optional<at::Tensor> importance_mask, // [N] or [nnz], uint8
    const bool compute_densify_grad,                // B2 mode flag
    // R2.1: per-branch masks
    const at::optional<at::Tensor> r2_geo_mask,     // [N] or [nnz], uint8
    const at::optional<at::Tensor> r2_app_mask,     // [N] or [nnz], uint8
    const at::optional<at::Tensor> r2_opacity_mask  // [N] or [nnz], uint8
) {"""

    content = content.replace(old_launch_sig, new_launch_sig, 1)
    print("  Applied: launch function signature — added 3 mask params")

    # 1h. Update kernel launch call in launch function
    old_kernel_call = """            compute_densify_grad,
            image_width,
            image_height,
            tile_size,
            tile_width,
            tile_height,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            render_alphas.data_ptr<float>(),
            last_ids.data_ptr<int32_t>(),
            v_render_colors.data_ptr<float>(),
            v_render_alphas.data_ptr<float>(),
            v_means2d_abs.has_value()
                ? reinterpret_cast<vec2 *>(
                      v_means2d_abs.value().data_ptr<float>()
                  )
                : nullptr,
            reinterpret_cast<vec2 *>(v_means2d.data_ptr<float>()),
            reinterpret_cast<vec3 *>(v_conics.data_ptr<float>()),
            v_colors.data_ptr<float>(),
            v_opacities.data_ptr<float>()
        );"""

    new_kernel_call = """            compute_densify_grad,
            r2_geo_mask.has_value()
                ? reinterpret_cast<const uint8_t *>(
                      r2_geo_mask.value().data_ptr<uint8_t>())
                : nullptr,
            r2_app_mask.has_value()
                ? reinterpret_cast<const uint8_t *>(
                      r2_app_mask.value().data_ptr<uint8_t>())
                : nullptr,
            r2_opacity_mask.has_value()
                ? reinterpret_cast<const uint8_t *>(
                      r2_opacity_mask.value().data_ptr<uint8_t>())
                : nullptr,
            image_width,
            image_height,
            tile_size,
            tile_width,
            tile_height,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            render_alphas.data_ptr<float>(),
            last_ids.data_ptr<int32_t>(),
            v_render_colors.data_ptr<float>(),
            v_render_alphas.data_ptr<float>(),
            v_means2d_abs.has_value()
                ? reinterpret_cast<vec2 *>(
                      v_means2d_abs.value().data_ptr<float>()
                  )
                : nullptr,
            reinterpret_cast<vec2 *>(v_means2d.data_ptr<float>()),
            reinterpret_cast<vec3 *>(v_conics.data_ptr<float>()),
            v_colors.data_ptr<float>(),
            v_opacities.data_ptr<float>()
        );"""

    content = content.replace(old_kernel_call, new_kernel_call, 1)
    print("  Applied: kernel launch call — pass 3 masks")

    # 1i. Update explicit instantiation macro
    old_ins = """#define __INS__(CDIM)                                                          \\
    template void launch_rasterize_to_pixels_3dgs_bwd_kernel<CDIM>(            \\
        const at::Tensor means2d,                                              \\
        const at::Tensor conics,                                               \\
        const at::Tensor colors,                                               \\
        const at::Tensor opacities,                                            \\
        const at::optional<at::Tensor> backgrounds,                            \\
        const at::optional<at::Tensor> masks,                                  \\
        uint32_t image_width,                                                  \\
        uint32_t image_height,                                                 \\
        uint32_t tile_size,                                                    \\
        const at::Tensor tile_offsets,                                         \\
        const at::Tensor flatten_ids,                                          \\
        const at::Tensor render_alphas,                                        \\
        const at::Tensor last_ids,                                             \\
        const at::Tensor v_render_colors,                                      \\
        const at::Tensor v_render_alphas,                                      \\
        at::optional<at::Tensor> v_means2d_abs,                                \\
        at::Tensor v_means2d,                                                  \\
        at::Tensor v_conics,                                                   \\
        at::Tensor v_colors,                                                   \\
        at::Tensor v_opacities,                                                \\
        const at::optional<at::Tensor> importance_mask,                        \\
        const bool compute_densify_grad                                        \\
    );"""

    new_ins = """#define __INS__(CDIM)                                                          \\
    template void launch_rasterize_to_pixels_3dgs_bwd_kernel<CDIM>(            \\
        const at::Tensor means2d,                                              \\
        const at::Tensor conics,                                               \\
        const at::Tensor colors,                                               \\
        const at::Tensor opacities,                                            \\
        const at::optional<at::Tensor> backgrounds,                            \\
        const at::optional<at::Tensor> masks,                                  \\
        uint32_t image_width,                                                  \\
        uint32_t image_height,                                                 \\
        uint32_t tile_size,                                                    \\
        const at::Tensor tile_offsets,                                         \\
        const at::Tensor flatten_ids,                                          \\
        const at::Tensor render_alphas,                                        \\
        const at::Tensor last_ids,                                             \\
        const at::Tensor v_render_colors,                                      \\
        const at::Tensor v_render_alphas,                                      \\
        at::optional<at::Tensor> v_means2d_abs,                                \\
        at::Tensor v_means2d,                                                  \\
        at::Tensor v_conics,                                                   \\
        at::Tensor v_colors,                                                   \\
        at::Tensor v_opacities,                                                \\
        const at::optional<at::Tensor> importance_mask,                        \\
        const bool compute_densify_grad,                                       \\
        const at::optional<at::Tensor> r2_geo_mask,                            \\
        const at::optional<at::Tensor> r2_app_mask,                            \\
        const at::optional<at::Tensor> r2_opacity_mask                         \\
    );"""

    content = content.replace(old_ins, new_ins, 1)
    print("  Applied: explicit instantiation macro — added 3 mask params")

    with open(filepath, 'w') as f:
        f.write(content)
    print(f"  Written {filepath}")


# ============================================================
# 2. Patch Rasterization.cpp — C++ dispatch
# ============================================================
def patch_cpp_dispatch():
    filepath = f"{CUDA_CSRC}/Rasterization.cpp"
    print(f"\n=== Patching {os.path.basename(filepath)} ===")
    backup(filepath)

    with open(filepath, 'r') as f:
        content = f.read()

    # 2a. Update function signature
    old_sig = """    // sparse backward
    const at::optional<at::Tensor> importance_mask, // [N] or [nnz], uint8
    const bool compute_densify_grad                 // B2 mode flag
) {"""

    new_sig = """    // sparse backward
    const at::optional<at::Tensor> importance_mask, // [N] or [nnz], uint8
    const bool compute_densify_grad,                // B2 mode flag
    // R2.1: per-branch masks
    const at::optional<at::Tensor> r2_geo_mask,     // [N] or [nnz], uint8
    const at::optional<at::Tensor> r2_app_mask,     // [N] or [nnz], uint8
    const at::optional<at::Tensor> r2_opacity_mask  // [N] or [nnz], uint8
) {"""

    content = content.replace(old_sig, new_sig, 1)
    print("  Applied: C++ dispatch signature — added 3 mask params")

    # 2b. Update CHECK_INPUT for new masks
    old_check = """    if (importance_mask.has_value()) {
        CHECK_INPUT(importance_mask.value());
    }"""

    new_check = """    if (importance_mask.has_value()) {
        CHECK_INPUT(importance_mask.value());
    }
    if (r2_geo_mask.has_value()) {
        CHECK_INPUT(r2_geo_mask.value());
    }
    if (r2_app_mask.has_value()) {
        CHECK_INPUT(r2_app_mask.value());
    }
    if (r2_opacity_mask.has_value()) {
        CHECK_INPUT(r2_opacity_mask.value());
    }"""

    content = content.replace(old_check, new_check, 1)
    print("  Applied: CHECK_INPUT for new masks")

    # 2c. Update __LAUNCH_KERNEL__ macro
    old_macro = """#define __LAUNCH_KERNEL__(N)                                                   \\
    case N:                                                                    \\
        launch_rasterize_to_pixels_3dgs_bwd_kernel<N>(                         \\
            means2d,                                                           \\
            conics,                                                            \\
            colors,                                                            \\
            opacities,                                                         \\
            backgrounds,                                                       \\
            masks,                                                             \\
            image_width,                                                       \\
            image_height,                                                      \\
            tile_size,                                                         \\
            tile_offsets,                                                      \\
            flatten_ids,                                                       \\
            render_alphas,                                                     \\
            last_ids,                                                          \\
            v_render_colors,                                                   \\
            v_render_alphas,                                                   \\
            absgrad ? c10::optional<at::Tensor>(v_means2d_abs) : c10::nullopt, \\
            v_means2d,                                                         \\
            v_conics,                                                          \\
            v_colors,                                                          \\
            v_opacities,                                                       \\
            importance_mask,                                                   \\
            compute_densify_grad                                               \\
        );                                                                     \\
        break;"""

    new_macro = """#define __LAUNCH_KERNEL__(N)                                                   \\
    case N:                                                                    \\
        launch_rasterize_to_pixels_3dgs_bwd_kernel<N>(                         \\
            means2d,                                                           \\
            conics,                                                            \\
            colors,                                                            \\
            opacities,                                                         \\
            backgrounds,                                                       \\
            masks,                                                             \\
            image_width,                                                       \\
            image_height,                                                      \\
            tile_size,                                                         \\
            tile_offsets,                                                      \\
            flatten_ids,                                                       \\
            render_alphas,                                                     \\
            last_ids,                                                          \\
            v_render_colors,                                                   \\
            v_render_alphas,                                                   \\
            absgrad ? c10::optional<at::Tensor>(v_means2d_abs) : c10::nullopt, \\
            v_means2d,                                                         \\
            v_conics,                                                          \\
            v_colors,                                                          \\
            v_opacities,                                                       \\
            importance_mask,                                                   \\
            compute_densify_grad,                                              \\
            r2_geo_mask,                                                       \\
            r2_app_mask,                                                       \\
            r2_opacity_mask                                                    \\
        );                                                                     \\
        break;"""

    content = content.replace(old_macro, new_macro, 1)
    print("  Applied: __LAUNCH_KERNEL__ macro — pass 3 masks")

    with open(filepath, 'w') as f:
        f.write(content)
    print(f"  Written {filepath}")


# ============================================================
# 3. Patch _wrapper.py — Python autograd Function
# ============================================================
def patch_wrapper():
    filepath = WRAPPER
    print(f"\n=== Patching {os.path.basename(filepath)} ===")
    backup(filepath)

    with open(filepath, 'r') as f:
        content = f.read()

    # 3a. Update rasterize_to_pixels() signature
    old_sig = """    importance_mask: Optional[Tensor] = None,  # [N] or [nnz], uint8
    compute_densify_grad: bool = False,  # B2 mode
) -> Tuple[Tensor, Tensor]:"""

    new_sig = """    importance_mask: Optional[Tensor] = None,  # [N] or [nnz], uint8
    compute_densify_grad: bool = False,  # B2 mode
    r2_geo_mask: Optional[Tensor] = None,  # [N] or [nnz], uint8, R2.1
    r2_app_mask: Optional[Tensor] = None,  # [N] or [nnz], uint8, R2.1
    r2_opacity_mask: Optional[Tensor] = None,  # [N] or [nnz], uint8, R2.1
) -> Tuple[Tensor, Tensor]:"""

    content = content.replace(old_sig, new_sig, 1)
    print("  Applied: rasterize_to_pixels() signature — added 3 mask params")

    # 3b. Update mask contiguity + apply call
    old_apply = """    if importance_mask is not None:
        importance_mask = importance_mask.contiguous()
    render_colors, render_alphas = _RasterizeToPixels.apply(
        means2d.contiguous(),
        conics.contiguous(),
        colors.contiguous(),
        opacities.contiguous(),
        backgrounds,
        masks,
        image_width,
        image_height,
        tile_size,
        isect_offsets.contiguous(),
        flatten_ids.contiguous(),
        absgrad,
        importance_mask,
        compute_densify_grad,
    )"""

    new_apply = """    if importance_mask is not None:
        importance_mask = importance_mask.contiguous()
    if r2_geo_mask is not None:
        r2_geo_mask = r2_geo_mask.contiguous()
    if r2_app_mask is not None:
        r2_app_mask = r2_app_mask.contiguous()
    if r2_opacity_mask is not None:
        r2_opacity_mask = r2_opacity_mask.contiguous()
    render_colors, render_alphas = _RasterizeToPixels.apply(
        means2d.contiguous(),
        conics.contiguous(),
        colors.contiguous(),
        opacities.contiguous(),
        backgrounds,
        masks,
        image_width,
        image_height,
        tile_size,
        isect_offsets.contiguous(),
        flatten_ids.contiguous(),
        absgrad,
        importance_mask,
        compute_densify_grad,
        r2_geo_mask,
        r2_app_mask,
        r2_opacity_mask,
    )"""

    content = content.replace(old_apply, new_apply, 1)
    print("  Applied: rasterize_to_pixels() apply call — pass 3 masks")

    # 3c. Update _RasterizeToPixels.forward() signature + ctx storage
    old_fwd_sig = """    @staticmethod
    def forward(
        ctx,
        means2d: Tensor,  # [..., N, 2] or [nnz, 2]
        conics: Tensor,  # [..., N, 3] or [nnz, 3]
        colors: Tensor,  # [..., N, channels] or [nnz, channels]
        opacities: Tensor,  # [..., N] or [nnz]
        backgrounds: Tensor,  # [..., channels], Optional
        masks: Tensor,  # [..., tile_height, tile_width], Optional
        width: int,
        height: int,
        tile_size: int,
        isect_offsets: Tensor,  # [..., tile_height, tile_width]
        flatten_ids: Tensor,  # [n_isects]
        absgrad: bool,
        importance_mask: Tensor,  # [N] or [nnz], uint8, Optional
        compute_densify_grad: bool,  # B2 mode
    ) -> Tuple[Tensor, Tensor]:"""

    new_fwd_sig = """    @staticmethod
    def forward(
        ctx,
        means2d: Tensor,  # [..., N, 2] or [nnz, 2]
        conics: Tensor,  # [..., N, 3] or [nnz, 3]
        colors: Tensor,  # [..., N, channels] or [nnz, channels]
        opacities: Tensor,  # [..., N] or [nnz]
        backgrounds: Tensor,  # [..., channels], Optional
        masks: Tensor,  # [..., tile_height, tile_width], Optional
        width: int,
        height: int,
        tile_size: int,
        isect_offsets: Tensor,  # [..., tile_height, tile_width]
        flatten_ids: Tensor,  # [n_isects]
        absgrad: bool,
        importance_mask: Tensor,  # [N] or [nnz], uint8, Optional
        compute_densify_grad: bool,  # B2 mode
        r2_geo_mask: Tensor,  # [N] or [nnz], uint8, Optional, R2.1
        r2_app_mask: Tensor,  # [N] or [nnz], uint8, Optional, R2.1
        r2_opacity_mask: Tensor,  # [N] or [nnz], uint8, Optional, R2.1
    ) -> Tuple[Tensor, Tensor]:"""

    content = content.replace(old_fwd_sig, new_fwd_sig, 1)
    print("  Applied: _RasterizeToPixels.forward() signature — added 3 mask params")

    # 3d. Update ctx storage in forward
    old_ctx = """        ctx.importance_mask = importance_mask
        ctx.compute_densify_grad = compute_densify_grad"""

    new_ctx = """        ctx.importance_mask = importance_mask
        ctx.compute_densify_grad = compute_densify_grad
        ctx.r2_geo_mask = r2_geo_mask
        ctx.r2_app_mask = r2_app_mask
        ctx.r2_opacity_mask = r2_opacity_mask"""

    content = content.replace(old_ctx, new_ctx, 1)
    print("  Applied: ctx storage — 3 masks")

    # 3e. Update backward() — retrieve masks + pass to C function
    old_bwd_masks = """        importance_mask = ctx.importance_mask
        compute_densify_grad = ctx.compute_densify_grad
        (
            v_means2d_abs,
            v_means2d,
            v_conics,
            v_colors,
            v_opacities,
        ) = _make_lazy_cuda_func("rasterize_to_pixels_3dgs_bwd")(
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            masks,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
            v_render_colors.contiguous(),
            v_render_alphas.contiguous(),
            absgrad,
            importance_mask,
            compute_densify_grad,
        )"""

    new_bwd_masks = """        importance_mask = ctx.importance_mask
        compute_densify_grad = ctx.compute_densify_grad
        r2_geo_mask = ctx.r2_geo_mask
        r2_app_mask = ctx.r2_app_mask
        r2_opacity_mask = ctx.r2_opacity_mask
        (
            v_means2d_abs,
            v_means2d,
            v_conics,
            v_colors,
            v_opacities,
        ) = _make_lazy_cuda_func("rasterize_to_pixels_3dgs_bwd")(
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            masks,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
            v_render_colors.contiguous(),
            v_render_alphas.contiguous(),
            absgrad,
            importance_mask,
            compute_densify_grad,
            r2_geo_mask,
            r2_app_mask,
            r2_opacity_mask,
        )"""

    content = content.replace(old_bwd_masks, new_bwd_masks, 1)
    print("  Applied: backward() — pass 3 masks to C function")

    # 3f. Update backward() return tuple
    old_return = """        return (
            v_means2d,
            v_conics,
            v_colors,
            v_opacities,
            v_backgrounds,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,  # importance_mask
            None,  # compute_densify_grad
        )"""

    new_return = """        return (
            v_means2d,
            v_conics,
            v_colors,
            v_opacities,
            v_backgrounds,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,  # importance_mask
            None,  # compute_densify_grad
            None,  # r2_geo_mask
            None,  # r2_app_mask
            None,  # r2_opacity_mask
        )"""

    content = content.replace(old_return, new_return, 1)
    print("  Applied: backward() return — 3 None for masks")

    with open(filepath, 'w') as f:
        f.write(content)
    print(f"  Written {filepath}")


# ============================================================
# 4. Patch rendering.py — top-level rasterization()
# ============================================================
def patch_rendering():
    filepath = RENDERING
    print(f"\n=== Patching {os.path.basename(filepath)} ===")
    backup(filepath)

    with open(filepath, 'r') as f:
        content = f.read()

    # 4a. Find the rasterization() function signature and add mask params
    # The function has a long signature; we add after compute_densify_grad
    old_rasterize_call = """                absgrad=absgrad,
                importance_mask=importance_mask,
                compute_densify_grad=compute_densify_grad,
            )
            render_colors.append(render_colors_)
            render_alphas.append(render_alphas_)"""

    new_rasterize_call = """                absgrad=absgrad,
                importance_mask=importance_mask,
                compute_densify_grad=compute_densify_grad,
                r2_geo_mask=r2_geo_mask,
                r2_app_mask=r2_app_mask,
                r2_opacity_mask=r2_opacity_mask,
            )
            render_colors.append(render_colors_)
            render_alphas.append(render_alphas_)"""

    if old_rasterize_call in content:
        content = content.replace(old_rasterize_call, new_rasterize_call, 1)
        print("  Applied: chunked rasterize_to_pixels call — pass 3 masks")
    else:
        print("  WARNING: chunked call not found, trying non-chunked")

    # Non-chunked path
    old_rasterize_call2 = """                absgrad=absgrad,
                importance_mask=importance_mask,
                compute_densify_grad=compute_densify_grad,
            )
    if render_mode in ["ED", "RGB+ED"]:"""

    new_rasterize_call2 = """                absgrad=absgrad,
                importance_mask=importance_mask,
                compute_densify_grad=compute_densify_grad,
                r2_geo_mask=r2_geo_mask,
                r2_app_mask=r2_app_mask,
                r2_opacity_mask=r2_opacity_mask,
            )
    if render_mode in ["ED", "RGB+ED"]:"""

    if old_rasterize_call2 in content:
        content = content.replace(old_rasterize_call2, new_rasterize_call2, 1)
        print("  Applied: non-chunked rasterize_to_pixels call — pass 3 masks")
    else:
        print("  WARNING: non-chunked call not found either")

    # 4b. Add mask params to rasterization() function signature
    # Find the function def and add after compute_densify_grad parameter
    old_param = """    compute_densify_grad: bool = False,"""

    # This might appear in multiple places (rasterization and _rasterization)
    # We need to add to the public rasterization() function
    # Let's find the first occurrence (which is the public function)
    count = content.count(old_param)
    if count >= 1:
        # Replace all occurrences — both public and private should accept the masks
        new_param = """    compute_densify_grad: bool = False,
    r2_geo_mask: Optional[Tensor] = None,
    r2_app_mask: Optional[Tensor] = None,
    r2_opacity_mask: Optional[Tensor] = None,"""
        content = content.replace(old_param, new_param)
        print(f"  Applied: function signatures — added 3 mask params ({count} occurrences)")

    with open(filepath, 'w') as f:
        f.write(content)
    print(f"  Written {filepath}")


if __name__ == "__main__":
    print("R2.1 Patch Script — Adding per-branch masks to rasterize_to_pixels_3dgs_bwd")
    patch_cuda_kernel()
    patch_cpp_dispatch()
    patch_wrapper()
    patch_rendering()
    print("\n=== All patches applied ===")
    print("Now rebuild gsplat with: cd /tmp/gsplat_baseline/gsplat-1.5.3 && pip install -e . --no-build-isolation")
