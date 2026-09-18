// rasterize_to_pixels_bwd_with_skip.cu
// Modified backward kernel with certificate-guided skip support.
// Based on gsplat 1.5.3 rasterize_to_pixels_bwd.cu
//
// Key change: accepts an additional `skip_mask` tensor [n_isects] that
// marks which (tile, Gaussian) intersections can be safely skipped
// in the backward pass without exceeding the certified error budget.
//
// When skip_mask[idx] == true for a Gaussian in the batch loop,
// the kernel updates T and buffer (for correctness of subsequent
// Gaussians' gradients) but skips all gradient computation and
// atomicAdd operations for that Gaussian.

#include "bindings.h"
#include "helpers.cuh"
#include "types.cuh"
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

namespace gsplat {

namespace cg = cooperative_groups;

template <uint32_t COLOR_DIM, typename S>
__global__ void rasterize_to_pixels_bwd_with_skip_kernel(
    const uint32_t C,
    const uint32_t N,
    const uint32_t n_isects,
    const bool packed,
    // fwd inputs
    const vec2<S> *__restrict__ means2d,
    const vec3<S> *__restrict__ conics,
    const S *__restrict__ colors,
    const S *__restrict__ opacities,
    const S *__restrict__ backgrounds,
    const bool *__restrict__ masks,
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets,
    const int32_t *__restrict__ flatten_ids,
    // fwd outputs
    const S *__restrict__ render_alphas,
    const int32_t *__restrict__ last_ids,
    // grad outputs
    const S *__restrict__ v_render_colors,
    const S *__restrict__ v_render_alphas,
    // skip mask: [n_isects] - true means skip this intersection's backward work
    const bool *__restrict__ skip_mask,
    // grad inputs
    vec2<S> *__restrict__ v_means2d_abs,
    vec2<S> *__restrict__ v_means2d,
    vec3<S> *__restrict__ v_conics,
    S *__restrict__ v_colors,
    S *__restrict__ v_opacities
) {
    auto block = cg::this_thread_block();
    uint32_t camera_id = block.group_index().x;
    uint32_t tile_id =
        block.group_index().y * tile_width + block.group_index().z;
    uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    tile_offsets += camera_id * tile_height * tile_width;
    render_alphas += camera_id * image_height * image_width;
    last_ids += camera_id * image_height * image_width;
    v_render_colors += camera_id * image_height * image_width * COLOR_DIM;
    v_render_alphas += camera_id * image_height * image_width;
    if (backgrounds != nullptr) {
        backgrounds += camera_id * COLOR_DIM;
    }
    if (masks != nullptr) {
        masks += camera_id * tile_height * tile_width;
    }

    if (masks != nullptr && !masks[tile_id]) {
        return;
    }

    const S px = (S)j + 0.5f;
    const S py = (S)i + 0.5f;
    const int32_t pix_id =
        min(i * image_width + j, image_width * image_height - 1);

    bool inside = (i < image_height && j < image_width);

    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end =
        (camera_id == C - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    const uint32_t num_batches =
        (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s;
    vec3<S> *xy_opacity_batch =
        reinterpret_cast<vec3<float> *>(&id_batch[block_size]);
    vec3<S> *conic_batch =
        reinterpret_cast<vec3<float> *>(&xy_opacity_batch[block_size]);
    S *rgbs_batch = (S *)&conic_batch[block_size];

    S T_final = 1.0f - render_alphas[pix_id];
    S T = T_final;
    S buffer[COLOR_DIM] = {0.f};
    const int32_t bin_final = inside ? last_ids[pix_id] : 0;

    S v_render_c[COLOR_DIM];
    GSPLAT_PRAGMA_UNROLL
    for (uint32_t k = 0; k < COLOR_DIM; ++k) {
        v_render_c[k] = v_render_colors[pix_id * COLOR_DIM + k];
    }
    const S v_render_a = v_render_alphas[pix_id];

    const uint32_t tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int32_t warp_bin_final =
        cg::reduce(warp, bin_final, cg::greater<int>());

    for (uint32_t b = 0; b < num_batches; ++b) {
        block.sync();

        const int32_t batch_end = range_end - 1 - block_size * b;
        const int32_t batch_size = min(block_size, batch_end + 1 - range_start);
        const int32_t idx = batch_end - tr;
        if (idx >= range_start) {
            int32_t g = flatten_ids[idx];
            id_batch[tr] = g;
            const vec2<S> xy = means2d[g];
            const S opac = opacities[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g];
            GSPLAT_PRAGMA_UNROLL
            for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                rgbs_batch[tr * COLOR_DIM + k] = colors[g * COLOR_DIM + k];
            }
        }
        block.sync();

        for (uint32_t t = max(0, batch_end - warp_bin_final); t < batch_size;
             ++t) {
            bool valid = inside;
            if (batch_end - t > bin_final) {
                valid = 0;
            }

            S alpha;
            S opac;
            vec2<S> delta;
            vec3<S> conic;
            S vis;

            if (valid) {
                conic = conic_batch[t];
                vec3<S> xy_opac = xy_opacity_batch[t];
                opac = xy_opac.z;
                delta = {xy_opac.x - px, xy_opac.y - py};
                S sigma = 0.5f * (conic.x * delta.x * delta.x +
                                  conic.z * delta.y * delta.y) +
                          conic.y * delta.x * delta.y;
                vis = __expf(-sigma);
                alpha = min(0.999f, opac * vis);
                if (sigma < 0.f || alpha < 1.f / 255.f) {
                    valid = false;
                }
            }

            if (!warp.any(valid)) {
                continue;
            }

            // === CERTIFICATE SKIP CHECK ===
            // Check if this intersection is marked as skippable.
            // If so, update T and buffer for correctness but skip
            // all gradient computation. The v_*_local values remain 0
            // (initialized above), so warpSum and atomicAdd are effectively
            // no-ops for skipped Gaussians.
            //
            // skip_mask is indexed by intersection position in the sorted
            // flatten_ids array. For the same (tile, Gaussian) pair, all
            // pixel threads in the warp share the same isect_idx, so the
            // skip decision is warp-uniform.
            int32_t isect_idx = batch_end - t;
            bool do_skip = (skip_mask != nullptr) && valid && skip_mask[isect_idx];

            S v_rgb_local[COLOR_DIM] = {0.f};
            vec3<S> v_conic_local = {0.f, 0.f, 0.f};
            vec2<S> v_xy_local = {0.f, 0.f};
            vec2<S> v_xy_abs_local = {0.f, 0.f};
            S v_opacity_local = 0.f;

            if (valid) {
                S ra = 1.0f / (1.0f - alpha);
                T *= ra;
                const S fac = alpha * T;

                if (!do_skip) {
                    // Full gradient computation
                    GSPLAT_PRAGMA_UNROLL
                    for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                        v_rgb_local[k] = fac * v_render_c[k];
                    }
                    S v_alpha = 0.f;
                    for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                        v_alpha +=
                            (rgbs_batch[t * COLOR_DIM + k] * T - buffer[k] * ra) *
                            v_render_c[k];
                    }

                    v_alpha += T_final * ra * v_render_a;

                    if (backgrounds != nullptr) {
                        S accum = 0.f;
                        GSPLAT_PRAGMA_UNROLL
                        for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                            accum += backgrounds[k] * v_render_c[k];
                        }
                        v_alpha += -T_final * ra * accum;
                    }

                    if (opac * vis <= 0.999f) {
                        const S v_sigma = -opac * vis * v_alpha;
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
                }

                // Update buffer regardless of skip (for correctness
                // of subsequent Gaussians' gradients)
                GSPLAT_PRAGMA_UNROLL
                for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                    buffer[k] += rgbs_batch[t * COLOR_DIM + k] * fac;
                }
            }

            // Warp reduction and atomicAdd always execute.
            // For skipped Gaussians, v_*_local are all 0, so these
            // are effectively no-ops. This avoids warp divergence
            // in the reduction path.
            warpSum<COLOR_DIM, S>(v_rgb_local, warp);
            warpSum<decltype(warp), S>(v_conic_local, warp);
            warpSum<decltype(warp), S>(v_xy_local, warp);
            if (v_means2d_abs != nullptr) {
                warpSum<decltype(warp), S>(v_xy_abs_local, warp);
            }
            warpSum<decltype(warp), S>(v_opacity_local, warp);
            if (warp.thread_rank() == 0) {
                int32_t g = id_batch[t];
                S *v_rgb_ptr = (S *)(v_colors) + COLOR_DIM * g;
                GSPLAT_PRAGMA_UNROLL
                for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                    gpuAtomicAdd(v_rgb_ptr + k, v_rgb_local[k]);
                }

                S *v_conic_ptr = (S *)(v_conics) + 3 * g;
                gpuAtomicAdd(v_conic_ptr, v_conic_local.x);
                gpuAtomicAdd(v_conic_ptr + 1, v_conic_local.y);
                gpuAtomicAdd(v_conic_ptr + 2, v_conic_local.z);

                S *v_xy_ptr = (S *)(v_means2d) + 2 * g;
                gpuAtomicAdd(v_xy_ptr, v_xy_local.x);
                gpuAtomicAdd(v_xy_ptr + 1, v_xy_local.y);

                if (v_means2d_abs != nullptr) {
                    S *v_xy_abs_ptr = (S *)(v_means2d_abs) + 2 * g;
                    gpuAtomicAdd(v_xy_abs_ptr, v_xy_abs_local.x);
                    gpuAtomicAdd(v_xy_abs_ptr + 1, v_xy_abs_local.y);
                }

                gpuAtomicAdd(v_opacities + g, v_opacity_local);
            }
        }
    }
}

// Host-side wrapper
template <uint32_t CDIM>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rasterize_to_pixels_bwd_with_skip(
    const torch::Tensor &means2d,
    const torch::Tensor &conics,
    const torch::Tensor &colors,
    const torch::Tensor &opacities,
    const at::optional<torch::Tensor> &backgrounds,
    const at::optional<torch::Tensor> &masks,
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const torch::Tensor &tile_offsets,
    const torch::Tensor &flatten_ids,
    const torch::Tensor &render_alphas,
    const torch::Tensor &last_ids,
    const torch::Tensor &v_render_colors,
    const torch::Tensor &v_render_alphas,
    const at::optional<torch::Tensor> &skip_mask,
    const bool absgrad
) {
    uint32_t C = means2d.size(0);
    uint32_t N = means2d.size(1);
    uint32_t n_isects = flatten_ids.size(0);
    uint32_t tile_height = tile_offsets.size(1);
    uint32_t tile_width = tile_offsets.size(2);

    torch::Tensor v_means2d = torch::zeros_like(means2d);
    torch::Tensor v_means2d_abs = absgrad ? torch::zeros_like(means2d) : torch::Tensor();
    torch::Tensor v_conics = torch::zeros_like(conics);
    torch::Tensor v_colors = torch::zeros_like(colors);
    torch::Tensor v_opacities = torch::zeros_like(opacities);

    const bool packed = false;

    uint32_t block_size = tile_size * tile_size;
    uint32_t smem_size = block_size * (sizeof(int32_t) + sizeof(vec3<float>) +
                                       sizeof(vec3<float>) + CDIM * sizeof(float));

    dim3 threads(tile_size, tile_size, 1);
    dim3 grid(C, tile_height, tile_width);

    if (smem_size >= 48 * 1024) {
        cudaFuncSetAttribute(
            rasterize_to_pixels_bwd_with_skip_kernel<CDIM, float>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }

    const bool *skip_ptr = nullptr;
    if (skip_mask.has_value() && skip_mask->defined()) {
        skip_ptr = skip_mask->data_ptr<bool>();
    }

    rasterize_to_pixels_bwd_with_skip_kernel<CDIM, float>
        <<<grid, threads, smem_size>>>(
            C, N, n_isects, packed,
            reinterpret_cast<const vec2<float> *>(means2d.data_ptr<float>()),
            reinterpret_cast<const vec3<float> *>(conics.data_ptr<float>()),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds->data_ptr<float>() : nullptr,
            masks.has_value() ? masks->data_ptr<bool>() : nullptr,
            image_width, image_height, tile_size, tile_width, tile_height,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            render_alphas.data_ptr<float>(),
            last_ids.data_ptr<int32_t>(),
            v_render_colors.data_ptr<float>(),
            v_render_alphas.data_ptr<float>(),
            skip_ptr,
            absgrad ? reinterpret_cast<vec2<float> *>(v_means2d_abs.data_ptr<float>()) : nullptr,
            reinterpret_cast<vec2<float> *>(v_means2d.data_ptr<float>()),
            reinterpret_cast<vec3<float> *>(v_conics.data_ptr<float>()),
            v_colors.data_ptr<float>(),
            v_opacities.data_ptr<float>()
        );

    return std::make_tuple(v_means2d_abs, v_means2d, v_conics, v_colors, v_opacities);
}

// Explicit instantiation for RGB (COLOR_DIM=3)
template std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rasterize_to_pixels_bwd_with_skip<3>(
    const torch::Tensor &,
    const torch::Tensor &,
    const torch::Tensor &,
    const torch::Tensor &,
    const at::optional<torch::Tensor> &,
    const at::optional<torch::Tensor> &,
    const uint32_t,
    const uint32_t,
    const uint32_t,
    const torch::Tensor &,
    const torch::Tensor &,
    const torch::Tensor &,
    const torch::Tensor &,
    const torch::Tensor &,
    const torch::Tensor &,
    const at::optional<torch::Tensor> &,
    const bool
);

}  // namespace gsplat
