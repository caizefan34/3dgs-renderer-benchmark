// atomic_instrument_v2.cu — Direct backward atomic event counter (v2)
// Self-contained: computes bin_final internally via forward compositing,
// then counts warp-level and block-level gradient atomic events.
//
// For each block (tile):
//   Pass 1: Forward compositing (front-to-back) to find bin_final per pixel
//   Pass 2: Backward iteration (back-to-front) counting atomic events
//
// Counts:
//   n_warp_atomics  = number of (warp, Gaussian) pairs with >=1 valid pixel
//   n_block_uniques = number of unique Gaussians per block with >=1 valid pixel
// Each warp-leader event = 11 gpuAtomicAdd calls.

#include <torch/extension.h>
#include <ATen/Dispatch.h>
#include <cooperative_groups.h>
#include <c10/cuda/CUDAStream.h>

namespace gsplat {
namespace cg = cooperative_groups;

constexpr float ALPHA_THRESHOLD = 1.0f / 255.0f;

__global__ void count_backward_atomics_kernel_v2(
    const uint32_t N,
    const uint32_t n_isects,
    const float *__restrict__ means2d,       // [N, 2]
    const float *__restrict__ conics,         // [N, 3]
    const float *__restrict__ opacities,      // [N]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets, // [tile_height * tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    uint64_t *n_warp_atomics,                 // [1] output
    uint64_t *n_block_uniques,                // [1] output
    uint64_t *n_total_isects                  // [1] output (for verification)
) {
    auto block = cg::this_thread_block();
    uint32_t tile_id = block.group_index().y * tile_width + block.group_index().z;
    uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    const float px = (float)j + 0.5f;
    const float py = (float)i + 0.5f;
    const int32_t pix_id = min(i * image_width + j, image_width * image_height - 1);

    bool inside = (i < image_height && j < image_width);

    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end = (tile_id == tile_width * tile_height - 1)
                            ? n_isects
                            : tile_offsets[tile_id + 1];

    const uint32_t block_size = block.size(); // 256
    const uint32_t num_batches = (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ char smem[];
    int32_t *id_batch = (int32_t *)smem;
    float *xy_batch = (float *)&id_batch[block_size];
    float *conic_batch = (float *)&xy_batch[block_size * 2];
    float *opac_batch = (float *)&conic_batch[block_size * 3];
    int32_t *gauss_active = (int32_t *)&opac_batch[block_size];

    // Per-pixel bin_final (computed in pass 1)
    int32_t bin_final = -1;  // -1 means no contributing Gaussian yet

    const uint32_t tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    __shared__ uint64_t s_warp_count;
    __shared__ uint64_t s_unique_count;
    __shared__ uint64_t s_isect_count;
    if (tr == 0) {
        s_warp_count = 0;
        s_unique_count = 0;
        s_isect_count = 0;
    }
    block.sync();

    // ── Pass 1: Forward compositing (front-to-back) to find bin_final ──
    // The flatten_ids are sorted front-to-back (by depth).
    // We iterate forward and track the last Gaussian that contributed.
    {
        float T = 1.0f;
        for (uint32_t b = 0; b < num_batches; ++b) {
            block.sync();
            const int32_t batch_start = range_start + block_size * b;
            const int32_t batch_size = min(block_size, range_end - batch_start);
            const int32_t idx = batch_start + tr;

            if (tr < batch_size && idx < range_end) {
                int32_t g = flatten_ids[idx];
                id_batch[tr] = g;
                xy_batch[tr * 2] = means2d[g * 2];
                xy_batch[tr * 2 + 1] = means2d[g * 2 + 1];
                conic_batch[tr * 3] = conics[g * 3];
                conic_batch[tr * 3 + 1] = conics[g * 3 + 1];
                conic_batch[tr * 3 + 2] = conics[g * 3 + 2];
                opac_batch[tr] = opacities[g];
            }
            block.sync();

            if (inside) {
                for (uint32_t t = 0; t < batch_size; ++t) {
                    float cx = conic_batch[t * 3];
                    float cy = conic_batch[t * 3 + 1];
                    float cz = conic_batch[t * 3 + 2];
                    float dx = xy_batch[t * 2] - px;
                    float dy = xy_batch[t * 2 + 1] - py;
                    float sigma = 0.5f * (cx * dx * dx + cz * dy * dy) + cy * dx * dy;
                    if (sigma < 0.f) continue;
                    float vis = __expf(-sigma);
                    float alpha = min(0.999f, opac_batch[t] * vis);
                    if (alpha < ALPHA_THRESHOLD) continue;
                    // This Gaussian contributes
                    int32_t global_idx = batch_start + t;
                    bin_final = global_idx;
                    T *= (1.0f - alpha);
                    if (T < 0.0001f) {
                        // T is essentially 0, no more contributions
                        // But we still need to check remaining Gaussians in this batch
                        // (the forward kernel checks all, but once T < threshold, alpha doesn't matter)
                    }
                }
            }
            block.sync();
        }
    }

    // If no Gaussian contributed, bin_final stays -1
    if (bin_final < 0) bin_final = 0;
    if (!inside) bin_final = 0;

    // Warp-level max of bin_final (like the original kernel)
    int32_t warp_bin_final = bin_final;
    for (int offset = 16; offset > 0; offset >>= 1) {
        int32_t other = __shfl_xor_sync(0xFFFFFFFF, warp_bin_final, offset);
        warp_bin_final = max(warp_bin_final, other);
    }

    // ── Pass 2: Backward iteration (back-to-front) counting atomic events ──
    // This replicates the backward kernel's iteration pattern.
    for (uint32_t b = 0; b < num_batches; ++b) {
        block.sync();

        if (tr < block_size) gauss_active[tr] = 0;
        block.sync();

        const int32_t batch_end = range_end - 1 - block_size * b;
        const int32_t batch_size = min(block_size, batch_end + 1 - range_start);
        const int32_t idx = batch_end - tr;

        if (idx >= range_start) {
            int32_t g = flatten_ids[idx];
            id_batch[tr] = g;
            xy_batch[tr * 2] = means2d[g * 2];
            xy_batch[tr * 2 + 1] = means2d[g * 2 + 1];
            conic_batch[tr * 3] = conics[g * 3];
            conic_batch[tr * 3 + 1] = conics[g * 3 + 1];
            conic_batch[tr * 3 + 2] = conics[g * 3 + 2];
            opac_batch[tr] = opacities[g];
        }
        block.sync();

        // Count total intersections for this block
        if (tr == 0) {
            atomicAdd((unsigned long long *)&s_isect_count, (unsigned long long)batch_size);
        }

        for (uint32_t t = max(0, batch_end - warp_bin_final); t < batch_size; ++t) {
            bool valid = inside;
            if (batch_end - t > bin_final) {
                valid = false;
            }
            if (valid) {
                float cx = conic_batch[t * 3];
                float cy = conic_batch[t * 3 + 1];
                float cz = conic_batch[t * 3 + 2];
                float dx = xy_batch[t * 2] - px;
                float dy = xy_batch[t * 2 + 1] - py;
                float sigma = 0.5f * (cx * dx * dx + cz * dy * dy) + cy * dx * dy;
                float vis = __expf(-sigma);
                float alpha = min(0.999f, opac_batch[t] * vis);
                if (sigma < 0.f || alpha < ALPHA_THRESHOLD) {
                    valid = false;
                }
            }

            bool warp_valid = warp.any(valid);

            if (warp_valid) {
                if (warp.thread_rank() == 0) {
                    atomicAdd((unsigned long long *)&s_warp_count, 1ULL);
                    gauss_active[t] = 1;
                }
            }
        }

        block.sync();
        if (tr < batch_size && gauss_active[tr]) {
            atomicAdd((unsigned long long *)&s_unique_count, 1ULL);
        }
        block.sync();
    }

    // Add block-level counts to global counters
    if (tr == 0) {
        atomicAdd((unsigned long long *)n_warp_atomics, s_warp_count);
        atomicAdd((unsigned long long *)n_block_uniques, s_unique_count);
        atomicAdd((unsigned long long *)n_total_isects, s_isect_count);
    }
}

void count_backward_atomics_v2(
    at::Tensor means2d,
    at::Tensor conics,
    at::Tensor opacities,
    int64_t image_width,
    int64_t image_height,
    int64_t tile_size,
    at::Tensor tile_offsets,
    at::Tensor flatten_ids,
    at::Tensor n_warp_atomics,
    at::Tensor n_block_uniques,
    at::Tensor n_total_isects
) {
    uint32_t N = means2d.size(0);
    uint32_t n_isects = flatten_ids.size(0);
    uint32_t tw = (image_width + tile_size - 1) / tile_size;
    uint32_t th = (image_height + tile_size - 1) / tile_size;

    n_warp_atomics.zero_();
    n_block_uniques.zero_();
    n_total_isects.zero_();

    dim3 grid(1, th, tw);
    dim3 block(tile_size, tile_size);

    uint32_t smem = block.x * block.y * (sizeof(int32_t) + 2 * sizeof(float) + 3 * sizeof(float) + sizeof(float) + sizeof(int32_t));

    auto stream = at::cuda::getCurrentCUDAStream();
    count_backward_atomics_kernel_v2<<<grid, block, smem, stream>>>(
        N, n_isects,
        means2d.data_ptr<float>(),
        conics.data_ptr<float>(),
        opacities.data_ptr<float>(),
        (uint32_t)image_width, (uint32_t)image_height, (uint32_t)tile_size,
        tw, th,
        tile_offsets.data_ptr<int32_t>(),
        flatten_ids.data_ptr<int32_t>(),
        (uint64_t *)n_warp_atomics.data_ptr(),
        (uint64_t *)n_block_uniques.data_ptr(),
        (uint64_t *)n_total_isects.data_ptr()
    );
    C10_CUDA_CHECK(cudaGetLastError());
}

} // namespace gsplat

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("count_backward_atomics_v2", &gsplat::count_backward_atomics_v2, "Count backward atomic events v2");
}
