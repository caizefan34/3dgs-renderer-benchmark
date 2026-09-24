// atomic_instrument.cu — Direct backward atomic event counter
// Counts warp-level and block-level gradient atomic events
// by replicating the backward kernel's iteration pattern.
//
// For each block (tile), iterates over Gaussians in the same order
// as rasterize_to_pixels_3dgs_bwd_kernel, checks pixel validity
// (alpha >= ALPHA_THRESHOLD, sigma >= 0, batch_end - t <= bin_final),
// and counts:
//   n_warp_atomics  = number of (warp, Gaussian) pairs with >=1 valid pixel
//   n_block_uniques = number of unique Gaussians per block with >=1 valid pixel
//
// Each warp-leader atomic event = 11 gpuAtomicAdd (3 v_rgb + 3 v_conic + 2 v_means2d + 2 v_means2d_abs + 1 v_opacity)

#include <torch/extension.h>
#include <ATen/Dispatch.h>
#include <cooperative_groups.h>
#include <c10/cuda/CUDAStream.h>

namespace gsplat {
namespace cg = cooperative_groups;

constexpr float ALPHA_THRESHOLD = 1.0f / 255.0f;

__global__ void count_backward_atomics_kernel(
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
    const float *__restrict__ render_alphas,  // [image_height * image_width]
    const int32_t *__restrict__ last_ids,     // [image_height * image_width]
    uint64_t *n_warp_atomics,                 // [1] output
    uint64_t *n_block_uniques                 // [1] output
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

    const uint32_t block_size = block.size(); // tile_size^2 = 256
    const uint32_t num_batches = (range_end - range_start + block_size - 1) / block_size;

    // Shared memory layout:
    // id_batch[block_size] : int32
    // xy_batch[block_size*2] : float
    // conic_batch[block_size*3] : float
    // opac_batch[block_size] : float
    // gauss_active[block_size] : int32 (flag: 1 if any warp was valid for this Gaussian)
    extern __shared__ char smem[];
    int32_t *id_batch = (int32_t *)smem;
    float *xy_batch = (float *)&id_batch[block_size];
    float *conic_batch = (float *)&xy_batch[block_size * 2];
    float *opac_batch = (float *)&conic_batch[block_size * 3];
    int32_t *gauss_active = (int32_t *)&opac_batch[block_size];

    const float T_final = 1.0f - render_alphas[pix_id];
    const int32_t bin_final = inside ? last_ids[pix_id] : 0;

    const uint32_t tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    // Manual warp-level max reduction (replaces cg::reduce)
    int32_t warp_bin_final = bin_final;
    for (int offset = 16; offset > 0; offset >>= 1) {
        int32_t other = __shfl_xor_sync(0xFFFFFFFF, warp_bin_final, offset);
        warp_bin_final = max(warp_bin_final, other);
    }

    // Per-block warp atomic count (use shared memory to accumulate)
    __shared__ uint64_t s_warp_count;
    __shared__ uint64_t s_unique_count;
    if (tr == 0) {
        s_warp_count = 0;
        s_unique_count = 0;
    }
    block.sync();

    for (uint32_t b = 0; b < num_batches; ++b) {
        block.sync();

        // Zero gauss_active for this batch
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
                    // Mark this Gaussian as active in this block
                    gauss_active[t] = 1;
                }
            }
        }

        // After processing all t in batch, count unique active Gaussians
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
    }
}

void count_backward_atomics(
    at::Tensor means2d,
    at::Tensor conics,
    at::Tensor opacities,
    int64_t image_width,
    int64_t image_height,
    int64_t tile_size,
    at::Tensor tile_offsets,
    at::Tensor flatten_ids,
    at::Tensor render_alphas,
    at::Tensor last_ids,
    at::Tensor n_warp_atomics,
    at::Tensor n_block_uniques
) {
    uint32_t N = means2d.size(0);
    uint32_t n_isects = flatten_ids.size(0);
    uint32_t tw = (image_width + tile_size - 1) / tile_size;
    uint32_t th = (image_height + tile_size - 1) / tile_size;

    n_warp_atomics.zero_();
    n_block_uniques.zero_();

    dim3 grid(1, th, tw);
    dim3 block(tile_size, tile_size);

    // Shared memory: id_batch + xy + conic + opac + gauss_active
    // = block_size * (4 + 8 + 12 + 4 + 4) = 256 * 32 = 8192 bytes
    uint32_t smem = block.x * block.y * (sizeof(int32_t) + 2 * sizeof(float) + 3 * sizeof(float) + sizeof(float) + sizeof(int32_t));

    auto stream = at::cuda::getCurrentCUDAStream();
    count_backward_atomics_kernel<<<grid, block, smem, stream>>>(
        N, n_isects,
        means2d.data_ptr<float>(),
        conics.data_ptr<float>(),
        opacities.data_ptr<float>(),
        (uint32_t)image_width, (uint32_t)image_height, (uint32_t)tile_size,
        tw, th,
        tile_offsets.data_ptr<int32_t>(),
        flatten_ids.data_ptr<int32_t>(),
        render_alphas.data_ptr<float>(),
        last_ids.data_ptr<int32_t>(),
        (uint64_t *)n_warp_atomics.data_ptr(),
        (uint64_t *)n_block_uniques.data_ptr()
    );
    C10_CUDA_CHECK(cudaGetLastError());
}

} // namespace gsplat

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("count_backward_atomics", &gsplat::count_backward_atomics, "Count backward atomic events");
}
