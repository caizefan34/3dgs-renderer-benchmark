#include "bindings.h"
#include "helpers.cuh"
#include "types.cuh"
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

namespace gsplat {

namespace cg = cooperative_groups;

// AccuTile's conservative predicate.  It uses the same opacity-thresholded
// conic as upstream gsplat, but evaluates the exact continuous minimum of q
// over each candidate tile.  The small positive margin only retains boundary
// tiles, so it cannot remove a tile containing a rasterizer pixel center.
inline __device__ float accutile_q(
    const float A, const float B, const float C, const float dx, const float dy
) {
    return A * dx * dx + 2.f * B * dx * dy + C * dy * dy;
}

inline __device__ bool accutile_tile_may_contribute(
    const float A,
    const float B,
    const float C,
    const float mean_x,
    const float mean_y,
    const float threshold,
    const float x0,
    const float y0,
    const float x1,
    const float y1
) {
    const float dx0 = x0 - mean_x;
    const float dx1 = x1 - mean_x;
    const float dy0 = y0 - mean_y;
    const float dy1 = y1 - mean_y;
    float qmin = fminf(fminf(accutile_q(A, B, C, dx0, dy0), accutile_q(A, B, C, dx0, dy1)),
                        fminf(accutile_q(A, B, C, dx1, dy0), accutile_q(A, B, C, dx1, dy1)));
    const float dy_at_x0 = fminf(dy1, fmaxf(dy0, -B * dx0 / C));
    const float dy_at_x1 = fminf(dy1, fmaxf(dy0, -B * dx1 / C));
    const float dx_at_y0 = fminf(dx1, fmaxf(dx0, -B * dy0 / A));
    const float dx_at_y1 = fminf(dx1, fmaxf(dx0, -B * dy1 / A));
    qmin = fminf(qmin, fminf(accutile_q(A, B, C, dx0, dy_at_x0), accutile_q(A, B, C, dx1, dy_at_x1)));
    qmin = fminf(qmin, fminf(accutile_q(A, B, C, dx_at_y0, dy0), accutile_q(A, B, C, dx_at_y1, dy1)));
    if (dx0 <= 0.f && dx1 >= 0.f && dy0 <= 0.f && dy1 >= 0.f) {
        qmin = 0.f;
    }
    return qmin <= threshold + 1e-5f * fmaxf(1.f, threshold);
}

/****************************************************************************
 * Gaussian Tile Intersection
 ****************************************************************************/

template <typename T>
__global__ void isect_tiles(
    // if the data is [C, N, ...] or [nnz, ...] (packed)
    const bool packed,
    // parallelize over C * N, only used if packed is False
    const uint32_t C,
    const uint32_t N,
    // parallelize over nnz, only used if packed is True
    const uint32_t nnz,
    const int64_t *__restrict__ camera_ids,   // [nnz] optional
    const int64_t *__restrict__ gaussian_ids, // [nnz] optional
    // data
    const T *__restrict__ means2d,                   // [C, N, 2] or [nnz, 2]
    const int32_t *__restrict__ radii,               // [C, N] or [nnz]
    const T *__restrict__ depths,                    // [C, N] or [nnz]
    const float *__restrict__ conics,                // [C, N, 3] or [nnz], optional
    const float *__restrict__ opacities,             // [C, N] or [nnz], optional
    const int64_t *__restrict__ cum_tiles_per_gauss, // [C, N] or [nnz]
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const uint32_t tile_n_bits,
    int32_t *__restrict__ tiles_per_gauss, // [C, N] or [nnz]
    int64_t *__restrict__ isect_ids,       // [n_isects]
    int32_t *__restrict__ flatten_ids      // [n_isects]
) {
    // For now we'll upcast float16 and bfloat16 to float32
    using OpT = typename OpType<T>::type;

    // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    bool first_pass = cum_tiles_per_gauss == nullptr;
    if (idx >= (packed ? nnz : C * N)) {
        return;
    }

    const OpT radius = radii[idx];
    if (radius <= 0) {
        if (first_pass) {
            tiles_per_gauss[idx] = 0;
        }
        return;
    }

    vec2<OpT> mean2d = glm::make_vec2(means2d + 2 * idx);

    OpT tile_radius = radius / static_cast<OpT>(tile_size);
    OpT tile_x = mean2d.x / static_cast<OpT>(tile_size);
    OpT tile_y = mean2d.y / static_cast<OpT>(tile_size);

    // tile_min is inclusive, tile_max is exclusive
    uint2 tile_min, tile_max;
    tile_min.x = min(max(0, (uint32_t)floor(tile_x - tile_radius)), tile_width);
    tile_min.y =
        min(max(0, (uint32_t)floor(tile_y - tile_radius)), tile_height);
    tile_max.x = min(max(0, (uint32_t)ceil(tile_x + tile_radius)), tile_width);
    tile_max.y = min(max(0, (uint32_t)ceil(tile_y + tile_radius)), tile_height);

    // `radii` still defines the candidate AABB exactly as the frozen path.
    // Passing both tensors enables conservative conic-vs-tile rejection.
    bool use_accutile = conics != nullptr && opacities != nullptr;
    float A = 0.f, B = 0.f, Cc = 0.f, threshold = 0.f;
    if (use_accutile) {
        A = conics[idx * 3];
        B = conics[idx * 3 + 1];
        Cc = conics[idx * 3 + 2];
        const float opacity = opacities[idx];
        threshold = fminf(3.33f * 3.33f, 2.f * logf(opacity * 255.f));
        // Invalid or sub-threshold inputs retain the original AABB path.
        use_accutile = isfinite(A) && isfinite(B) && isfinite(Cc) &&
                       isfinite(threshold) && A > 0.f && Cc > 0.f &&
                       A * Cc - B * B > 0.f && threshold > 0.f;
    }

    if (first_pass) {
        if (use_accutile) {
            int32_t count = 0;
            for (int32_t i = tile_min.y; i < tile_max.y; ++i) {
                for (int32_t j = tile_min.x; j < tile_max.x; ++j) {
                    if (accutile_tile_may_contribute(
                            A, B, Cc, mean2d.x, mean2d.y, threshold,
                            j * tile_size, i * tile_size,
                            (j + 1) * tile_size, (i + 1) * tile_size)) {
                        ++count;
                    }
                }
            }
            tiles_per_gauss[idx] = count;
            return;
        }
        // first pass only writes out tiles_per_gauss
        tiles_per_gauss[idx] = static_cast<int32_t>(
            (tile_max.y - tile_min.y) * (tile_max.x - tile_min.x)
        );
        return;
    }

    int64_t cid; // camera id
    if (packed) {
        // parallelize over nnz
        cid = camera_ids[idx];
        // gid = gaussian_ids[idx];
    } else {
        // parallelize over C * N
        cid = idx / N;
        // gid = idx % N;
    }
    const int64_t cid_enc = cid << (32 + tile_n_bits);

    int64_t depth_id_enc = (int64_t) * (int32_t *)&(depths[idx]);
    int64_t cur_idx = (idx == 0) ? 0 : cum_tiles_per_gauss[idx - 1];
    for (int32_t i = tile_min.y; i < tile_max.y; ++i) {
        for (int32_t j = tile_min.x; j < tile_max.x; ++j) {
            if (use_accutile && !accutile_tile_may_contribute(
                    A, B, Cc, mean2d.x, mean2d.y, threshold,
                    j * tile_size, i * tile_size,
                    (j + 1) * tile_size, (i + 1) * tile_size)) {
                continue;
            }
            int64_t tile_id = i * tile_width + j;
            // e.g. tile_n_bits = 22:
            // camera id (10 bits) | tile id (22 bits) | depth (32 bits)
            isect_ids[cur_idx] = cid_enc | (tile_id << 32) | depth_id_enc;
            // the flatten index in [C * N] or [nnz]
            flatten_ids[cur_idx] = static_cast<int32_t>(idx);
            ++cur_idx;
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> isect_tiles_tensor(
    const torch::Tensor &means2d,                    // [C, N, 2] or [nnz, 2]
    const torch::Tensor &radii,                      // [C, N] or [nnz]
    const torch::Tensor &depths,                     // [C, N] or [nnz]
    const at::optional<torch::Tensor> &conics,       // [C, N, 3] or [nnz, 3]
    const at::optional<torch::Tensor> &opacities,    // [C, N] or [nnz]
    const at::optional<torch::Tensor> &camera_ids,   // [nnz]
    const at::optional<torch::Tensor> &gaussian_ids, // [nnz]
    const uint32_t C,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const bool sort,
    const bool double_buffer
) {
    GSPLAT_DEVICE_GUARD(means2d);
    GSPLAT_CHECK_INPUT(means2d);
    GSPLAT_CHECK_INPUT(radii);
    GSPLAT_CHECK_INPUT(depths);
    TORCH_CHECK(conics.has_value() == opacities.has_value(),
                "AccuTile requires conics and opacities together.");
    const float *conics_ptr = nullptr;
    const float *opacities_ptr = nullptr;
    if (conics.has_value()) {
        GSPLAT_CHECK_INPUT(conics.value());
        GSPLAT_CHECK_INPUT(opacities.value());
        TORCH_CHECK(conics.value().scalar_type() == torch::kFloat &&
                    opacities.value().scalar_type() == torch::kFloat,
                    "AccuTile conics and opacities must be float32.");
        TORCH_CHECK(conics.value().size(0) == means2d.size(0) &&
                    opacities.value().sizes() == depths.sizes(),
                    "AccuTile tensors must match projected Gaussian dimensions.");
        conics_ptr = conics.value().data_ptr<float>();
        opacities_ptr = opacities.value().data_ptr<float>();
    }
    if (camera_ids.has_value()) {
        GSPLAT_CHECK_INPUT(camera_ids.value());
    }
    if (gaussian_ids.has_value()) {
        GSPLAT_CHECK_INPUT(gaussian_ids.value());
    }
    bool packed = means2d.dim() == 2;

    uint32_t N = 0, nnz = 0, total_elems = 0;
    int64_t *camera_ids_ptr = nullptr;
    int64_t *gaussian_ids_ptr = nullptr;
    if (packed) {
        nnz = means2d.size(0);
        total_elems = nnz;
        TORCH_CHECK(
            camera_ids.has_value() && gaussian_ids.has_value(),
            "When packed is set, camera_ids and gaussian_ids must be provided."
        );
        camera_ids_ptr = camera_ids.value().data_ptr<int64_t>();
        gaussian_ids_ptr = gaussian_ids.value().data_ptr<int64_t>();
    } else {
        N = means2d.size(1); // number of gaussians
        total_elems = C * N;
    }

    uint32_t n_tiles = tile_width * tile_height;
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    // the number of bits needed to encode the camera id and tile id
    // Note: std::bit_width requires C++20
    // uint32_t tile_n_bits = std::bit_width(n_tiles);
    // uint32_t cam_n_bits = std::bit_width(C);
    uint32_t tile_n_bits = (uint32_t)floor(log2(n_tiles)) + 1;
    uint32_t cam_n_bits = (uint32_t)floor(log2(C)) + 1;
    // the first 32 bits are used for the camera id and tile id altogether, so
    // check if we have enough bits for them.
    assert(tile_n_bits + cam_n_bits <= 32);

    // first pass: compute number of tiles per gaussian
    torch::Tensor tiles_per_gauss =
        torch::empty_like(depths, depths.options().dtype(torch::kInt32));

    int64_t n_isects;
    torch::Tensor cum_tiles_per_gauss;
    if (total_elems) {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            means2d.scalar_type(),
            "isect_tiles_total_elems",
            [&]() {
                isect_tiles<<<
                    (total_elems + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
                    GSPLAT_N_THREADS,
                    0,
                    stream>>>(
                    packed,
                    C,
                    N,
                    nnz,
                    camera_ids_ptr,
                    gaussian_ids_ptr,
                    reinterpret_cast<scalar_t *>(means2d.data_ptr<scalar_t>()),
                    radii.data_ptr<int32_t>(),
                    depths.data_ptr<scalar_t>(),
                    conics_ptr,
                    opacities_ptr,
                    nullptr,
                    tile_size,
                    tile_width,
                    tile_height,
                    tile_n_bits,
                    tiles_per_gauss.data_ptr<int32_t>(),
                    nullptr,
                    nullptr
                );
            }
        );
        cum_tiles_per_gauss = torch::cumsum(tiles_per_gauss.view({-1}), 0);
        n_isects = cum_tiles_per_gauss[-1].item<int64_t>();
    } else {
        n_isects = 0;
    }

    // second pass: compute isect_ids and flatten_ids as a packed tensor
    torch::Tensor isect_ids =
        torch::empty({n_isects}, depths.options().dtype(torch::kInt64));
    torch::Tensor flatten_ids =
        torch::empty({n_isects}, depths.options().dtype(torch::kInt32));
    if (n_isects) {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            means2d.scalar_type(),
            "isect_tiles_n_isects",
            [&]() {
                isect_tiles<<<
                    (total_elems + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
                    GSPLAT_N_THREADS,
                    0,
                    stream>>>(
                    packed,
                    C,
                    N,
                    nnz,
                    camera_ids_ptr,
                    gaussian_ids_ptr,
                    reinterpret_cast<scalar_t *>(means2d.data_ptr<scalar_t>()),
                    radii.data_ptr<int32_t>(),
                    depths.data_ptr<scalar_t>(),
                    conics_ptr,
                    opacities_ptr,
                    cum_tiles_per_gauss.data_ptr<int64_t>(),
                    tile_size,
                    tile_width,
                    tile_height,
                    tile_n_bits,
                    nullptr,
                    isect_ids.data_ptr<int64_t>(),
                    flatten_ids.data_ptr<int32_t>()
                );
            }
        );
    }

    // optionally sort the Gaussians by isect_ids
    if (n_isects && sort) {
        torch::Tensor isect_ids_sorted = torch::empty_like(isect_ids);
        torch::Tensor flatten_ids_sorted = torch::empty_like(flatten_ids);

        // https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceRadixSort.html
        // DoubleBuffer reduce the auxiliary memory usage from O(N+P) to O(P)
        if (double_buffer) {
            // Create a set of DoubleBuffers to wrap pairs of device pointers
            cub::DoubleBuffer<int64_t> d_keys(
                isect_ids.data_ptr<int64_t>(),
                isect_ids_sorted.data_ptr<int64_t>()
            );
            cub::DoubleBuffer<int32_t> d_values(
                flatten_ids.data_ptr<int32_t>(),
                flatten_ids_sorted.data_ptr<int32_t>()
            );
            GSPLAT_CUB_WRAPPER(
                cub::DeviceRadixSort::SortPairs,
                d_keys,
                d_values,
                n_isects,
                0,
                32 + tile_n_bits + cam_n_bits,
                stream
            );
            switch (d_keys.selector) {
            case 0: // sorted items are stored in isect_ids
                isect_ids_sorted = isect_ids;
                break;
            case 1: // sorted items are stored in isect_ids_sorted
                break;
            }
            switch (d_values.selector) {
            case 0: // sorted items are stored in flatten_ids
                flatten_ids_sorted = flatten_ids;
                break;
            case 1: // sorted items are stored in flatten_ids_sorted
                break;
            }
            // printf("DoubleBuffer d_keys selector: %d\n", d_keys.selector);
            // printf("DoubleBuffer d_values selector: %d\n",
            // d_values.selector);
        } else {
            GSPLAT_CUB_WRAPPER(
                cub::DeviceRadixSort::SortPairs,
                isect_ids.data_ptr<int64_t>(),
                isect_ids_sorted.data_ptr<int64_t>(),
                flatten_ids.data_ptr<int32_t>(),
                flatten_ids_sorted.data_ptr<int32_t>(),
                n_isects,
                0,
                32 + tile_n_bits + cam_n_bits,
                stream
            );
        }
        return std::make_tuple(
            tiles_per_gauss, isect_ids_sorted, flatten_ids_sorted
        );
    } else {
        return std::make_tuple(tiles_per_gauss, isect_ids, flatten_ids);
    }
}

__global__ void isect_offset_encode(
    const uint32_t n_isects,
    const int64_t *__restrict__ isect_ids,
    const uint32_t C,
    const uint32_t n_tiles,
    const uint32_t tile_n_bits,
    int32_t *__restrict__ offsets // [C, n_tiles]
) {
    // e.g., ids: [1, 1, 1, 3, 3], n_tiles = 6
    // counts: [0, 3, 0, 2, 0, 0]
    // cumsum: [0, 3, 3, 5, 5, 5]
    // offsets: [0, 0, 3, 3, 5, 5]
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= n_isects)
        return;

    int64_t isect_id_curr = isect_ids[idx] >> 32;
    int64_t cid_curr = isect_id_curr >> tile_n_bits;
    int64_t tid_curr = isect_id_curr & ((1 << tile_n_bits) - 1);
    int64_t id_curr = cid_curr * n_tiles + tid_curr;

    if (idx == 0) {
        // write out the offsets until the first valid tile (inclusive)
        for (uint32_t i = 0; i < id_curr + 1; ++i)
            offsets[i] = static_cast<int32_t>(idx);
    }
    if (idx == n_isects - 1) {
        // write out the rest of the offsets
        for (uint32_t i = id_curr + 1; i < C * n_tiles; ++i)
            offsets[i] = static_cast<int32_t>(n_isects);
    }

    if (idx > 0) {
        // visit the current and previous isect_id and check if the (cid,
        // tile_id) pair changes.
        int64_t isect_id_prev = isect_ids[idx - 1] >> 32; // shift out the depth
        if (isect_id_prev == isect_id_curr)
            return;

        // write out the offsets between the previous and current tiles
        int64_t cid_prev = isect_id_prev >> tile_n_bits;
        int64_t tid_prev = isect_id_prev & ((1 << tile_n_bits) - 1);
        int64_t id_prev = cid_prev * n_tiles + tid_prev;
        for (uint32_t i = id_prev + 1; i < id_curr + 1; ++i)
            offsets[i] = static_cast<int32_t>(idx);
    }
}

torch::Tensor isect_offset_encode_tensor(
    const torch::Tensor &isect_ids, // [n_isects]
    const uint32_t C,
    const uint32_t tile_width,
    const uint32_t tile_height
) {
    GSPLAT_DEVICE_GUARD(isect_ids);
    GSPLAT_CHECK_INPUT(isect_ids);

    uint32_t n_isects = isect_ids.size(0);
    torch::Tensor offsets = torch::empty(
        {C, tile_height, tile_width}, isect_ids.options().dtype(torch::kInt32)
    );
    if (n_isects) {
        uint32_t n_tiles = tile_width * tile_height;
        uint32_t tile_n_bits = (uint32_t)floor(log2(n_tiles)) + 1;
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        isect_offset_encode<<<
            (n_isects + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
            GSPLAT_N_THREADS,
            0,
            stream>>>(
            n_isects,
            isect_ids.data_ptr<int64_t>(),
            C,
            n_tiles,
            tile_n_bits,
            offsets.data_ptr<int32_t>()
        );
    } else {
        offsets.fill_(0);
    }
    return offsets;
}

} // namespace gsplat
