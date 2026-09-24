// C17-0: Tile-segmented depth-only sort — ALL FUNCTIONS
// These are appended to IntersectTile.cu (inside namespace gsplat)
// Requires: #include <ATen/Functions.h> at top of IntersectTile.cu

// C17-0 Step 1: Histogram kernel — count intersections per (image, tile)
__global__ void histogram_tile_kernel(
    const int64_t n_isects,
    const int64_t *__restrict__ isect_ids,
    const uint32_t I,
    const uint32_t n_tiles,
    const uint32_t tile_n_bits,
    int32_t *__restrict__ tile_counts  // [I * n_tiles], initialized to 0
) {
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= n_isects) return;

    int64_t upper = isect_ids[idx] >> 32;
    int64_t iid = upper >> tile_n_bits;
    int64_t tid = upper & ((1LL << tile_n_bits) - 1);
    int64_t linear_idx = iid * n_tiles + tid;

    atomicAdd(&tile_counts[linear_idx], 1);
}

// C17-0 Step 3: Counting sort scatter — group intersections by tile
__global__ void counting_sort_scatter_kernel(
    const int64_t n_isects,
    const int64_t *__restrict__ isect_ids_in,
    const int32_t *__restrict__ flatten_ids_in,
    const uint32_t I,
    const uint32_t n_tiles,
    const uint32_t tile_n_bits,
    const int32_t *__restrict__ tile_offsets,  // [I * n_tiles], exclusive scan result
    int32_t *__restrict__ write_ptrs,           // [I * n_tiles], copy of tile_offsets
    int64_t *__restrict__ isect_ids_out,
    int32_t *__restrict__ flatten_ids_out
) {
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= n_isects) return;

    int64_t upper = isect_ids_in[idx] >> 32;
    int64_t iid = upper >> tile_n_bits;
    int64_t tid = upper & ((1LL << tile_n_bits) - 1);
    int64_t linear_idx = iid * n_tiles + tid;

    int32_t pos = atomicAdd(&write_ptrs[linear_idx], 1);
    isect_ids_out[pos] = isect_ids_in[idx];
    flatten_ids_out[pos] = flatten_ids_in[idx];
}

// C17-0: Tile-segmented sort — counting sort by tile + segmented sort by depth
void tile_segmented_sort_double_buffer(
    const int64_t n_isects,
    const uint32_t I,
    const uint32_t n_tiles,
    const uint32_t image_n_bits,
    const uint32_t tile_n_bits,
    at::Tensor isect_ids,
    at::Tensor flatten_ids,
    at::Tensor isect_ids_sorted,
    at::Tensor flatten_ids_sorted
) {
    if (n_isects <= 0) return;

    auto opt = isect_ids.options();
    int32_t n_segments = static_cast<int32_t>(I) * static_cast<int32_t>(n_tiles);

    // Step 1: Histogram — count intersections per tile
    at::Tensor tile_counts = at::zeros({n_segments}, opt.dtype(at::kInt));
    {
        dim3 threads(256);
        dim3 grid((n_isects + threads.x - 1) / threads.x);
        histogram_tile_kernel<<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            n_isects,
            isect_ids.data_ptr<int64_t>(),
            I,
            n_tiles,
            tile_n_bits,
            tile_counts.data_ptr<int32_t>()
        );
    }

    // Step 2: Exclusive scan to get per-tile offsets
    at::Tensor tile_offsets = at::empty({n_segments}, opt.dtype(at::kInt));
    CUB_WRAPPER(
        cub::DeviceScan::ExclusiveSum,
        tile_counts.data_ptr<int32_t>(),
        tile_offsets.data_ptr<int32_t>(),
        n_segments,
        at::cuda::getCurrentCUDAStream()
    );

    // Step 3: Counting sort scatter — group by tile
    at::Tensor write_ptrs = tile_offsets.clone();
    at::Tensor isect_ids_grouped = at::empty_like(isect_ids);
    at::Tensor flatten_ids_grouped = at::empty_like(flatten_ids);
    {
        dim3 threads(256);
        dim3 grid((n_isects + threads.x - 1) / threads.x);
        counting_sort_scatter_kernel<<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            n_isects,
            isect_ids.data_ptr<int64_t>(),
            flatten_ids.data_ptr<int32_t>(),
            I,
            n_tiles,
            tile_n_bits,
            tile_offsets.data_ptr<int32_t>(),
            write_ptrs.data_ptr<int32_t>(),
            isect_ids_grouped.data_ptr<int64_t>(),
            flatten_ids_grouped.data_ptr<int32_t>()
        );
    }

    // Step 4: Segmented radix sort by depth [0:32] within each tile segment
    at::Tensor seg_offsets = at::empty({n_segments + 1}, opt.dtype(at::kInt));
    seg_offsets.slice(0, 0, n_segments) = tile_offsets;
    seg_offsets[n_segments] = static_cast<int32_t>(n_isects);

    cub::DoubleBuffer<int64_t> d_keys(
        isect_ids_grouped.data_ptr<int64_t>(),
        isect_ids_sorted.data_ptr<int64_t>()
    );
    cub::DoubleBuffer<int32_t> d_values(
        flatten_ids_grouped.data_ptr<int32_t>(),
        flatten_ids_sorted.data_ptr<int32_t>()
    );
    CUB_WRAPPER(
        cub::DeviceSegmentedRadixSort::SortPairs,
        d_keys,
        d_values,
        n_isects,
        n_segments,
        seg_offsets.data_ptr<int32_t>(),
        seg_offsets.data_ptr<int32_t>() + 1,
        0,
        32,  // C17-0: sort only depth bits (4 passes instead of 6)
        at::cuda::getCurrentCUDAStream()
    );
    switch (d_keys.selector) {
    case 0: isect_ids_sorted.set_(isect_ids_grouped); break;
    case 1: break;
    }
    switch (d_values.selector) {
    case 0: flatten_ids_sorted.set_(flatten_ids_grouped); break;
    case 1: break;
    }
}
