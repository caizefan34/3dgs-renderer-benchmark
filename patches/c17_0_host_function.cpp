// C17-0: Host-side tile-segmented sort function — goes into Intersect.cpp
// This function is inserted before the closing namespace brace in Intersect.cpp
//
// It uses the launch wrappers from IntersectTile.cu + CUB segmented sort

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
    launch_histogram_tile_kernel(
        n_isects, isect_ids, I, n_tiles, tile_n_bits, tile_counts);

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
    launch_counting_sort_scatter_kernel(
        n_isects, isect_ids, flatten_ids,
        I, n_tiles, tile_n_bits,
        tile_offsets, write_ptrs,
        isect_ids_grouped, flatten_ids_grouped);

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
