// C17-1: exact tile-local construction.  This file is inserted just before
// `} // namespace gsplat` in gsplat/cuda/csrc/IntersectTile.cu by the
// reproducible patcher in this directory.

template <typename scalar_t>
__global__ void c17_tile_local_count_kernel(
    const bool packed, const uint32_t I, const uint32_t N, const uint32_t nnz,
    const int64_t* __restrict__ image_ids, const scalar_t* __restrict__ means2d,
    const int32_t* __restrict__ radii, const scalar_t* __restrict__ depths,
    const uint32_t tile_size, const uint32_t tile_width, const uint32_t tile_height,
    int32_t* __restrict__ tiles_per_gauss, int32_t* __restrict__ tile_counts) {
    const uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= (packed ? nnz : I * N)) return;
    const float radius_x = radii[idx * 2];
    const float radius_y = radii[idx * 2 + 1];
    if (radius_x <= 0 || radius_y <= 0) { tiles_per_gauss[idx] = 0; return; }
    const vec2 mean2d = glm::make_vec2(means2d + 2 * idx);
    const float tx = mean2d.x / float(tile_size), ty = mean2d.y / float(tile_size);
    const float rx = radius_x / float(tile_size), ry = radius_y / float(tile_size);
    const uint2 tile_min = make_uint2(min(max(0, (uint32_t)floor(tx-rx)), tile_width),
                                      min(max(0, (uint32_t)floor(ty-ry)), tile_height));
    const uint2 tile_max = make_uint2(min(max(0, (uint32_t)ceil(tx+rx)), tile_width),
                                      min(max(0, (uint32_t)ceil(ty+ry)), tile_height));
    tiles_per_gauss[idx] = static_cast<int32_t>((tile_max.y-tile_min.y)*(tile_max.x-tile_min.x));
    const uint32_t iid = packed ? static_cast<uint32_t>(image_ids[idx]) : idx / N;
    for (uint32_t y = tile_min.y; y < tile_max.y; ++y)
        for (uint32_t x = tile_min.x; x < tile_max.x; ++x)
            atomicAdd(&tile_counts[iid * tile_width * tile_height + y * tile_width + x], 1);
}

template <typename scalar_t>
__global__ void c17_tile_local_write_kernel(
    const bool packed, const uint32_t I, const uint32_t N, const uint32_t nnz,
    const int64_t* __restrict__ image_ids, const scalar_t* __restrict__ means2d,
    const int32_t* __restrict__ radii, const scalar_t* __restrict__ depths,
    const uint32_t tile_size, const uint32_t tile_width, const uint32_t tile_height,
    const int32_t* __restrict__ tile_offsets, int32_t* __restrict__ tile_cursors,
    uint64_t* __restrict__ local_order_keys, int32_t* __restrict__ flatten_ids) {
    const uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= (packed ? nnz : I * N)) return;
    const float radius_x = radii[idx * 2];
    const float radius_y = radii[idx * 2 + 1];
    if (radius_x <= 0 || radius_y <= 0) return;
    const vec2 mean2d = glm::make_vec2(means2d + 2 * idx);
    const float tx = mean2d.x / float(tile_size), ty = mean2d.y / float(tile_size);
    const float rx = radius_x / float(tile_size), ry = radius_y / float(tile_size);
    const uint2 tile_min = make_uint2(min(max(0, (uint32_t)floor(tx-rx)), tile_width),
                                      min(max(0, (uint32_t)floor(ty-ry)), tile_height));
    const uint2 tile_max = make_uint2(min(max(0, (uint32_t)ceil(tx+rx)), tile_width),
                                      min(max(0, (uint32_t)ceil(ty+ry)), tile_height));
    const uint32_t iid = packed ? static_cast<uint32_t>(image_ids[idx]) : idx / N;
    // This is deliberately not the legacy (tile, depth) key.  The tile is
    // supplied by the arena segment; the temporary key is (depth bits, input
    // index).  The input-index suffix restores CUB's stable baseline tie rule.
    const uint32_t depth_bits = *reinterpret_cast<const uint32_t*>(depths + idx);
    const uint64_t order_key = (static_cast<uint64_t>(depth_bits) << 32) | idx;
    for (uint32_t y = tile_min.y; y < tile_max.y; ++y) {
        for (uint32_t x = tile_min.x; x < tile_max.x; ++x) {
            const uint32_t tile = iid * tile_width * tile_height + y * tile_width + x;
            const int32_t local = atomicAdd(&tile_cursors[tile], 1);
            const int32_t out = tile_offsets[tile] + local;
            local_order_keys[out] = order_key;
            flatten_ids[out] = static_cast<int32_t>(idx);
        }
    }
}

void launch_c17_tile_local_count_kernel(
    const at::Tensor means2d, const at::Tensor radii, const at::Tensor depths,
    const at::optional<at::Tensor> image_ids, const uint32_t I,
    const uint32_t tile_size, const uint32_t tile_width, const uint32_t tile_height,
    at::Tensor tiles_per_gauss, at::Tensor tile_counts) {
    const bool packed = means2d.dim() == 2;
    const uint32_t N = packed ? 0 : means2d.size(-2);
    const uint32_t nnz = packed ? means2d.size(0) : 0;
    const int64_t n = packed ? nnz : int64_t(I) * N;
    if (!n) return;
    const dim3 threads(256), grid((n + threads.x - 1) / threads.x);
    AT_DISPATCH_FLOATING_TYPES(means2d.scalar_type(), "c17_tile_local_count_kernel", [&] {
        c17_tile_local_count_kernel<scalar_t><<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            packed, I, N, nnz, image_ids ? image_ids.value().data_ptr<int64_t>() : nullptr,
            means2d.data_ptr<scalar_t>(), radii.data_ptr<int32_t>(), depths.data_ptr<scalar_t>(),
            tile_size, tile_width, tile_height, tiles_per_gauss.data_ptr<int32_t>(), tile_counts.data_ptr<int32_t>());
    });
}

void launch_c17_tile_local_write_kernel(
    const at::Tensor means2d, const at::Tensor radii, const at::Tensor depths,
    const at::optional<at::Tensor> image_ids, const uint32_t I,
    const uint32_t tile_size, const uint32_t tile_width, const uint32_t tile_height,
    const at::Tensor tile_offsets, at::Tensor tile_cursors, at::Tensor local_order_keys,
    at::Tensor flatten_ids) {
    const bool packed = means2d.dim() == 2;
    const uint32_t N = packed ? 0 : means2d.size(-2);
    const uint32_t nnz = packed ? means2d.size(0) : 0;
    const int64_t n = packed ? nnz : int64_t(I) * N;
    if (!n) return;
    const dim3 threads(256), grid((n + threads.x - 1) / threads.x);
    AT_DISPATCH_FLOATING_TYPES(means2d.scalar_type(), "c17_tile_local_write_kernel", [&] {
        c17_tile_local_write_kernel<scalar_t><<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            packed, I, N, nnz, image_ids ? image_ids.value().data_ptr<int64_t>() : nullptr,
            means2d.data_ptr<scalar_t>(), radii.data_ptr<int32_t>(), depths.data_ptr<scalar_t>(),
            tile_size, tile_width, tile_height, tile_offsets.data_ptr<int32_t>(),
            tile_cursors.data_ptr<int32_t>(), local_order_keys.data_ptr<uint64_t>(), flatten_ids.data_ptr<int32_t>());
    });
}

void c17_tile_local_sort(
    const int64_t n_isects, const int32_t n_segments, const at::Tensor tile_offsets,
    at::Tensor local_order_keys, at::Tensor flatten_ids,
    at::Tensor local_order_keys_sorted, at::Tensor flatten_ids_sorted) {
    if (!n_isects) return;
    // CUB's segmented primitive sees already-contiguous tile ranges.  It never
    // receives a tile-id bit field and cannot reconstruct the legacy global key.
    auto opt = tile_offsets.options();
    at::Tensor segment_offsets = at::empty({int64_t(n_segments) + 1}, opt.dtype(at::kInt));
    segment_offsets.slice(0, 0, n_segments).copy_(tile_offsets);
    segment_offsets[n_segments] = static_cast<int32_t>(n_isects);
    cub::DoubleBuffer<uint64_t> keys(local_order_keys.data_ptr<uint64_t>(), local_order_keys_sorted.data_ptr<uint64_t>());
    cub::DoubleBuffer<int32_t> vals(flatten_ids.data_ptr<int32_t>(), flatten_ids_sorted.data_ptr<int32_t>());
    CUB_WRAPPER(cub::DeviceSegmentedRadixSort::SortPairs, keys, vals, n_isects, n_segments,
                segment_offsets.data_ptr<int32_t>(), segment_offsets.data_ptr<int32_t>() + 1,
                0, 64, at::cuda::getCurrentCUDAStream());
    if (vals.selector == 0) flatten_ids_sorted.set_(flatten_ids);
}
