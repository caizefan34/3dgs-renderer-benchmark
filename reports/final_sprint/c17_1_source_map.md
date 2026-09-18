# C17-1 source map

Frozen baseline: gsplat 1.5.3.

`IntersectTile.cu::intersect_tile_kernel` performs baseline per-Gaussian count
and materialization.  Pass two writes an int64 key with image/tile/depth and an
int32 `flatten_id`. `Intersect.cpp::intersect_tile` calls `at::cumsum`, reads
the total on host, then calls `radix_sort_double_buffer`. `IntersectTile.cu::radix_sort_double_buffer`
uses CUB `DeviceRadixSort::SortPairs` on the composite key. `intersect_offset_kernel`
then derives tile ranges.

`RasterizeToPixels3DGSFwd.cu` and `RasterizeToPixels3DGSBwd.cu` consume only
`isect_offsets` and sorted `flatten_ids`; their ABI is preserved by C17.

The opt-in patch is [apply_c17_1_patch.py](/C:/Users/36570/3dgs-renderer-benchmark/scripts/final_sprint/apply_c17_1_patch.py).
