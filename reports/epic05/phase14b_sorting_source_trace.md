# Phase 14B — Sorting Source Trace

**Date:** 2026-09-23
**File:** gsplat v1.5.3 source trace for `isect_tiles(sort=True, segmented=True)`

---

## File Map

| File | Role |
|------|------|
| `gsplat/cuda/_wrapper.py:443-517` | Python `isect_tiles()` entry point |
| `gsplat/cuda/_wrapper.py:520-540` | `isect_offset_encode()` |
| `gsplat/cuda/csrc/Intersect.cpp` | C++ dispatch: two-pass intersect kernel + global/segmented sort selection |
| `gsplat/cuda/csrc/IntersectTile.cu:296-339` | `radix_sort_double_buffer()` — CUB global radix sort |
| `gsplat/cuda/csrc/IntersectTile.cu:343-394` | `segmented_radix_sort_double_buffer()` — CUB segmented radix sort |
| `gsplat/cuda/csrc/Intersect.h` | Header declarations for both sort functions |
| `gsplat/cuda/include/Ops.h` | `segmented` bool in operator declaration |
| `gsplat/rendering.py:634-646` | High-level `rasterization()` calling `isect_tiles(..., segmented=segmented, ...)` |

---

## Call Chain

```
rasterization(segmented=False)                              # rendering.py:58
  → isect_tiles(means2d, radii, depths, ..., 
                sort=True, segmented=False, packed=True)     # _wrapper.py:443
    → _make_lazy_cuda_func("intersect_tile")(..., sort, segmented)  # _wrapper.py:504
      → intersect_tile(C++):                                 # Intersect.cpp
        Pass 1: launch_intersect_tile_kernel(cum=null)       # count tiles_per_gauss
        cumsum → n_isects
        Pass 2: launch_intersect_tile_kernel(cum=tiles_per_gauss)
        if segmented: compute per-image offsets             # Intersect.cpp:81-90
        if sort:
          if segmented: segmented_radix_sort_double_buffer() # IntersectTile.cu:343
          else:        radix_sort_double_buffer()            # IntersectTile.cu:296
  → isect_offset_encode(isect_ids)                          # _wrapper.py:520
```

---

## Key Code Snippets

### 1. Python entry point (`_wrapper.py:443-517`)

```python
@torch.no_grad()
def isect_tiles(means2d, radii, depths, tile_size, tile_width, tile_height,
                sort=True, segmented=False, packed=False, ...):
    tiles_per_gauss, isect_ids, flatten_ids = _make_lazy_cuda_func("intersect_tile")(
        means2d.contiguous(), radii.contiguous(), depths.contiguous(),
        image_ids, gaussian_ids, I, tile_size, tile_width, tile_height,
        sort, segmented,
    )
    return tiles_per_gauss, isect_ids, flatten_ids
```

### 2. Two-pass intersect kernel + sort dispatch (`Intersect.cpp`)

```cpp
// Pass 1: count tiles per gaussian
launch_intersect_tile_kernel(... means2d, radii, depths, ..., 
    /*cum_tiles_per_gauss=*/nullopt, /*tiles_per_gauss=*/output, ...);

// cumsum tiles_per_gauss → n_isects
auto cum_tiles_per_gauss = at::cumsum(tiles_per_gauss, -1);
n_isects = cum_tiles_per_gauss[-1].item<int64_t>();

if (segmented) {
    offsets = at::cumsum(at::sum(tiles_per_gauss, -1).view({-1}), 0);
    offsets = at::cat({at::tensor({0}), offsets});
}

// Pass 2: write isect_ids and flatten_ids
launch_intersect_tile_kernel(... cum_tiles_per_gauss, ...);

// Sort
if (n_isects && sort) {
    if (segmented)
        segmented_radix_sort_double_buffer(n_isects, I, ...);  // per-image segments
    else
        radix_sort_double_buffer(n_isects, ...);               // global sort
}
```

### 3. Global radix sort (`IntersectTile.cu:296-339`)

```cpp
void radix_sort_double_buffer(n_isects, image_n_bits, tile_n_bits,
    isect_ids, flatten_ids, isect_ids_sorted, flatten_ids_sorted) 
{
    CUB_WRAPPER(
        cub::DeviceRadixSort::SortPairs,
        d_keys, d_values, n_isects,
        0,                               // begin_bit
        32 + tile_n_bits + image_n_bits,  // end_bit (all 64 bits)
        at::cuda::getCurrentCUDAStream()
    );
}
```

### 4. Segmented radix sort (`IntersectTile.cu:343-394`)

```cpp
void segmented_radix_sort_double_buffer(n_isects, n_segments, image_n_bits, tile_n_bits,
    offsets, isect_ids, flatten_ids, isect_ids_sorted, flatten_ids_sorted) 
{
    CUB_WRAPPER(
        cub::DeviceSegmentedRadixSort::SortPairs,
        d_keys, d_values, n_isects,
        n_segments,                              // = I (number of images)
        offsets.data_ptr<int64_t>(),             // segment begin
        offsets.data_ptr<int64_t>() + 1,         // segment end
        0,                                       // begin_bit
        32 + tile_n_bits,                        // end_bit (EXCLUDES image_n_bits)
        at::cuda::getCurrentCUDAStream()
    );
}
```

---

## Summary

| Property | Global sort (default) | Segmented sort |
|:---------|:---------------------:|:--------------:|
| CUB primitive | `DeviceRadixSort::SortPairs` | `DeviceSegmentedRadixSort::SortPairs` |
| Sort key width | `32 + tn + in` bits (full) | `32 + tn` bits (no image_id) |
| Segments | 1 (all intersections) | I (one per image) |
| Preprocessing | None | O(I) cumsum |
| Single-image (I=1) | Full key width | Narrower key, 1 segment |
| Pixel equivalence | Identical | Identical |
| Gradient support | Full | Full |
| Ablation | Single boolean | Same boolean |
