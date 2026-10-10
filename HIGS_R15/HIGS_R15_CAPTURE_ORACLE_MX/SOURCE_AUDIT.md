# HiGS source-derived audit (2026-10-08)

Source: uploaded `HIGS_SOURCE_HANDOFF_ACTUAL.tar.gz`. **Static source analysis only**, not a live MX CUDA measurement.

## Established code path

1. `gsplat/experimental/render/functional/gaussian_inference.py`, `_HigsAutogradFunction._native_forward_capture`, around lines 2010–2110:
   - `fully_fused_projection(...)` **already produces** `radii`, `means2d`, `depths`, `conics`.
   - `opacities_bc` and optional `tile_mask` are prepared before `isect_tiles(...)`.
   - `isect_tiles` is imported *inside the function* from `gsplat.cuda._wrapper`.
2. `gsplat/cuda/_wrapper.py`, around lines 1036–1112, routes to `torch.ops.gsplat.intersect_tile`.
3. `gsplat/cuda/csrc/Intersect.cpp` defines separate `intersect_tile` (`CUDA`) and `intersect_tile_privateuseone` (`PrivateUse1`) implementations.
4. `gsplat/cuda/ext.cpp` around lines 1267 and 1300 registers both dispatch keys.
5. `gsplat/cuda/csrc/IntersectTile.cu`:
   - ordinary CUDA `intersect_tile` obtains a *first pass* per-primitive `tiles_per_gauss` count, an int64 `cumsum`, then allocates `isect_ids` and `flatten_ids` in `Intersect.cpp` (not at line 833 of `IntersectTile.cu`);
   - `IntersectTile.cu:833` is the **PrivateUse1** code path's allocation, after summing per-device totals; the function begins near line 692.
   - The CUDA kernel at `IntersectTile.cu:219` selects **AccuTile ellipse mode** if `conics` and `opacities` are non-null; plain radius AABB counting applies only otherwise.
   - In the PrivateUse1 path, `c10::cuda::device_count()` determines the number of workers, and the key range is partitioned by `chunk(total_tile_keys, device_id)`. 8 GPUs alone do not prove 8x duplicate counting: each worker receives its own key subrange.
6. Original `R8_WORKING_ENTRYPOINT.py` in the archive uses a 580K random subset (seed 42), a single identity camera and 960x540 resolution. It does **not** establish identity with any original dataset-camera R3 4.41M benchmark.

## Important unresolved questions

- The observed R10 111.28 GiB `at::empty` request has not been independently reconciled with the actual **selected dispatch key** and source line.
- A single int64 request of `119,488,359,424 bytes` *would* correspond to 14,936,044,928 elements. This is only an arithmetic inference, **not** an independently measured `n_isects`.
- A Gaussian intersects at most `tile_width * tile_height` different tiles per image under one-record-per-pair semantics. Valid primitive mapping and input tensor dimensions must be checked before using this as a claim about the historical aggregate metadata.
- At 4.41M/4 images, effective total intersections could exceed a signed 32-bit limit while some downstream code uses uint32/int32 arguments and offsets. Inspect e.g. `IntersectTile.cu` offset-kernel code around lines 935–975, as a **separate potential overflow risk**, not an established bug.

## New diagnostic route — no CUDA rebuild

The `capture_higs.py` launcher preloads `gsplat.cuda._wrapper` and temporarily wraps its `isect_tiles` Python function. `_native_forward_capture` dynamically imports that symbol, so the wrapper receives **exactly the projected tensors at the intersection call site**.

For the sampled Gaussian IDs it calls the **unmodified compiled CUDA** `isect_tiles(..., sort=False)` with `conics`, `opacities`, and the same camera/tile mask. It saves the per-Gaussian CUDA counts together with projected inputs and exits before executing the full, potentially enormous all-Gaussian allocation. The live source and `.so` files are untouched.

`oracle_higs.py` compares these GPU counts against a mathematically independent ellipse-versus-tile rectangle oracle. GPU/C++ AccuTile uses a strip-based conservative algorithm with float arithmetic, so floating-point/boundary differences alone do **not** prove a counting bug. The script also tests physical tile-count bounds and a padded radii-AABB upper bound.

No live MX results are included in this package. This method tests the same *CUDA counting implementation on sampled projected inputs*, but does not prove an original full-N count is correct. That requires subsequent full-scale count-only diagnostics if needed.
