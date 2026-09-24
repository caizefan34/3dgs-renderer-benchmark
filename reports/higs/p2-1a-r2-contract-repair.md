# P2-1A-R2 — Contract-True FP32 Native-Hierarchy Adapter Repair

**Status: `P2_1A_R2_BUILD_PASS`**. This is an integration/ABI repair and an executability smoke only; it makes no timing or renderer-correctness claim.

## Closed provenance

`1799facc0fa3c30375aad4edeff28b091ce35891cbb794f467c0984e76e14089` remains closed as `P2_1A_CORRECTNESS_FAIL_AT_ENTRY`. Its evidence remains in [p2-1a-native-hierarchy-forward.md](p2-1a-native-hierarchy-forward.md) and `artifacts/higs-p2-1a/`; it was not overwritten.

## R2 entry

`experimental::higs_native_hierarchy_from_projected(Tensor visible_ids, Tensor radii, Tensor means2d, Tensor depths, Tensor conics, Tensor opacities, Tensor colors, int width, int height, int tile_size, Tensor? background=None, bool debug=False) -> (Tensor rgb, Tensor alpha, Tensor[] diagnostics)`.

The entry consumes one F9 camera slice. It validates contiguous CUDA input layouts, converts F9's FP32 inverse-covariance lanes to the native FP32 Cholesky conic lanes, constructs `float4(l0,l1,l2,opacity)` and `float4(r,g,b,0)`, derives the native packed visibility words from F9 radii, then calls `IntersectMTFused::execute` and `IntersectMTFused::rasterize`. Thus the active path is macro count/fill, MT offsets/chunk bases, segmented macro sort, 32-G fine masks/WarpBitTranspose, active fine tile work stealing, FP32 raster, and ordered native post-compose.

It never calls `Projection.cu`, packed inference projection, SH evaluation, flat F4, or flat F5. No FP16 conversion occurs on the TEST path. C0/F9 conics, opacity, depth ordering, alpha threshold, termination, background, and native ellipse/tile support are retained by the existing native hierarchy.

With `debug=True`, the entry returns native macro offsets, sorted local rows, macro-batch offsets, active fine-tile masks, and the row-to-gid/depth maps. Together with F9 state these recover `gid`, FP32 depth, fine tile, macro batch, 32-G mini-batch, and last-contributor traversal offline, without any diagnostic write in the authoritative raster launch.

## Build and smoke

The clean build used `/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/nvcc` 12.8.93, torch 2.9.1+cu128, and `TORCH_CUDA_ARCH_LIST=8.0`, from source commit `77ab983ffe43420b2131669cb35776b883ca4c3c`. The FP32 F9-to-native smoke on an A100 passed: RGB `[192,256,3]` and alpha `[192,256,1]`, both contiguous FP32 and finite, with no dtype exception, CUDA launch error, or illegal memory access.

The R2 `.so` SHA256 is `d7cb5d83721473de3729b7baf4500592427e6eae31d0bc2b20fce126a74debae`; patch SHA256 is `0d5434fc2e943bcc0310b72bdb8368347a42136296619ec2fa2c9e1cab9c85b8`.

Raster source and raster kernel resource results are unchanged. The corrected static statement remains **3 CTA/SM, 46.875% theoretical occupancy; FP32 spills `<8>` 32/32 and `<16>` 72/76**. The obsolete 1-CTA/SM statement is not restored.

Full contracts and evidence are in `artifacts/higs-p2-1a-r2/`. DSH structural/order/RGB-alpha runtime gates may now rerun; no benchmark was run by R2.
