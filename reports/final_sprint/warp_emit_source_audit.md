# Warp-cooperative emit source audit

Status: `W0 PASS — unique output ranges and deterministic baseline ordering verified`.

Audited runtime is Reference V1: commit `02375033388d4348376b6b607ab85f551e498a77`,
gsplat 2.4.1+cu124, tile size 16. The production CUDA source snapshot is
`tmp_gsplat_src/IntersectTile.cu`.

| Item | Baseline implementation |
| --- | --- |
| Kernel | `intersect_tile_kernel` in `IntersectTile.cu:24-114` |
| Launch | `launch_intersect_tile_kernel`, 256 threads/CTA, `ceil(n_elements/256)` CTAs (`:116-209`) |
| Mapping | one CUDA thread owns one packed Gaussian (`idx = cg::this_grid().thread_rank()`, `:49`) |
| Pass 1 | `cum_tiles_per_gauss == nullptr`; computes clipped tile rectangle and stores rectangular count (`:50-84`) |
| Scan | `Intersect.cpp:56-80`: `int32 tiles_per_gauss → int64 cumsum` |
| Pass 2 | `Intersect.cpp:95-115` calls the same kernel with `cum_tiles_per_gauss`, `isect_ids`, and `flatten_ids` |
| Rectangle | `floor(mean/tile - radius/tile)` inclusive min; `ceil(...)` exclusive max; both clipped (`IntersectTile.cu:66-77`) |
| Output range | `start = idx == 0 ? 0 : cum_tiles_per_gauss[idx-1]` (`:102`); count is exactly `cum[idx]-start` |
| Enumeration order | nested `for y = tile_min.y..tile_max.y-1`, then `for x = tile_min.x..tile_max.x-1` (`:103-113`) |
| Key | `(image_id << (32 + tile_bits)) | (tile_id << 32) | uint32(depth_bits)` (`:87-113`) |

The exclusive prefix sum guarantees that every active Gaussian has a unique
preallocated interval `[start_g, end_g)` in both output arrays. Therefore a
warp can write `start_g + k` without atomics or allocation, provided that its
logical `k` reconstructs the exact baseline row-major tile:

```text
row = k / (tile_max.x - tile_min.x)
col = k % (tile_max.x - tile_min.x)
tile = (tile_min.y + row, tile_min.x + col)
```

This candidate changes only which CUDA lane writes an already-determined
logical output index. It does not change tile rectangles, counts, prefix sums,
key packing, sorting, offsets, or rasterization.

## Implemented isolated variant

The integration was compiled from a scratch copy of the gsplat package rather
than changing the installed baseline. Its only source edit is
`IntersectTile.cu`: pass 2 launches `32 * n_elements` logical threads, maps
`idx = thread_idx >> 5`, and substitutes the serial nested loops with the
row-major `k` mapping above. Pass 1 stays at 256 threads/CTA; W1 selected 128
threads/CTA for pass 2. The reversible patch set is retained in
`experiments/final_sprint/warp_emit/gsplat_warp_emit*.patch`.
