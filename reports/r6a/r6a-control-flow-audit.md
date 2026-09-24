# R6-A Control-Flow Audit

## Scope and provenance

This audit precedes the R6-A CUDA modification.  It inspected the frozen B1A
source at `/mnt/storage_pool/liaoyuanjun/gsplat-true-accutile-v153`, gsplat
commit `937e29912570c372bed6747a5c9bf85fed877bae`, with the B1A AccuTile port
identified by SHA-256 `33292a08ebb74437b5108fcf9282b01ac9621d45495bbba219f8b41000c803f1`.
The audited backward source is
`gsplat/cuda/csrc/RasterizeToPixels3DGSBwd.cu`, SHA-256
`2848738467309afa7d277c692ae1a53643f0658a23a06f066b8eaa9daf67c455`.
AccuTile affects the forward-produced intersection list, not this backward
control flow.

## A. Exact work mapping

`grid = {I, tile_height, tile_width}` and `threads = {tile_size, tile_size, 1}`.
Thus one block is one `(image_id, tile_y, tile_x)` tile.  With the frozen
16x16 configuration, it contains 256 threads, eight consecutive 32-lane
warps, and maps lane `(threadIdx.y, threadIdx.x)` to pixel
`(tile_y * 16 + threadIdx.y, tile_x * 16 + threadIdx.x)`.  Edge threads remain
in the block but have `inside=false`.

For the block's tile range `[range_start, range_end)`, batches have at most
`block_size` entries.  Batch `b` is loaded in reverse intersection order:
`batch_end = range_end - 1 - block_size*b`, `idx = batch_end - thread_rank`,
and `id_batch[thread_rank] = flatten_ids[idx]`.  `t=0` is the farthest
Gaussian within that reverse-loaded batch; `id_batch[t]` is therefore common
to the entire block for a given `(b,t)`.

## B. Warp alignment result: no

Warps cannot be guaranteed to process the same `id_batch[t]` at the same
dynamic loop iteration in the original kernel.  Each lane reads its own
`bin_final = inside ? last_ids[pix_id] : 0`; the kernel reduces it only within
its warp: `warp_bin_final = max_warp(bin_final)`.  The loop begins at
`first_t = max(0, batch_end - warp_bin_final)`, which therefore differs by
warp.  A warp that has no valid lanes for a particular Gaussian also performs
`continue` after `!warp.any(valid)`.  Both facts make a barrier inserted inside
the original per-warp loop unsafe: different warps could write partials for
different Gaussian IDs, or fail to reach the barrier.

## C. Existing legal block-wide synchronizations

All threads that have not returned for a false tile mask execute two existing
block-wide barriers per batch: (1) before overwriting the shared batch storage,
and (2) after all threads load the batch and before it is read.  The mask return
is block-uniform because `masks[tile_id]` is common to the block.  There is no
legal existing block barrier in the divergent `t` loop.

## D. Per-warp/per-lane state

`bin_final`, `T_final`, `T`, `buffer[CDIM]`, `valid`, and the local gradient
vectors are lane-local. `warp_bin_final` and the resulting `first_t` are
warp-local. `id_batch`, `xy_opacity_batch`, `conic_batch`, and `rgbs_batch`
are block-shared and are indexed by the common batch position `t`.

## E. Required correctness-preserving transformation

R6-A must make the Gaussian traversal structurally uniform at block scope:
every non-masked block participant executes `t=0..batch_size-1` for every
batch.  Lanes retain the original validity predicate; invalid work contributes
zero and does not modify `T` or `buffer`.  Each warp writes one zero-or-reduced
partial for the common `id_batch[t]`, all threads synchronize, one designated
warp reduces the eight partials and one lane emits the global atomics, then all
threads synchronize before the next `t`.  This is Design A (uniform block
Gaussian loop).  It preserves the mathematical per-pixel sequence but removes
the original warp-level `first_t` work skip; that cost must be measured.

No CUDA behavior was changed before completing this audit.
