# H3-FWD-1B — Trainable HiGS Forward Co-design Preparation

Status: **DESIGN READY; NO IMPLEMENTATION / NO GO-NO-GO.** H3-FWD-1A-R2 establishes an exact B2 AccuTile-to-macro-mask bridge for room, bicycle, and garden. It also shows that the isolated exact Macro-F4 path is slower than B2 F4, so the unit of optimization must be F4+F5 rather than the macro representation alone. DSH's Macro-Raster break-even oracle is still required before a production-kernel decision.

## Scope and acceptance boundary

The first prototype is deliberately narrow:

```text
B2 F0 cull -> B2 F1 gather -> B2 FP32 F2 projection -> B2 FP32 F3 SH
-> exact HiGS macro structure -> FP32 HiGS macro raster -> RGB + alpha + last_ids
```

It has no autograd registration and no backward kernel. It must exactly reproduce B2 coverage, depth order except equal-depth permutations, RGB, alpha, active-pixel status, termination status, and the defined `last_ids` convention. It must not use FP16 conics, colors, partial RGBT, final RGBT, approximate culling, or altered support.

## 1. Official inference execution graph

The pinned official inference path is represented by `.build_tmp/r6b_b1fixed/gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference/`. `GaussianRenderInferenceScene.cu` owns persistent scene/work buffers and calls `IntersectMTFused::execute`, then `IntersectMTFused::rasterize`.

```text
projection + SH -> visible / FP32 means2d, depths / FP16 conics, colors
  -> COUNT macro coverage in the inference conic representation (8192-G chunk CTA; shared macro histogram)
  -> chunk-base scan -> macro offsets + 1024-G batch offsets
  -> FILL macro (depth-key, Gaussian-ID) pairs -> segmented per-macro depth sort
  -> one 320-thread CTA per (macro, 1024-G batch)
       -> 32-G mini-batch loads, per-G 32-tile coverage, WarpBitTranspose
       -> active-tile mask and CTA work queue
       -> per-active-tile partial FP16 RGBT
  -> one warp per fine tile post-blends ordered partials -> FP16 RGB,T framebuffer
  -> wrapper extracts RGB, converts alpha=1-T and casts results to FP32.
```

The detailed graph, per-stage layout, precision, synchronization, and source anchors are in [official_raster_map.json](../../artifacts/higs-h3-fwd-1b-design/official_raster_map.json) and [execution_graph.json](../../artifacts/higs-h3-fwd-1b-design/execution_graph.json).

## 2. Inference-only assumptions to remove

Official raster inputs and intermediate RGBT are FP16. It performs half2 Gaussian evaluation and approximate `exp2`, recomputes a visibility mask from the quantized raster representation, emits a sparse FP16 partial buffer, then performs a second FP16 post-blend. Its wrapper destroys temporary scene state and exposes RGB/alpha only; it does not retain `last_ids`, per-pixel termination, or a backward replay contract. Scene-owned high-water-mark buffers assume a static, prepacked inference scene and reuse capacity across frames.

H3-FWD-1B keeps hierarchy-only concepts (8x4 macro, 1024-G batches, 32-G mask words, sorted depth order) but replaces arithmetic/storage semantics with B2 FP32 training semantics. The complete KEEP/CHANGE/REMOVE/ADD classification is in [fp32_training_delta.json](../../artifacts/higs-h3-fwd-1b-design/fp32_training_delta.json).

## 3. FP32 training raster contract

Inputs are FP32 `means2d[N,2]`, `conics[N,3]`, `colors[N,3]`, and `opacities[N]`; exact macro state is `macro_offsets`, `macro_sorted_ids`, sorted-aligned `uint32 fine_tile_masks`, and `macro_batch_offsets`. Outputs are FP32 RGB, FP32 alpha (with final transmittance retained or derivable), and int32 `last_ids`.

For each selected Gaussian and pixel, use B2's operation order:

```text
delta = mean2d - (pixel + 0.5)
sigma = 0.5*(A*dx*dx + C*dy*dy) + B*dx*dy
alpha = min(MAX_ALPHA, opacity * exp(-sigma))
reject if sigma < 0 or alpha < ALPHA_THRESHOLD
next_T = T*(1-alpha)
terminate this pixel without compositing the current Gaussian if next_T <= 1e-4
otherwise RGB += color*(alpha*T); T = next_T
final RGB += T*background; alpha_out = 1-T.
```

`ALPHA_THRESHOLD=1/255`, `MAX_ALPHA=0.999` for the B2 reference, the strict comparisons, background behavior, and sorted traversal are immutable compatibility requirements. A macro mask controls only whether a Gaussian belongs to a fine tile; it cannot replace the per-pixel Gaussian test.

## 4. Exact `last_ids` design

For a fine tile whose local macro bit is `ell`, define `m[j]` as the sorted-aligned 32-bit mask of macro-entry `j`. The implicit fine-tile list is the stable filter:

```text
E_ell = [ j | (m[j] & (1u << ell)) != 0 ], in increasing j
rank_ell(j) = popc( m[0..j] restricted to bit ell ) - 1.
```

The raster initializes `tile_local_position=0`, increments it exactly once for every masked entry before pixel-specific alpha evaluation, and uses its prior value as `rank_ell(j)`. For each pixel it writes that rank only after the Gaussian is valid **and** `next_T > 1e-4`; the threshold-crossing Gaussian terminates the pixel without updating `last_ids`, matching B2 forward control flow. The empty/no-contribution sentinel must be copied from the B2 adapter's initialized value (the reference initializes `cur_idx` to zero).

If the locally compiled B2 backward consumes an absolute flattened index rather than the documented tile-local form, the adapter conversion is `tile_offsets[tile] + rank_ell(j)`. The prototype validation will assert which form its actual B2 entrypoint uses; the hierarchy itself retains enough information for either representation. See [last_id_design.json](../../artifacts/higs-h3-fwd-1b-design/last_id_design.json).

## 5. F4+F5 co-design opportunities

The promising boundary is to make exact mask production part of macro materialization and make F5 consume that stored mask directly. In particular, the current repaired F4 recomputes exact coverage in COUNT and FILL, then separately regenerates masks after sorting. During FILL it already has the exact 8x4 mask at the point it writes the macro entry. Carrying that mask with the sort payload removes the third coverage pass and establishes the F5 input contract without a conventional fine-list materialization.

The strongest candidate for the subsequent raster prototype is an FP32 direct macro consumer: retain sorted IDs and masks, schedule only active fine tiles per macro batch, and compose in exact B2 order while maintaining rank. Whether its partial state should be post-blended or held through all batches is a break-even decision, not an assumption. The candidate comparison, memory bounds, and risks are in [f4_f5_codesign_candidates.json](../../artifacts/higs-h3-fwd-1b-design/f4_f5_codesign_candidates.json).

## 6. Can duplicate coverage enumeration be removed?

Yes in principle, but not safely by a simple one-pass append. The existing count->scan->fill establishes exact compact offsets and therefore repeats geometric enumeration. The immediate, low-risk removal is only the **third** mask-generation enumeration: emit the exact mask in FILL and transport it through the segmented sort. Removing COUNT+FILL duplication needs either a bounded capacity/overflow protocol, a global temporary hit stream plus later compacting, or paged per-macro queues. All preserve geometry only when their overflow/fallback paths are exact. None is approved for implementation before the break-even target is known.

## 7. Backward retained state

`macro_offsets`, `macro_sorted_ids`, sorted-aligned masks, `macro_batch_offsets`, final per-pixel T/alpha, and `last_ids` are required now. Their storage makes a native backward possible without reconstructing exact coverage. Per-pixel termination rank and per-pixel/per-batch prefix T are cheap optional replay accelerators; they are not required for the first forward-only prototype. Full per-Gaussian alpha, pixel-Gaussian visibility, or a conventional fine list are deferred: retaining them would erase the representation benefit. The contract appears in [backward_state_contract.json](../../artifacts/higs-h3-fwd-1b-design/backward_state_contract.json).

## 8. Correctness and performance gates

The A/B oracle compares RGB, alpha/T, last IDs, active pixels, termination, ordering, and NaN/Inf on the same B2 F0-F3 tensors. It reports max/mean absolute error, relative L2, support and last-ID disagreements. Equal-depth ordering is separately bucketed and only semantic non-tie inversions fail order validation.

CUDA events must place only GPU work in the measured region; NVTX splits coverage, metadata, masks, load, transpose, evaluation, compositing, termination, and post-blend. No Python reconstruction belongs inside these intervals. See [correctness_oracle.json](../../artifacts/higs-h3-fwd-1b-design/correctness_oracle.json) and [instrumentation_plan.json](../../artifacts/higs-h3-fwd-1b-design/instrumentation_plan.json).

## 9. Recommended first implementation, pending oracle

1. Do not write the production macro raster yet.
2. Extend the exact F4 representation so FILL writes a `uint32` exact mask and segmented sorting carries it in sorted-ID order; this only removes redundant mask regeneration and does not change coverage.
3. Implement a validation-only FP32 raster consumer of those retained arrays, with B2 formula/order, rank state, RGB+alpha+last IDs, and full instrumentation. Do not add FP16 paths, backward, capacity speculation, or producer-consumer fusion.
4. Consume DSH's per-scene Macro-F5 break-even targets. Promote only if the measured co-designed F4+F5 meets every scene target and the correctness oracle passes.

The remaining blockers are the pending DSH break-even evidence, the actual B2 entrypoint's stored-ID coordinate assertion (local versus absolute compatibility adapter), and a measured choice between direct final composition and sparse partial/post-blend. No final performance claim is made here.
