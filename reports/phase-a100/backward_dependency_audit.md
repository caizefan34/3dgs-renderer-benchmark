# Corrected A100 Baseline: Backward Dependency Audit

**Scope.** Read-only audit of the corrected gsplat-style A100 baseline's forward-to-backward data dependencies. No renderer, CUDA kernel, or optimization-path source was modified.

**Source basis.** The canonical installed gsplat Python autograd wrapper is `C:\\Users\\36570\\miniconda3\\Lib\\site-packages\\gsplat\\cuda\\_wrapper.py`; its standard 3DGS rasterizer autograd function is `_RasterizeToPixels` (lines 1251-1378). The archival CUDA/C++ baseline sources are `patches/gsplat_orig/`; the corrected intersection implementation is `patches/IntersectTile.c1.cu`. `temp_rasterization_src.py` is a checked-in wrapper snapshot used to trace orchestration. The HiGS patch is cited separately only where its native backward capture makes the same saved-state contract explicit.

**Canonical autograd confirmation.** `_RasterizeToPixels.forward()` calls `rasterize_to_pixels_3dgs_fwd` and saves `means2d`, `conics`, `colors`, `opacities`, backgrounds/masks, `isect_offsets`, `flatten_ids`, `render_alphas`, and `last_ids` (`_wrapper.py:1251-1305`). `_RasterizeToPixels.backward()` retrieves that exact tuple and passes it to `rasterize_to_pixels_3dgs_bwd` (`_wrapper.py:1307-1353`). The same function supports both packed (`[nnz,...]`) and unpacked (`[...,N,...]`) attribute layouts, as documented in its argument annotations. This corrects the archival-tree limitation: `patches/gsplat_orig/` contains the CUDA/C++ implementation but not the canonical Python wrapper.

## Executive conclusion

Backward rasterization cannot reconstruct its per-tile, depth-ordered Gaussian work list solely from the per-Gaussian attributes. It needs:

1. `tile_offsets`, which maps every `(image, tile)` to a contiguous interval in the sorted intersection list;
2. `flatten_ids`, whose ordered entries identify the Gaussian attributes used at each list position; and
3. the forward output state `render_alphas` and `last_ids`, which permits the reverse compositing recurrence and preserves each pixel's early-termination boundary.

`isect_ids` is required while producing `tile_offsets`, but after offset encoding it is not passed to the rasterizer backward. `depths`, `radii`, and `tiles_per_gauss` are likewise not rasterizer-backward inputs. The standard rasterizer backward reuses saved 2D attributes rather than recomputing them; later projection backward performs the 3D VJP/recomputation needed to reach master parameters.

## 1. Forward provenance and lifetime

| Item | Produced by | Forward consumer | Rasterization backward consumer | Required persistence |
|---|---|---|---|---|
| `means2d`, `conics` | fused projection | rasterizer | attribute lookup through `flatten_ids` | saved |
| `colors`, `opacities` | color/SH and opacity preparation | rasterizer | attribute lookup through `flatten_ids` | saved |
| `radii`, `depths` | fused projection | intersection generation | not read by rasterizer backward | not needed after intersection/rasterization setup |
| `tiles_per_gauss`, cumulative counts | first intersection pass | allocation/second intersection pass | not read | temporary |
| `isect_ids` | second intersection pass, then radix sort | offset encoding | not read after offset encoding | temporary before backward |
| `tile_offsets` | `isect_offset_encode(isect_ids, ...)` | tile interval lookup | same tile interval lookup | saved |
| `flatten_ids` | second intersection pass and paired radix sort | ordered Gaussian lookup | same ordered Gaussian lookup | saved |
| `render_alphas`, `last_ids` | forward rasterizer | output/state | reverse compositing and stopping boundary | saved |

Wrapper evidence: `temp_rasterization_src.py:599-617` constructs intersections and offsets, then lines 671-684 / 714-727 pass `means2d`, `conics`, `colors`, `opacities`, `isect_offsets`, and `flatten_ids` to rasterization. The corrected A100 profiling harness uses the same low-level sequence at `@corrected_baseline_cuda_profile.py:181-217`.

## 2. Intersection and sorting dependency

`patches/gsplat_orig/Intersect.cpp:56-115` runs a counting pass, computes an exclusive-placement source (`cum_tiles_per_gauss`), allocates `isect_ids` (`int64`) and `flatten_ids` (`int32`), then emits one pair per Gaussian/tile intersection. Its `sort=True` branch sorts those key/value pairs together and returns the sorted pair at lines 118-148.

The corrected C1 implementation makes the pair semantics explicit:

- `patches/IntersectTile.c1.cu:105-116`: every intersected tile receives a key in `isect_ids[cur_idx]` and `flatten_ids[cur_idx] = idx`.
- `patches/IntersectTile.c1.cu:300-345`: CUB `SortPairs` applies exactly the same permutation to keys and values.
- `patches/IntersectTile.c1.cu:214-263`: offset encoding removes the low depth bits from the sorted key, detects image/tile transitions, and writes the start index for each tile.

Therefore sorting is not merely an allocation detail: the resulting position order of `flatten_ids` is the depth order within the ranges represented by `tile_offsets`. Replacing or regenerating either independently would break that relationship.

## 3. Exact baseline backward reads

The standard backward host API accepts the following at `patches/gsplat_orig/Rasterization.cpp:118-194`: 2D attributes, optional background/mask, `tile_offsets`, `flatten_ids`, `render_alphas`, `last_ids`, and upstream output gradients. It allocates gradient arrays and launches the CUDA kernel.

The CUDA kernel demonstrates the irreducible reconstruction inputs:

- `RasterizeToPixels3DGSBwd.cu:88-95` obtains each tile range from `tile_offsets`; final range end is `n_isects`.
- Lines 106-120 restore final transmittance from `render_alphas`, pixel cutoff from `last_ids`, and upstream gradients.
- Lines 137-151 traverse the range in reverse, read `g = flatten_ids[idx]`, and load `means2d[g]`, `conics[g]`, `opacities[g]`, and `colors[g * CDIM]` into shared memory.
- Lines 156-275 execute reverse compositing and atomically accumulate gradients into the attribute-gradient rows indexed by the same `g`.

Forward uses the same pair of topology structures: `RasterizeToPixels3DGSFwd.cu:85-89` determines the range from offsets and `:122-133` loads `flatten_ids[idx]`; it records `render_alphas` and `last_ids` at `:172-187`.

The audited native HiGS implementation agrees: `patches/higs-differentiable.patch:2712-2724` captures flattened 2D attributes, offsets, IDs, alphas, last IDs, radii, and optional depth accumulation; `:2756-2765` retrieves the saved tuple; `:2849-2885` passes the requisite subset to native backward. Its CUDA blend kernel uses `tile_offsets` at `:4112-4118`, `flatten_ids` at `:4162-4179`, and emits atomics indexed by the recovered ID at `:4250-4269`.

## 4. Saved-state findings

### Standard gsplat rasterizer

The concrete archived C++ rasterizer API proves that its backward receives all rasterization data directly; the wrapper-level `save_for_backward` implementation is not included in `patches/gsplat_orig`. The standard wrapper snapshot and the existing source-audit evidence identify the intended autograd contract: persist `means2d`, `conics`, `colors`, `opacities`, backgrounds/masks, `isect_offsets`, `flatten_ids`, `render_alphas`, and `last_ids`.

### Native patched path

The patch explicitly invokes `ctx.save_for_backward` at `patches/higs-differentiable.patch:2473-2482`. The capture tuple (lines 2712-2724) preserves:

`means2d_f`, `conics_f`, `colors_eval_f`, `opacities_f`, `tile_offsets_f`, `flatten_ids_f`, `render_alphas_f`, `last_ids_f`, `radii_f`, and optional `depth_acc`, in addition to master tensors, camera data, visibility IDs, and visible master subsets.

This establishes a precise distinction:

- **Rasterization backward:** reuses saved forward state; no intersection rebuild and no 2D attribute recomputation.
- **Projection/parameter backward:** consumes generated 2D gradients and performs a projection VJP. The native patch's projection kernel recomputes covariance from quaternion/scale at `:4631-4640`; the standard fused projection backward similarly takes original parameters, `radii`, `conics`, and 2D upstream gradients (`patches/gsplat_orig/Projection.cpp:191-278`).

## 5. Reconstruction answers

| Question | Answer | Evidence |
|---|---|---|
| Can backward recover tile membership from attributes alone? | No, not without rerunning intersection/binning. | Membership is materialized as `tile_offsets` plus its `flatten_ids` interval; Bwd CUDA directly reads both. |
| Can it use unsorted `flatten_ids`? | No. | Pair sorting supplies depth order; backward reverses the tile range to undo front-to-back compositing. |
| Is `isect_ids` needed during backward? | No after offset encoding. | It is input to offset encoding, while rasterization Bwd API only receives offsets and flatten IDs. |
| Are `radii`, depths, and count/cumsum needed by rasterization backward? | No. | They are consumed during intersection or projection stages, absent from `Rasterization.cpp` backward signature. |
| Can alphas/last IDs be discarded? | No for this implementation. | Backward initializes `T_final` from alpha and bounds reverse traversal using `last_ids`. |
| Are saved 2D attributes recomputed in rasterization backward? | No. | CUDA Bwd loads saved arrays by `g`; recomputation begins only in the projection VJP stage. |

## 6. Correctness validation

Two existing minimal CUDA backward tests were executed:

```text
python -m pytest tests/test_higs_native_backward.py::TestNativeBackwardGradients::test_all_params_gradients_nonzero_finite -q -rs
python -m pytest tests/test_higs_trainable.py::TestGradientFlow::test_gradients_exist_for_all_params -q -rs
```

Both commands exited successfully but were skipped because the HiGS CUDA backend is unavailable in this environment. The first test is a 64-Gaussian gradient existence/finite-value test (`tests/test_higs_native_backward.py:287-309`); the native suite also contains finite-difference and `gradcheck` coverage (`:337-406`). The trainable test checks finite non-zero gradients for all five parameter classes (`tests/test_higs_trainable.py:138-172`).

**Validation status: NOT EXECUTED (environment skip), not a pass.** No numerical backward-correctness claim is made from this host.

## 7. Audit boundaries and result

- This is a source/dataflow audit, not a performance claim.
- The working tree already contained unrelated modifications/untracked artifacts; this task did not alter them.
- Only the requested report artifacts were created.
- No optimization recommendation is implemented by this audit.
