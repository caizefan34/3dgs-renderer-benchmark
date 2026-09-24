# R6-A — Block-Level Gradient Aggregation Minimal CUDA Prototype

## 1. Baseline identity

The frozen comparison is B1A: gsplat v1.5.3 commit
`937e29912570c372bed6747a5c9bf85fed877bae` plus the true-AccuTile v1.5.3
port (SHA-256 `33292a08ebb74437b5108fcf9282b01ac9621d45495bbba219f8b41000c803f1`).
The declared repository freeze is `02375033388d4348376b6b607ab85f551e498a77`.
All CUDA work used an independent copy of
`/mnt/storage_pool/liaoyuanjun/gsplat-true-accutile-v153`; that B1A source was
not modified.

Hardware was A100-PCIE-40GB (SM 8.0), PyTorch 2.4.1+cu124, CUDA/nvcc
12.4/12.4.131, gcc 11.4.0.

## 2. Control-flow audit

See [r6a-control-flow-audit.md](r6a-control-flow-audit.md).  In short, a
16x16 tile is one 256-thread block with eight warps. `id_batch[t]` is block
shared, but `bin_final`, `T`, `buffer[]`, `valid`, `warp_bin_final`, and hence
`first_t=max(0,batch_end-warp_bin_final)` are lane/warp specific.  A block
barrier in the original loop would be unsafe because warps can be at different
`t` values or have taken `continue`.

## 3. Implementation

R6-A uses Design A, a uniform block Gaussian loop.  Every non-masked block
executes `t=0..batch_size-1`; lanes retain the original valid predicate, so
skipped work contributes zero and leaves `T`/`buffer[]` unchanged. Warp leaders
write RGB, conic, means2d, abs-means2d, opacity, and an active flag into
shared memory. After a block barrier, warp 0 reduces the eight rows and lane 0
does the original global writes only if some warp was active. A second barrier
protects the partial storage before the next `t`.

The default B1A remains untouched. `B1A_R6A` is a separately generated source
tree, and `--count-atomics` creates a separate diagnostic build with its
counter enabled. The timing build has no counter increment.

## 4. Correctness

Room/camera-0 compared exact rasterizer inputs and VJP outputs. RGB, alpha,
and intersection count were bit-identical. All output shapes/dtypes matched,
and R6-A had zero NaN/Inf. The relative-L2 values were `6.41e-6` (colors),
`4.68e-6` (conics), `3.28e-6` (means2d), `8.09e-6` (means2d_abs), and
`4.51e-6` (opacity): all pass the `1e-4` gate. Full detailed values are in
`r6a-minimal-cuda-results.json`.

## 5. Atomic-count reduction

The counts below are direct device counters in the actual backward kernels.
They count writer events; with RGB=3 and absgrad, each event issues 11 global
`gpuAtomicAdd` calls.

| Scene | B1A writer events | R6-A writer events | R6-A block reductions | Realized reduction |
|---|---:|---:|---:|---:|
| room | 5,475,862 | 687,077 | 14,260,555 | 7.970x |
| bicycle | 16,910,211 | 2,142,988 | 12,127,849 | 7.891x |
| garden | 34,693,503 | 4,590,302 | 9,400,526 | 7.558x |

The third column is also direct instrumentation: one counter increment by the
block reducer for each uniform `t` iteration. It shows the key replacement
cost—R6-A executes a block-local reduction for every flattened intersection,
including iterations with no final global writer.

Thus the previously measured approximately 7.5x opportunity is realizable as
global atomic traffic: R6-A reduces global atomic calls by the same factors.
The direct 7.56–7.97x range is slightly above the earlier approximately-7.5x
footprint/proxy estimate; it supersedes that estimator for these exact
camera-0 workloads rather than representing a speedup prediction.

## 6. CUDA resource usage

`cuobjdump --dump-resource-usage` on the CDIM=3 `sm_80` kernel reported 70
registers/thread for B1A and 68 for R6-A; both have zero static shared and
zero local/spill memory. Dynamic shared grows from 10,240 B to 10,624 B
(eight rows x 12 floats = +384 B). With 256 threads/block, registers limit
both to three blocks/SM, or 37.5% theoretical occupancy. Therefore occupancy
is not the observed regression mechanism.

## 7. Barrier and control-flow overhead

B1A has two legal block barriers per Gaussian batch (before shared overwrite,
after shared load). R6-A has those two plus **two barriers per Gaussian**:
one after partial writes, one after warp-0 reduction/global write. It also
removes every warp's `first_t` skip. This is the dominant cost of Design A.

## 8. Microbenchmark

Warmup=20, measure=100, camera 0, true AccuTile enabled. Numbers are mean GPU
event milliseconds; total is forward plus backward.

| Scene | B1A bwd | R6-A bwd | bwd speedup | B1A total | R6-A total | total speedup |
|---|---:|---:|---:|---:|---:|---:|
| room | 6.313 | 74.571 | -91.53% | 11.952 | 80.186 | -85.09% |
| bicycle | 15.903 | 68.712 | -76.86% | 25.132 | 77.784 | -67.69% |
| garden | 24.772 | 56.151 | -55.88% | 35.659 | 66.723 | -46.56% |

Forward is unchanged source code; small measured forward differences are normal
independent-run timing noise.

## 9. Short-training semantics

The required room run used 778x519, seed 42, 800 iterations, and densification
at 500/600/700. It passed the semantic gate: maximum loss delta was
`3.46e-4`, maximum PSNR delta was `0.0129 dB`, and final Gaussian counts were
1,582,849 (B1A) and 1,582,837 (R6-A). At the three densification steps, clone
was zero for both; splits/prunes differed only at FP-level trajectory noise.

## 10. Mechanism interpretation

The global traffic reduction is real (7.56–7.97x) but it is not useful in this
minimal mapping. For every candidate Gaussian, eight warps must now execute a
uniform loop, write shared partials, and wait twice. The added synchronization
and reintroduced inactive-loop work outweigh eliminated global atomics by a
large margin. This is not evidence that atomics are hidden; it is evidence
that this exact uniform-barrier implementation has an unacceptable replacement
cost.

## 11. Gate

**DROP.** Gradients and training semantics pass and atomic traffic falls, but
backward regresses on all three scenes (55.9–91.5%). This fails the performance
gate and leaves no credible tuning change within the required minimal Design-A
scope.

## 12. Limitations

This rejects Design A only. A no-global-write sink oracle was not run because
the full candidate already has a 2.27–11.82x backward slowdown, decisively
attributable to its required uniform loop/barriers. No HiGS, cross-block,
approximation, pruning, precision, or AccuTile changes were made. This report
makes no novelty claim: local gradient aggregation is prior-art-calibrated
baseline engineering, not R6-H.

## 13. Exact source paths

- Generator: `experiments/r6/r6_a/prepare_r6a_blockagg_source.py`
- Direct rasterizer parity test: `experiments/r6/r6_a/r6a_rasterizer_correctness.py`
- B1A source: `/mnt/storage_pool/liaoyuanjun/gsplat-true-accutile-v153/gsplat/cuda/csrc/RasterizeToPixels3DGSBwd.cu`
- B1A_R6A timing source: `/mnt/storage_pool/liaoyuanjun/gsplat-r6a-blockagg-v153/gsplat/cuda/csrc/RasterizeToPixels3DGSBwd.cu`
- Diagnostic source: `/mnt/storage_pool/liaoyuanjun/gsplat-r6a-blockagg-count-v153/gsplat/cuda/csrc/RasterizeToPixels3DGSBwd.cu`
- Raw A100 outputs: `/mnt/storage_pool/liaoyuanjun/r6a_results/`

## 14. Git diff

The workspace was already an all-untracked research worktree before this task;
therefore `git diff --stat` and `git diff -- <CUDA files>` are empty for these
new untracked files. `git status --short` lists the new R6-A paths together
with pre-existing untracked material. No commit was made.
