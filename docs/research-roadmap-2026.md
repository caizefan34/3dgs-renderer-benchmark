# 3DGS renderer and compression research roadmap (2026)

This document is the decision record for turning this repository into a
benchmark and research platform. It separates measured facts from external
claims: **A** is measured on EPIC-05, **B** is reproduced outside the authority
host, and **C** is paper or public-project evidence. A C claim is never copied
into `results/measured/` or a ranking.

## 1. Current architecture and gaps

The repository has a clean measurement boundary: canonical PLY and camera
manifests enter renderer adapters, the speed and quality collectors emit a
versioned result contract, and `benchmark_matrix.py` validates immutable cohort
identity before aggregation. The leaderboard keeps measured, reproduced, and
paper evidence separate and rejects incomplete rows. This is the right core
for a research platform.

Strengths are pinned source commits, hashed datasets/cameras/protocols,
paired GT quality, warmup/repeat statistics, P95/P99 latency, NVML memory,
failure retention, and isolated renderer environments. The main gaps are
training-track coverage, temporal sequences, compression artifacts, EGL and
WebGPU paths, and a public experiment registry for ablations. The current
automatic adapters cover original 3DGS, gsplat, HiGS modes, Speedy-Splat, and
TC-GS. FlashGS, Local-GS, GEMM-GS, StopThePop, and fast-gaussian-rasterization
must remain separate rows or blocked until their representation and display
requirements are explicit.

## 2. EPIC-05 measured baseline (Tier A)

The complete 25-row baseline is summarized in
[`reports/epic05-tier-a-baseline-2026-07-20.md`](../reports/epic05-tier-a-baseline-2026-07-20.md).
The useful decision is not a single winner: HiGS is the throughput frontier,
Speedy-Splat is the near-lossless speed/memory point, and original 3DGS remains
the quality and API reference.

## 3. Renderer capability matrix

The following is an integration matrix. FPS potential is a hypothesis until a
row enters the common suite.

| Renderer | Strategy | Visibility / sort | Memory | Common row | FPS potential | Effort |
| --- | --- | --- | --- | --- | --- | --- |
| Original 3DGS | one-pass tile rasterization | global duplicate keys, depth sort | full FP32 intermediates | yes, reference | low | done |
| gsplat packed | packed visible pairs | radix sort of packed keys | lower intermediates | yes | medium | done |
| gsplat dense | dense pair traversal | dense tile ordering | higher intermediates | yes, config row | medium | done |
| gsplat HiGS | fused macro-tile intersection and fine-tile blend | segmented sort per macro tile | persistent FP16 workspace | yes, inference track | very high | done |
| Speedy-Splat | localized exact rasterization and pruning path | localized tile work | compact buffers | yes, same-checkpoint row | high | done |
| TC-GS | FP16 alpha GEMM/Tensor Cores | inherited localized ordering | compact FP16 path | yes, precision config | high | done |
| Fast Gaussian Rasterization | CUDA sort plus OpenGL raster | global CUDA sort | framebuffer plus GL | separate EGL track | high | medium |
| FlashGS | redundancy elimination and pipeline scheduling | hierarchical scheduling | compressed/reordered | adapter required | high | high |
| GEMM-GS | matrix blending and double buffering | tile grouping | Tensor Core tiles | adapter required | high | high |
| Local-GS / TiCoGS | tile-local warp-coherent blend | local ordering | shared tile state | adapter required | medium-high | high |
| StopThePop | view-consistent hierarchical sorting | hierarchical visibility | model-dependent | retrained/native track | medium-high | high |
| TensorGS | tensorized appearance/render path | method-specific | method-specific | paper/native track | unknown | high |
| Mip-Splatting | anti-aliasing and scale filtering | standard sort | extra filtering state | quality track | medium | medium |
| Octree / voxel-hierarchy | spatial hierarchy and culling | node traversal then local sort | hierarchy overhead | native track | scene-dependent | high |

Compatibility rule: a renderer that changes pruning, training, SH degree,
quantization, or representation gets a distinct `config_id`; it cannot quietly
compete with the common checkpoint row.

## 4. HiGS reverse engineering

The pinned implementation in the gsplat source uses a persistent
`InferenceRenderState`. It packs means as `[3,N]`, quaternion/scale/opacity as
`[N,8]` half, and colors as SH or pre-activated RGB. Projection writes reusable
2-D means, depths, conics, and a bitset of visible Gaussians.

The intersection path groups work into fused macro tiles. Each macro tile first
enumerates Gaussian intersections, then performs a segmented depth sort. The
raster kernel loads 32-Gaussian mini-batches, computes a 32-bit overlap mask for
the fine tiles, transposes masks with warp shuffles, and queues only active
tiles. Colors are loaded after visibility is known. A work-stealing warp queue
blends active fine tiles in shared memory, and a post-blend kernel composites
macro-tile batches front to back with transmittance early-out. Tile 8 and tile
16 are compile-time paths; SH32/SH16 compression is an eager scene-time codec.

This explains the Tier A result: hierarchy and persistent buffers remove global
sort and overdraw work, but creation performs allocations, optional SH codec
precomputation, a synchronization, and hierarchy-sized workspace allocation.

### Why the path is inference-oriented

1. Densification changes `N` and invalidates every packed index, offset, mask,
   and workspace allocation.
2. Pruning changes the same identity and requires a full rebuild; incremental
   edits are not represented by the scene object.
3. The cached projection, intersection, and sort buffers are forward-only;
   they do not retain the per-Gaussian intermediates and ordering needed for a
   backward pass.
4. Fused visibility masks and half precision accumulators are not exposed as
   differentiable PyTorch operations. Autograd would need custom backward
   kernels and deterministic saved state.
5. The scene constructor explicitly synchronizes after SH compression, and the
   state is documented as single-stream and not thread-safe.

Therefore HiGS is a strong inference renderer, not evidence of faster training.
The correct benchmark extension is a separate training track with explicit
rebuild and backward costs.

## 5. Trainable HiGS variants

| Variant | Algorithm | Complexity / VRAM | Expected gain | Effort / novelty |
| --- | --- | --- | --- | --- |
| Lazy-Rebuild | keep a stable superset; rebuild only every K optimizer steps | O(N) stale mask, +5-15% VRAM | 1.1-1.5x training | medium / medium |
| Differentiable | save compact visibility and tile order; custom backward blend | O(intersections), +20-60% VRAM | 1.2-2.0x if backward dominates | very high / high |
| Late-Stage | use standard gsplat until densification ends, then switch HiGS | one rebuild, low extra VRAM | 1.3-2.0x late iterations | low / low |
| Incremental Hierarchy | local insert/delete updates to macro-tile offsets | O(changed Gaussians), fragmentation risk | 1.2-1.8x | high / high |
| Train-Time Sparse | visibility-driven sparse optimizer with periodic exact refresh | sparse state, quality risk | 1.5-3.0x | high / high |

The first experiment should be Late-Stage HiGS because it is easy to validate
and cannot corrupt early densification. Each variant needs loss parity,
checkpoint hashes, iteration-time breakdown, rebuild frequency, peak VRAM, and
final PSNR/SSIM/LPIPS. No variant should be promoted until it beats standard
gsplat on a fixed training budget and final quality gate.

## 6. Acceleration taxonomy

| Category | Principle | Strength | Weakness | Benchmark metric |
| --- | --- | --- | --- | --- |
| Visibility | frustum, radius, hierarchical, learned culling | removes work early | false negatives damage quality | visible count, recall, FPS |
| Sorting | radix, segmented, local, approximate order | attacks long-tail latency | order errors create artifacts | sort time, P99, error |
| Rasterization | warp-coherent, fused, early-out | higher arithmetic throughput | register/shared-memory pressure | kernel time, occupancy |
| Memory | packing, half, reuse, compression | less bandwidth and VRAM | decode and precision cost | peak VRAM, bandwidth |
| Tensor Core | FP16/INT8 GEMM and alpha evaluation | high throughput | architecture and quality constraints | FPS, PSNR delta |
| Temporal | reuse visible sets/reprojection | amortizes stable frames | camera cuts and disocclusion | sequence FPS, temporal error |
| Hybrid | combine hierarchy, local blend, and temporal cache | multiplicative potential | complex failure modes | Pareto FPS/quality/VRAM |

## 7. HiGS fusion opportunities

Rank is ROI for this repository, not a paper claim. FPS deltas are target
ranges for an ablation, not measured results.

| Rank | Combination | Effort | Compatibility | Target FPS | Quality / VRAM risk | Why |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | HiGS + calibrated tile selector | low | direct | +5-20% | low / neutral | replaces count heuristic with occupancy calibration |
| 2 | HiGS + visibility cache | medium | direct inference | +10-35% sequences | stale-set artifacts / +5% | reuses stable macro-tile masks |
| 3 | HiGS + Local-GS blend | high | kernel-level | +10-30% | low / +5-15% | reduces warp divergence in active tiles |
| 4 | HiGS + dynamic SH degree | medium | direct | +5-20% | view-dependent / lower | spend SH only where it affects pixels |
| 5 | HiGS + FlashGS scheduling | high | representation audit | +10-30% | mapping risk | pipeline and redundancy removal |
| 6 | HiGS + GEMM-GS | high | alpha path only | +15-40% | FP16 quality / workspace | complementary blend acceleration |
| 7 | HiGS + TC-GS | high | precision contract required | +10-35% | precision / +10% | Tensor Core alpha with hierarchical visibility |
| 8 | HiGS + temporal reprojection | high | new sequence track | +30-3x | ghosting / cache | amortizes unchanged pixels |
| 9 | HiGS + learned visibility | very high | new model track | +20-50% | false negatives / model VRAM | predicts work from camera and scene |
| 10 | HiGS + StopThePop | very high | retrained track | unknown | quality and format risk | view-consistent ordering may conflict with cache |

Easiest win is the tile selector; most publishable is incremental/differentiable
HiGS; highest risk/reward is learned visibility plus temporal reprojection.

## 8. Thirty renderer research ideas

Every item is a hypothesis. Feasibility is an engineering estimate; expected
speedup is versus the matching baseline and must be measured.

| # | Idea | Feasibility | Speedup | Complexity | Publishability |
| ---: | --- | --- | ---: | --- | --- |
| 1 | Temporal visibility cache | high for sequences | 1.3-3x | medium | high |
| 2 | Learned Gaussian scheduler | medium | 1.2-2x | high | high |
| 3 | Learned tile scheduler | medium | 1.1-1.8x | high | high |
| 4 | Neural visibility predictor | medium | 1.3-2x | very high | high |
| 5 | Dynamic SH evaluation | high | 1.1-1.5x | medium | medium |
| 6 | Camera-aware Gaussian clustering | high | 1.1-1.6x | medium | medium |
| 7 | Hierarchical SH compression | high | 1.1-1.4x | medium | high |
| 8 | Reprojection-assisted rendering | high | 1.5-3x | high | high |
| 9 | Persistent visible-set cache | high | 1.2-2x | medium | medium |
| 10 | Tensor-Core-centric renderer | medium | 1.3-2x | high | high |
| 11 | Macro-tile occupancy autotuner | high | 1.05-1.3x | low | medium |
| 12 | P99-aware work stealing | high | 1.1-1.5x tail | medium | medium |
| 13 | Asynchronous hierarchy rebuild | medium | 1.1-1.4x | high | high |
| 14 | Multi-resolution Gaussian cache | medium | 1.2-2x | high | high |
| 15 | Foveated Gaussian budgets | medium | 1.3-2x | medium | high |
| 16 | View-space error culling | high | 1.1-1.5x | medium | medium |
| 17 | Opacity-aware sort truncation | high | 1.1-1.4x | low | medium |
| 18 | Warp-specialized SH decode | high | 1.05-1.3x | medium | medium |
| 19 | CUDA graph camera batches | high | 1.05-1.2x | low | low |
| 20 | Cooperative-group segmented sort | medium | 1.1-1.4x | high | medium |
| 21 | Tile-local quantized conics | medium | 1.1-1.4x | medium | high |
| 22 | Learned radius clipping | medium | 1.1-1.5x | high | high |
| 23 | Surface-aware Gaussian ordering | medium | 1.1-1.4x | high | high |
| 24 | Persistent multi-camera batching | high | 1.2-1.8x | medium | medium |
| 25 | Render-time adaptive background | high | 1.02-1.1x | low | low |
| 26 | Visibility confidence maps | medium | 1.1-1.4x | medium | high |
| 27 | Hybrid exact/approximate sorting | high | 1.1-1.6x | medium | high |
| 28 | Shared scene preprocessing service | high | lower startup | medium | medium |
| 29 | Kernel autotuning by Gaussian density | high | 1.05-1.3x | medium | medium |
| 30 | Multi-GPU macro-tile partitioning | medium | 1.5-3x | very high | high |

Top ten easiest wins are 5, 11, 12, 16, 17, 18, 19, 24, 25, and 29. Top ten
most publishable are 1, 2, 3, 4, 7, 8, 10, 13, 15, and 22. Top ten
highest-risk/highest-reward are 2, 4, 8, 10, 13, 14, 15, 22, 23, and 30.

## 9. Compression survey

The common benchmark must measure a compressed artifact as a separate track:
compressed bytes, decode time, peak decode VRAM, render FPS, PSNR/SSIM/LPIPS,
and whether retraining or a special renderer is required. A near-lossless gate
is PSNR drop below 0.2 dB plus a declared LPIPS threshold and visual audit.

The first artifact-layer Tier A measurements are published in
[`reports/epic05-compression-artifact-encoding-2026-07-23.md`](../reports/epic05-compression-artifact-encoding-2026-07-23.md).
Across all five canonical checkpoints, block-float measured 2.170x aggregate
compression and tile-codebook measured 3.840x. These rows are deliberately
`artifact_ready`: decoded-render FPS, quality deltas, and visual audit are still
required before either codec can be called near-lossless.

| Method / format | Main idea | Typical evidence | Rendering compatibility | Track |
| --- | --- | --- | --- | --- |
| PLY | raw float attributes | C baseline | universal | common reference |
| SPLAT | viewer-oriented binary packing | C/B | broad viewers | format row |
| compressed SPLAT | quantized packed attributes | C/B | viewer-dependent | format row |
| LightGaussian | pruning plus quantization/distillation | C | often retrained | native track |
| Compact3DGS | compact representation and pruning | C | method-specific | native track |
| HAC | hash-grid / entropy-style learned coding | C | decoder required | compression track |
| HAC++ | improved learned anchors and entropy coding | C | decoder required | compression track |
| VecTree Quantization | tree/vector quantization | C | decoder required | compression track |
| Codebook compression | shared attribute dictionaries | C | simple decoder possible | common-compatible candidate |
| Neural Gaussian compression | learned rate-distortion decoder | C | decoder and runtime cost | native track |
| GPU texture coding | hardware-friendly block/texture coding | C, arXiv:2607.14513 | GPU decode path | new format row |
| SpeedyGS | content-aware two-stage optimization | C, arXiv:2607.12656 | likely retrained/native | native track |
| QIRF | function-space compression | C, arXiv:2607.18067 | decoder required | research track |

The July 2026 arXiv search also exposes SPARE-GS, CoSAG, and packet-loss
robust Gaussian packaging. They are discovery leads, not measured results.
The repository should record submission date and revision when adding them.

For a 100 MB PLY scene, planning ranges (not measurements) are: conservative
80-95 MB with less than 0.05 dB expected loss; realistic 35-70 MB with
approximately 0.05-0.2 dB expected loss; aggressive 10-35 MB with scene-
dependent visible artifacts. These ranges are deliberately not attributed to
one method until the same checkpoint and decoder are measured.

## 10. Twenty compression ideas

| # | Idea | Ratio target | Quality target | Practicality | Publication |
| ---: | --- | ---: | --- | --- | --- |
| 1 | hierarchical codebooks | 4-12x | <0.2 dB | high | medium |
| 2 | learned Gaussian dictionaries | 6-20x | <0.2 dB | medium | high |
| 3 | neural entropy coding | 8-30x | <0.2 dB | medium | high |
| 4 | SH tokenization | 4-16x | <0.2 dB | high | medium |
| 5 | hybrid geometry/appearance coding | 6-20x | <0.2 dB | high | high |
| 6 | occupancy-conditioned quantization | 4-12x | <0.15 dB | high | medium |
| 7 | tile-local codebooks | 4-14x | <0.2 dB | high | high |
| 8 | view-weighted rate distortion | 5-18x | <0.2 dB target views | medium | high |
| 9 | vector-quantized quaternion/scale | 3-10x | <0.1 dB | high | medium |
| 10 | opacity-run encoding | 2-8x | negligible | high | low |
| 11 | learned residual refinement | 8-25x | <0.2 dB | medium | high |
| 12 | multi-level anchor coding | 8-30x | <0.2 dB | medium | high |
| 13 | camera-aware SH pruning | 4-12x | bounded view loss | medium | high |
| 14 | block floating-point attributes | 3-8x | <0.1 dB | high | medium |
| 15 | GPU texture block packing | 4-16x | <0.2 dB | high | high |
| 16 | progressive network packets | 4-20x | graceful degradation | medium | high |
| 17 | entropy-coded sorted Gaussians | 5-18x | <0.2 dB | high | medium |
| 18 | semantic-region codebooks | 6-20x | region-weighted | medium | high |
| 19 | decoder-fused dequantization | 4-16x | unchanged | medium | high |
| 20 | joint renderer/compressor training | 10-40x | <0.2 dB | low | breakthrough |

The first practical implementation is block floating point plus tile-local
codebooks, because it can preserve the canonical PLY semantics and use the
existing quality adapter. Learned decoders should start in a native track;
otherwise decoder cost and retraining make comparisons invalid.

## 11. Benchmark decision rules

Every proposed row must answer: can it consume `common_representation`? If
yes, add a config and run the quality gate. If it changes the checkpoint,
prunes, retrains, quantizes, or requires a decoder, use a native/compression
track with an explicit baseline. Required metrics are FPS, P95/P99, startup,
decode time, peak VRAM, bytes on disk, PSNR, SSIM, LPIPS, and failure reason.
Ideas that cannot produce a reproducible row are research notes, not rankings.

## 12. Roadmaps

### One month

- **Benchmark Engineering:** publish the EPIC-05 cohort and add a compressed-
  artifact schema with byte and decode metrics.
- **Renderer Research:** calibrate HiGS tile size on occupancy and add a
  deterministic ablation runner.
- **Compression Research:** implement block-float and tile-codebook baselines.
- **Open Source Impact:** add a results manifest and CI checks for stale claims.

### Three months

- **Benchmark Engineering:** add native training and temporal sequence tracks;
  validate fast-gauss on Linux EGL.
- **Renderer Research:** evaluate late-stage and lazy-rebuild HiGS.
- **Compression Research:** compare codebooks, HAC/HAC++, and texture coding
  on identical checkpoints.
- **Workshop Paper:** submit the measurement protocol and HiGS ablation.

### Six months

- **Renderer Research:** prototype incremental hierarchy and Local-GS fusion;
  publish P99 and VRAM ablations.
- **Compression Research:** ship a decoder-fused near-lossless format with a
  strict 0.2 dB gate.
- **Open Source Impact:** add FlashGS/GEMM-GS adapters if quality parity holds.
- **Conference Paper:** submit a renderer/compression Pareto study.

### Twelve months

- **Benchmark Engineering:** multi-GPU, 4K, WebGPU/EGL, and mobile cohorts.
- **Renderer Research:** differentiable HiGS or learned visibility with a
  public training recipe.
- **Compression Research:** joint rate-distortion and renderer co-design.
- **Breakthrough Research:** only claim a breakthrough after independent
  reproduction and a complete common/native matrix.

## 13. Immediate acceptance criteria

The next change should be accepted only when tests pass, the result has a
source commit and hashes, the benchmark command is recorded, the quality gate
is paired to the same camera order, and the generated report explains why the
row is measured, reproduced, paper-reported, or rejected. This keeps the
benchmark itself as the product.
