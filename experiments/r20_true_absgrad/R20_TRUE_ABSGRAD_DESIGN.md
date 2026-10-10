# R20 True AbsGrad implementation and review

Status: SOURCE_PATCH_COMPLETE; local validation in R20_LOCAL_VALIDATION.json;
full frozen MX parity and training NOT_EXECUTED. Date: 2026-10-09.

## Scope

Only four renderer source files are patched. Forward code, Gaussian projection
VJP, SH VJP, parameter-gradient reductions, thresholds, binaries and checkpoints
are unchanged. A separate R20 trainer copy selects the true side channel; the
historical R19 trainer remains intact. The requested branch is
`feature/r20-higs-true-absgrad`.

Both blend kernels have a compile-time `ABSGRAD` parameter. The dense variant
captures `fabsf` of the per-pixel output. The PX=1/2/4 variant adds each pixel's
componentwise fabs into a separate thread accumulator before signed q sums.
All lanes join a separate warpSum; lane zero adds two FP32 absolute components
to the same flattened Gaussian row used for signed gradients. No pixel-by-Gaussian
or intersection-by-Gaussian global intermediate is introduced.

The existing signed statements and helper calls retain their order. The extra
accumulator never flows into conic, opacity, SH, position, rotation or scale VJP.
`ABSGRAD=false` compiles out local fabs, extra reductions and atomics. It creates
no absolute buffer; the internal eighth tensor is undefined. Legacy callers
retain the seven-tensor tuple through a thin C++ adapter. This preserves numerical
behavior; strict binary identity and identical generated machine code are not claimed.

## API and output contract

Old API: `higs_rasterize_backward(...)` -> exactly seven existing tensors.

New API: `higs_rasterize_backward_with_absgrad(..., absgrad=False)` ->

| Index | Output | Shape/dtype |
|---|---|---|
| 0 | grad means master | [M,3], FP32 |
| 1 | grad quaternions master | [M,4], FP32 |
| 2 | grad activated scales master | [M,3], FP32 |
| 3 | grad activated opacity master | [M], FP32 |
| 4 | grad RGB or SH master | [M,3] or [M,K,3], FP32 |
| 5 | grad background | [D], FP32, summed over images |
| 6 | signed projected-mean gradient | [C*N_visible,2], FP32 |
| 7 | absolute projected-mean gradient | [C*N_visible,2], FP32 when enabled; undefined/None otherwise |

The extension exports `higs_true_absgrad_version=1`. Stage B/C public wrappers
accept a new keyword-only `absgrad=False`; the autograd Function has a final
non-tensor input and matching final `None` gradient. False uses the old API,
including support for the old frozen experimental binary. True requires the
versioned native extension and rejects missing capability or recompute fallback.
The capability check occurs during forward, before training work is accepted.

Python maps `[C,N_visible,2]` into a zero-filled `[C,M,2]` using the existing,
strictly increasing duplicate-free `visible_ids`. The all-invisible fast path
sets the full absolute tensor to zero without launching CUDA. Densification
reads `out['densification_info']['means2d'].absgrad` after `loss.backward()`.
No Python hook computes abs(signed) in the True AbsGrad arm.

## Cost estimate, not a measurement

Additional GPU storage is `8*C*N_visible` bytes for the CUDA buffer plus
`8*C*M` bytes for the public master-layout side channel. These coexist during
scatter. At C=4 and M=N_visible=4,413,928: each is 141,245,696 bytes = 134.7 MiB;
transient extra total about 269.4 MiB, final retained side channel about 134.7 MiB.
Allocator rounding/fragmentation and tensor metadata add overhead. At C=1,
the corresponding transient total is about 67.4 MiB. Zero initialization and
Python index_copy add bandwidth cost. The public full tensor is necessary for
the current consumer; the compact buffer is the kernel output, not a pixel cache.

Each contributing Gaussian/warp group adds two FP32 atomics. RGB path existing
scatter has 3 color + 3 conic + 2 signed + 1 opacity = 9 scalar atomics, so this
stage's scatter count becomes 11 (+22.2%); this is NOT a latency prediction.
PX shares each lane across pixels, so PX=2/4 retains fewer scatter groups while
preserving pixel fabs semantics. Two extra accumulator registers and two warp
reductions can affect register allocation/occupancy. Highly overlapping large
Gaussians can contend across tiles. Measure ON/OFF at fixed Gaussian topology,
camera sequence and cotangents with the supplied microbench; then measure training.

## Review findings and constraints

| Risk | Reasoning / required validation |
|---|---|
| Race | Output starts zero, cross-warp/block writes use atomicAdd; Python scatter is stream ordered and IDs unique. No new unprotected global writes. |
| Synchronization | No new barriers or early returns; existing block.sync unchanged. ABSGRAD specialization is uniform for every lane, and all lanes in a participating warp reduce zeros for invalid pixels. |
| Warp errors | fabs occurs before q/warp sums; inactive scratch zero; lane-zero scatter uses the same g as signed. Test cancellation inside a PX thread and across warp lanes. |
| Indexing | g is an image-major compact row from flatten_ids; never apply visible_ids in the pixel kernel. Master mapping occurs once in Python. Test C>1 and invisible holes. |
| Tile overlap | Distinct tiles own disjoint pixels. A Gaussian appearing in several tile lists is expected; each pixel contributes once. Duplicate active tile IDs would double count ALL existing gradients; current wrapper constructs IDs by nonzero on a boolean mask. |
| Gradient scaling | Use identical upstream cotangents, no CUDA W/H/C multiplier, consumer unchanged. Tests include alpha-only loss and raw sigmoid/exp/normalize/SH chains. AMP unscale is outside this validated FP32 contract. |
| Floating point | FP32 atomics are order dependent; native system atomics vs device atomics and PX grouping can produce rounding differences. Evaluate absolute/relative errors and exact decision masks, especially near thresholds. |
| Tile geometry | Existing PX math hardcodes 16 rows; True AbsGrad rejects tile_size !=16. False preserves old behavior. |
| PX=8 | Existing runtime parser accepts 8 but launcher has no 8 branch and raises. R20 does not silently enable this pre-existing unsupported path. Test supported 0/1/2/4 only. |
| Background edges | Existing dense/PX background atomic lacks inside gating on padded pixels. Partial-tile nonzero-background gradients need separate audit; R20 does not modify that signed behavior. Black-background RGB tests avoid this confound. |
| Active tiles | Existing sampled path requires unique tile IDs and consistent forward mask; no semantics change. Full R20 reference recipe uses full coverage. Partial sampling needs a separate masked-reference gate before use. |

MX acceptance requires per-Gaussian native comparison, raw-gradient and forward
parity, false-path comparison against frozen HiGS, threshold/candidate agreement,
then complete paired training. A successful compilation or local extracted
pixel-kernel test is insufficient to declare frozen CUDA parity or quality recovery.
