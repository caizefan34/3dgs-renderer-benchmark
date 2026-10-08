# DPC0 — Deterministic Primitive-Centric Backward Feasibility (FROZEN)

## 0. Goal

Determine whether the dominant rasterized 3DGS backward can be reorganized from pixel-centric
floating-point atomic scatter into a race-free primitive-centric gather/reduction that is:

1. deterministic;
2. gradient-correct;
3. faster enough to justify a full deterministic 3DGS training system.

DPC0 is a feasibility gate. It cannot make a publication or >=3x full-training claim.

## 1. Cells

- room@10K
- room@15K
- bicycle@10K
- bicycle@15K

For each cell:
- 8 deterministic midpoint-stratified TRAIN source cameras.
- topology/densification frozen.
- black background and exact C0 loss semantics.

Use GPU0 for primary same-GPU tests.
Use GPU0/GPU1/GPU2/GPU3 for cross-A100 deterministic checks.

## 2. Candidate architecture — frozen

### 2.1 Preserve exact forward

DPC0 MUST use the authoritative C0 forward renderer and exact forward ordering.
No forward approximation.

### 2.2 Tile-primitive partial gradient

For every existing valid (tile, Gaussian) intersection used by the forward pass:

- assign one deterministic warp/cooperative group to the (tile, Gaussian) pair;
- traverse the tile's covered pixels in a fixed pixel-index order;
- compute the same backward contribution formulas as C0;
- reduce lanes with a fixed reduction tree;
- write exactly one partial-gradient record for that (tile, Gaussian) pair.

No floating-point atomicAdd to the final Gaussian gradient is permitted.

### 2.3 Deterministic primitive-major reduction

Create a deterministic primitive-major ordering of partial records:
    primary key = Gaussian ID
    secondary key = tile ID

Stable key construction/tie behavior is required.

Reduce each Gaussian's partials with a fixed tree/order to produce the final raster gradient.

Allowed implementations:
- custom fixed-tree segmented reduction;
- CUB/CCCL run-to-run deterministic reduction, if the ordering and segmentation meet the frozen
  Gaussian-major specification.

Not allowed:
- atomic accumulation to final floating-point Gaussian gradients;
- nondeterministic unordered block completion as the combining order.

### 2.4 Scope

DPC0 covers the rasterization backward outputs needed by the existing projection/backward path.
Do NOT fuse optimizer, projection backward, loss, or densification in DPC0.
Those belong to DPC1 if DPC0 passes.

## 3. Build integrity

Build DPC as a separate extension/library.
Do not overwrite authoritative C0 binaries.

Record:
- source tree hash;
- compiler flags;
- CUDA toolkit;
- cub/cccl version;
- cubin/so SHA256.

## 4. Determinism gate

For every cell/source camera:

A. SAME GPU
- run DPC backward 32 times with identical inputs on GPU0;
- hash every raster-gradient output tensor byte-for-byte;
- require ONE unique hash for all 32 runs.

B. CROSS A100
- run the same input once on GPU0/1/2/3;
- require identical hashes across all four A100s for STRONG promotion.
- Cross-GPU mismatch does not invalidate ordinary promotion if each GPU is independently run-to-run
  deterministic, but must be reported.

C. Baseline noise
Run authoritative C0 backward 8 times on the same inputs and report its gradient relL2/maxabs
repeatability distribution. This is diagnostic only.

## 5. Correctness gate

Because authoritative C0 is itself non-deterministic, correctness is evaluated by three independent
checks.

### 5.1 C0 ensemble agreement

For every source camera:
- obtain 8 authoritative C0 backward gradient samples;
- compute their fp64 mean gradient after copying outputs to CPU/fp64;
- compare DPC to this mean in optimizer-unit coordinates.

Required for every cell:
- cosine median >= 0.999999
- cosine p10 >= 0.999995
- relL2 median <= 5e-5
- relL2 p90 <= 1e-4

### 5.2 Directional derivative sanity

Use 16 frozen optimizer-unit Rademacher directions per cell.
For each direction, compare dot(g_DPC, d) to a central finite-difference derivative of the exact C0
loss.

Use three frozen step sizes in optimizer-unit norm:
    h in {1e-3, 3e-4, 1e-4}

For each direction take the median of the three relative errors as the diagnostic value.

Required:
- median relative directional error <= 5e-3
- p90 <= 2e-2.

If finite differences are numerically degenerate for >25% of directions:
    DPC0_DIRECTIONAL_CHECK_INCONCLUSIVE
This does not automatically fail if 5.1 and 5.3 pass, but prevents STRONG promotion.

### 5.3 One-step native update equivalence

From identical theta0 + optimizer state:
- arm C0: one native C0 gradient/update, repeated 8 times to estimate stochastic-numeric spread;
- arm DPC: one DPC gradient, then the SAME native optimizer update.

Evaluate exact forward loss after update on:
- source camera;
- 3 deterministic peer-train cameras;
- 8 corrected true-holdout cameras.

DPC must lie within the min/max envelope of the 8 C0 runs plus:
- source loss tolerance: 0.1% of |C0 improvement|;
- peer mean-loss tolerance: 0.2%;
- true-holdout PSNR tolerance: 0.01 dB median, 0.03 dB worst.

No LR changes.

## 6. Performance gate

CUDA-event timing after warmup.

Per source camera:
- C0 raster backward only;
- DPC pair/CSR preparation;
- DPC tile-primitive partial kernel;
- DPC primitive reduction;
- DPC total backward path.

Measure 128 repetitions with identical frozen inputs.

Metrics:
- raster backward speedup = C0_bwd / DPC_total_bwd
- DPC preparation fraction
- total partial-record bytes
- peak extra VRAM
- partials per Gaussian distribution
- intersections processed
- deterministic reduction throughput.

Primary timing includes ALL additional DPC preparation/sort/CSR work required for backward.
Do not hide preprocessing.

## 7. Mechanical verdict

Specific failures have precedence:
1. DPC0_SOURCE_IDENTITY_FAIL
2. DPC0_BUILD_INTEGRITY_FAIL
3. DPC0_NONDETERMINISTIC
4. DPC0_GRADIENT_INCORRECT
5. DPC0_TOO_SLOW
6. DPC0_PROMOTE
7. DPC0_STRONG_PROMOTE

### DPC0_STRONG_PROMOTE

All 4 cells:
- SAME-GPU determinism: 32/32 identical hash for every source;
- CROSS-A100 determinism: GPU0/1/2/3 hashes identical for every source;
- C0 ensemble correctness gates all pass;
- directional check conclusive and passes;
- one-step native-update equivalence passes;
- DPC total backward speedup >= 1.75x in every cell;
- median speedup >= 2.0x over all cells;
- peak extra VRAM <= 25% of frozen C0 peak;
- no NaN/Inf.

### DPC0_PROMOTE

All 4 cells:
- SAME-GPU 32/32 deterministic;
- C0 ensemble correctness gates pass;
- one-step native-update equivalence passes;
- DPC total backward speedup >= 1.35x in every cell;
- median speedup >= 1.50x;
- peak extra VRAM <= 40%;
- no NaN/Inf.

Cross-A100 bitwise equality is not mandatory for ordinary PROMOTE.

### DPC0_TOO_SLOW

Correct + deterministic, but speed gate fails.

### DPC0_GRADIENT_INCORRECT

Deterministic but correctness gate fails.

### DPC0_NONDETERMINISTIC

Any same-GPU source camera produces more than one DPC gradient hash.

## 8. DPC1 authorization if DPC0 passes

Only DPC0_PROMOTE or STRONG_PROMOTE authorizes DPC1.

DPC1 may integrate already-known systems components:
- exact tile culling;
- deterministic/fused projection backward + Adam;
- fused loss;
- deterministic densification statistics/decisions;
- SH memory-layout optimization.

These are SYSTEM components and are not individually novelty claims.

DPC1 target:
- deterministic full state transition;
- >=2.5x full-iteration speed on room/bicycle mature cells.

DPC2 target:
- full 30K room/garden/bicycle;
- repeated same-seed trajectories;
- bitwise-identical final checkpoint/topology on the same GPU/build;
- preferably identical across A100 devices;
- >=3.0x full-training wall vs frozen C0;
- held-out PSNR delta >= -0.05 dB median, worst >= -0.15 dB;
- comparison to E-FGS and a current fast systems baseline.

Only DPC2-level evidence supports a strong paper claim.

## 9. Forbidden rescue

No:
- changing forward equations/order;
- approximate gradients;
- fp16/bf16 accumulation;
- relaxed determinism ("close enough" is not deterministic);
- excluding pair-list construction from timing;
- threshold changes after results;
- hidden atomics to final gradients;
- local GPU;
- densification in DPC0.

## 10. Required outputs

DPC0_ENVIRONMENT.json
DPC0_SOURCE_IDENTITY.json
DPC0_BUILD.json
DPC0_DETERMINISM.json
DPC0_CORRECTNESS.json
DPC0_DIRECTIONAL.json
DPC0_ONE_STEP.json
DPC0_TIMING.json
DPC0_SUMMARY.json
DPC0_VERDICT.json
DPC0_REPORT.md
MANIFEST.sha256
