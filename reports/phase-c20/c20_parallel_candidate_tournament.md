# C20 — Parallel Optimization Candidate Tournament

## Decision

**Keep C6 only.** It produced a measured **3.48% system-level wall-clock reduction** under the stated four-A100, 64-camera protocol. C2 retains `tile_size=16` as the best tested stock-gsplat configuration, but is **not** evidence for an independent CTA mechanism. C3 and C7 are dropped at their implementation/environment gates.

## Experimental record

- **A100 host:** `bms-39468022-001`; NVIDIA A100-PCIE-40GB; Torch `2.7.1+cu118`; gsplat `1.5.3`.
- **Scene:** official Mip-NeRF360 `room`, 1,593,376 Gaussians, 1920×1080, `packed=False`.
- **C2:** two simultaneous eight-camera shards (cameras 0–7 and 80–87), 5 warmups and 20 timed renders per camera/configuration.
- **C3:** fixed tile/CTA `16×16×1` workload replay on A100, 25 repeats per cap. The replay truncates intersections and is intentionally not a correctness candidate.
- **C6:** 64 cameras, four A100s (GPU4–7), 6 profile repeats and 5 execution repeats per policy.
- **C7:** local RTX 5070 execution attempted using the same C2 harness; it failed before first render.

## Ranked candidates

| Rank | Candidate | Verdict | Gain class | Evidence |
|---:|---|---|---|---|
| 1 | **C6 — measured-render-time LPT scheduling** | **KEEP** | System-level | 7,805.591 ms vs static 8,087.171 ms: **3.48% reduction**. All items exactly once; rasterizer configuration unchanged. |
| 2 | C2 — tile/CTA sweep | Drop as mechanism; retain tile16 baseline | Configuration-only | Tile16 is fastest on both A100 camera shards. API couples tile size and CTA, so no decoupled-CTA gain exists in this evidence. |
| 3 | C3 — batch-size candidate | **DROP** | None established | No independent stock-gsplat batch parameter; no modified CUDA kernel; replay changes intersections. |
| 4 | C7 — RTX 5070 cross-hardware check | **DROP** | Not measured | gsplat `csrc` unavailable and JIT fallback cannot find MSVC `cl.exe`. |

## C2 — stock gsplat tile/CTA configuration result

### A100 cameras 0–7

| tile size | CTA | Threads/block | Aggregate median render ms |
|---:|---|---:|---:|
| 8 | 8×8×1 | 64 | 3.428608 |
| 12 | 12×12×1 | 144 | 2.782976 |
| **16** | **16×16×1** | **256** | **2.496000** |
| 20 | 20×20×1 | 400 | 2.600448 |
| 24 | 24×24×1 | 576 | 2.685952 |
| 32 | 32×32×1 | 1024 | 3.052288 |

The camera-80–87 shard independently also selected tile16 (aggregate median **2.564608 ms**). Tile16 is therefore the retained configuration baseline for this workload.

**Interpretation boundary:** standard gsplat ties `tile_size` to the rasterizer CTA dimensions. This test changed both simultaneously. It cannot test C1/C2’s proposed logical-tile/CTA decoupling, and must not be reported as such.

**Correctness audit:** the raw harness made an invalid image comparison for cameras 1–N by comparing them with camera 0’s tile16 reference. Those failures are cross-camera mismatches, not tile-size failures. Same-camera camera-0 checks for tile20 and tile24 were exactly equal. A same-camera comparison for every configuration remains required before adopting a non-tile16 configuration.

## C3 — batch-size implementation gate

C3 held the CTA at `16×16×1` (256 threads) and measured replay workloads from 92.94 to 191.01 mean intersections/tile. The observed cost decreased from **1.234 ns/intersection** at cap 96 to **0.700 ns/intersection** at cap 320. This is workload sensitivity only: the replay truncates input intersections and therefore changes the image.

The decisive result is architectural: stock gsplat exposes **no independent rasterizer batch-size parameter**, and no CUDA kernel was modified. Hence C3 has neither configuration gain nor mechanism gain and is dropped. A new candidate would require a custom compiled rasterizer with an independently parameterized batch plus same-image/gradient validation.

## C6 — multi-GPU scheduling result

| Policy | Wall-clock ms | Delta vs static |
|---|---:|---:|
| Static round-robin | 8,087.171 | baseline |
| Gaussian-count proxy | 7,820.628 | 3.30% |
| Intersection-count LPT | 7,898.607 | 2.33% |
| **Measured-render-time LPT** | **7,805.591** | **3.48%** |

The render-time LPT profile balanced estimated work tightly (36.943–37.048 ms per GPU assignment), and the experiment confirmed every one of 64 camera work items occurred exactly once for each policy. This is a scheduling-only result: tile size, CTA, renderer kernels, intersection ordering, and image computation are unchanged.

**Caveat:** the wall-clock includes launching/loading isolated worker processes. The follow-up must use persistent workers, randomized policy ordering, and several independent repetitions before treating 3.48% as a deployment expectation.

## C7 — RTX 5070 environment gate

No RTX performance datapoint exists. Local gsplat `1.4.0` failed to import `csrc`; its JIT fallback then failed because `where cl` could not find MSVC’s compiler. Repair by installing a compatible gsplat binary or activating MSVC Build Tools, then rerun `scripts/phase-c20/c7_cross_hardware.py` unchanged.

## Raw evidence

- `results/phase-c20/c2_gpu0_cta.json`
- `results/phase-c20/c2_gpu1_workload.json`
- `results/phase-c20/c3_gpu2.json`
- `results/phase-c20/c3_gpu3_dense.json`
- `results/phase-c20/c6_multigpu.json`
- `results/phase-c20/c7_rtx5070.json`
- `results/phase-c20/c20_parallel_candidate_tournament.json`

## Candidate research characterization

| Candidate | Observed mechanism | Evidence strength | Workload dependence | Hardware dependence | Implementation cost | Potential research direction |
|---|---|---|---|---|---|---|
| **C6** | Profiling actual camera render cost before LPT assignment reduces the slowest-device tail relative to static assignment. | Moderate: one 64-camera, four-GPU run; exact-once work and fixed renderer checks pass, but startup/loading is included. | Expected when camera render cost is heterogeneous; benefit may disappear for uniform trajectories. | Measured only on homogeneous A100 GPUs. A cross-GPU policy needs new profiling data. | Low–moderate: profiling plus persistent-worker scheduler; no CUDA changes. | Persistent workers, randomized policy order, and repeated steady-state scheduling trials. |
| **C2** | A coupled tile/CTA configuration curve has a minimum at tile16; smaller tiles increase intersections and larger tiles increase per-tile execution cost. | Moderate for the chosen A100 workload: two disjoint camera shards agree on tile16. | Strong: tile occupancy/intersection distribution determines the configuration trade-off. | A100 only; C7 did not execute. | None for retaining tile16; high for a true decoupled logical-tile/CTA prototype. | Only after a separate CUDA implementation can expose CTA geometry independently of logical tile size. |
| **C3** | No mechanism was exercised: stock gsplat has no independently configurable rasterizer batch parameter. | Strong for this implementation boundary; replay timings do not constitute optimization evidence. | Replay confirms rasterizer cost changes with input intersection count, but it changes image content. | A100 only; hardware comparison is not applicable without an implementation. | High: custom CUDA rasterizer, forward and gradient correctness gates. | Parameterize shared-memory batch size independently and compare same-image/gradient-correct implementations. |
| **C7** | No mechanism observed because the local renderer did not initialize. | Strong for the environment failure; absent for performance. | Not measured. | The result is specific to the local Windows/gsplat/MSVC environment, not the RTX 5070 architecture. | Low environment repair, then rerun existing harness. | Install compatible gsplat binary or MSVC Build Tools and repeat the unchanged A100 protocol. |

## Next action

Proceed only with **C6 persistent-worker scheduling validation**. Keep tile16 as the C2 baseline. Do not implement C3, and do not claim RTX cross-hardware behavior until C7’s local environment is repaired and the shared protocol completes.
