# P11 Opportunity + Prior-Art Audit — Phase 1

## Decision

Do **not** implement another raster-backward arithmetic trick.

The new search must target work large enough to survive complete training. The fresh
DSH-A BASE campaign gives the strongest workload decomposition currently available:

| Phase | Mean ms/iter | Fraction |
|---|---:|---:|
| forward | 4.553 | 12.62% |
| loss forward | 7.326 | **20.31%** |
| `loss.backward()` total | 19.057 | **52.83%** |
| densification + per-step grad-stat work | 0.780 | 2.16% |
| optimizer | 4.226 | 11.71% |
| other | 0.132 | 0.36% |

Important: the 19.057 ms timer contains both photometric-loss backward and renderer
backward. It must be decomposed before any new CUDA design.

## Historical candidates that are now closed

### Sparse-tail / active interaction
The project already ran C25–C28. Early measurements suggested a large tail, but the
later audit found that existing `last_ids`/warp termination explains most of the
apparent waste. Current gsplat additionally has active-tile and contribution-index
machinery. Do not restart this line without new current-main evidence.

### Gradient zero-init
R6 originally over-attributed all memset/fill activity to gradient zeroing. Direct
CUDA-event measurement corrected the actual gradient-buffer clear to only
0.14–0.36% of backward. This candidate is closed.

### Block-level atomic aggregation
After direct instrumentation, the estimated median E2E opportunity fell to about
1.38% (best ~2.57%), while Taming/DISTWAR/current-gsplat already crowd the reduction
design space. It is not a suitable new core contribution.

### Backward-optimizer fusion
The repaired R6-C bandwidth model gives only ~0.48–2.21% E2E opportunity. Modern
public implementations also include fused/sparse Adam paths. Defer.

## Primary candidate: exact loss-adjoint ↔ raster-backward boundary

The frozen publication trainer computes

\[
L = 0.8 L_1 + 0.2(1-\mathrm{SSIM})
\]

with an 11x11 Gaussian SSIM. Loss forward alone costs about 7.33 ms/iter in the
fresh 13-scene BASE campaign.

However, **CUDA SSIM itself is not novel**. Current gsplat can call the public
`fused-ssim` package, and the package supports `padding="same"`, which is the relevant
semantic gate for the trainer's same-padded separable SSIM. Graphdeco's accelerated
training path also uses fused SSIM. Therefore P11 must first establish a strong
same-semantics fused-SSIM baseline.

The research question is narrower:

> After replacing the Python/PyTorch SepSSIM implementation with an existing
> same-semantics fused SSIM baseline, is there still at least 2% E2E of physically
> removable work at the **loss-adjoint → raster-backward boundary**, specifically
> separate loss-backward launches and materialization/readback of the full
> `dL/dimage` tensor?

Only if the answer is yes may we prototype a tile-coupled exact loss-adjoint producer
that feeds raster backward directly.

Potential representation:
- raster backward already owns a 16x16 pixel tile;
- exact SSIM backward has bounded spatial support;
- a fused design may compute the final L1+SSIM pixel adjoint using a halo and consume
  it immediately as `v_render_colors`, avoiding a standalone full-image adjoint
  materialization.
- This is a hypothesis, not yet a contribution claim.

Kill the idea if:
1. same-padding fused-SSIM already removes essentially all loss cost; or
2. remaining loss-backward + image-adjoint traffic/launch overhead is <2% E2E; or
3. exact halo/recomputation cost makes the fusion bound unattractive.

## Backup candidate: densification-stat sink

During iterations <15K the training recipe requires `absgrad=True`. Current gsplat
materializes the per-Gaussian absolute 2D gradient; the trainer then performs
visibility selection, pixel-space rescaling, L2 norm, and accumulation into persistent
densification statistics.

This is exact auxiliary work and can be profiled separately. A possible fusion would
sink the final densification statistic into persistent state without the Python/Torch
postprocessing chain.

But the entire measured densification/stat phase averages only ~2.16% E2E, so the
candidate is viable only if the **absgrad generation inside backward plus post-backward
state update** jointly exceed the 2% gate.

Historical evidence that turning absgrad off after densification stops improved the
late-training throughput is useful as a control, not as a new contribution.

## Strong-baseline rule

No new P11 optimization may be evaluated only against the current slow SepSSIM trainer
if an existing same-semantics fused implementation removes the same work.

The implementation gate is:

\[
\text{remaining exact removable headroom after strong baseline} \ge 2\%\ T_{iter}.
\]

Prefer >=3% predicted renderer/training headroom before writing substantial CUDA.

## P11 Phase-2 output

Run profiler-only gates on room, bicycle, garden using representative fixed states
(early/mid/late when needed). Do not train 30K and do not modify renderer algorithms.

At the end classify exactly one candidate as:

- `ADVANCE_TO_MINIMAL_PROTOTYPE`
- or `NO_NEW_OPTIMIZATION_JUSTIFIED`.

A negative result is acceptable.
