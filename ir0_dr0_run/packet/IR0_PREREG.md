# IR0 — IncrementalRaster-GS Opportunity Oracle (FROZEN)

## 0. Purpose

Cheaply falsify IRGS before implementing a persistent differentiable compositing data structure.

The test must answer four questions:

1. Is optimizer utility concentrated enough that a small active Gaussian set is meaningful?
2. Under the *current exact support*, how much pixel-hit / segment work would those active Gaussians dirty? This is an optimistic lower bound.
3. After applying the exact sparse Adam update, how much additional projection/intersection/order repair is required?
4. Does the sparse update retain enough realized loss improvement to justify a block-coordinate training method?

If the optimistic fixed-support lower bound already fails, STOP. Do not rescue with new selection heuristics.

## 1. Frozen source / cells

Primary cells:
- room @ 20K
- room @ 25K
- bicycle @ 20K
- bicycle @ 25K

Use exactly the same frozen C0 renderer/checkpoints and 8 frozen cameras per cell used by the completed formal A0/MOONSHOT measurements when available.

Known C0 identities:
- C0 extension SHA256:
  `d26cccf46c7932bfb5fd44ee05e553af27e17c93eaff6dacf3197258a513f11`
- core `gsplat_cuda.so` SHA256:
  `935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e`
- room@25K checkpoint SHA256:
  `b31e5a6b0e733911b736bbad2e4f9d4d804d5929655f64efb51a9e5132fc047d`

For all other checkpoint/camera/source hashes, recover the COMPLETE expected values from the formal A0/MOONSHOT manifests. Never infer or truncate.

If identity does not match:
    IR0_IDENTITY_FAIL
    STOP.

Do not change:
- resolution
- loss
- eps2d
- tile size
- camera set
- background
- checkpoint
- optimizer semantics
- densification state

No densification is executed inside a DR0 probe.

## 2. Active-set oracle

For each camera probe, run one ordinary exact native forward/backward and construct the native Adam update `delta_theta`.

For Gaussian i define the *first-order positive optimizer utility proxy*:

    u_i = max(0, - <g_i, delta_theta_i>)

where the inner product includes all trainable parameter families belonging to Gaussian i, using the exact native learning-rate / Adam update that would be applied at this state.

This is an ORACLE ranking only. It is not a deployable selector and must not be described as one.

Freeze active fractions globally:

    f in {0.05, 0.10, 0.20, 0.30}

For each f, select top-f visible Gaussians by u_i.
Ties are broken by Gaussian ID.

Do not add new fractions after looking at results.

Record:
- positive utility captured by active set
- parameter-family utility composition
- visible Gaussian count
- active Gaussian count

## 3. Stage IR0-A — optimistic fixed-support dirty-work lower bound

Use the exact native ordered hit / bucket trace at theta_t.

For each active set, mark every primitive work item whose value would change if only those active Gaussians changed.

At minimum report:

- `dirty_hit_fraction`
  = pixel-Gaussian accepted hits whose Gaussian is active / all accepted hits.

- `dirty_tile_intersection_fraction`
  = Gaussian-tile intersections belonging to active Gaussians / all intersections.

- `dirty_pixel_fraction`
  = pixels touched by >=1 active Gaussian / all rendered pixels.

- `dirty_segment_fraction_B`
  for B in {8,16,32,64}:
  partition each depth-ordered pixel list into consecutive B-hit segments;
  a segment is dirty iff it contains >=1 active hit;
  ratio = dirty segments / all segments.

- `dirty_segment_hit_fraction_B`:
  total hit slots in dirty segments / total accepted hits.

This stage deliberately assumes support/order do NOT change. Therefore it is an optimistic LOWER BOUND on incremental work.

### IR0-A immediate stop

If, for both f=0.10 and f=0.20, ANY cell has:

    median_8cams(dirty_segment_hit_fraction_B=32) > 0.50

then:
    IR0_NO_LOCALITY
    STOP before dynamic-repair probes.

This is a hard frozen gate.

## 4. Cache-memory model

For each B in {8,16,32,64}, estimate a persistent per-pixel segment-summary cache.

Minimal summary per segment:
- RGB premultiplied C: 3 x fp32
- transmittance T: 1 x fp32
= 16 bytes.

Also report a conservative 32-byte/node model to allow metadata/alignment/double buffering.

Report:
- number of segment nodes
- cache_GiB_16B
- cache_GiB_32B

Do not claim an implementation memory footprint from this model.

Hard memory gate for promotion:
    cache_GiB_32B(B=32) <= 16 GiB
in every cell.

## 5. Stage IR0-B — exact sparse-update repair census

Run only if IR0-A does not hard-stop.

For each f in {0.10,0.20}:

1. Clone all model + optimizer state.
2. Apply the exact native Adam update only to active Gaussians.
3. Freeze inactive Gaussian parameters exactly.
4. Reproject/rasterize using the original renderer for measurement only.
5. Compare theta_t support/order to theta_sparse.

Count exactly:
- active projection rows
- old and new Gaussian-tile intersections for active Gaussians
- inserted intersections
- deleted intersections
- active-hit additions/deletions
- pixels whose ordered hit list changes
- depth-order swaps involving an active Gaussian
- B=32 segments invalidated after exact repair
- total hit slots that must be recomputed under exact B=32 segment repair

Define:

    exact_repair_hit_fraction

as the union of B=32 hit slots whose segment must be recomputed after the sparse update, divided by all native accepted-hit slots at theta_t plus newly inserted slots.

Do not rebuild or mutate the frozen source tree to collect this census.

## 6. Stage IR0-C — realized one-step utility retention

For the SAME native gradient / Adam state, construct three states:

BASE:
    theta_t

FULL:
    exact ordinary native Adam update to all Gaussians

SPARSE_f:
    exact same Adam update, but only active Gaussians are committed

No re-optimization, LR tuning, damping, extra views, or compensation is allowed.

Primary metric on the camera that generated the gradient:

    R_loss(f) =
      (L(BASE) - L(SPARSE_f))
      /
      max(eps, L(BASE) - L(FULL))

If FULL does not improve the current-view loss, mark the probe `NON_DESCENT` and exclude it from the ratio denominator statistics while reporting its count.

Also evaluate BASE/FULL/SPARSE on the other 7 frozen cameras as a descriptive multiview guard:
- mean loss delta
- PSNR delta
- SSIM delta
Do not use held-out results to retune f.

## 7. Same-state phase timing for oracle headroom

Measure, on the same cell/camera cohort, publication-clean median timing for:
- projection/SH production
- tile intersection/sort
- raster forward
- loss
- raster backward / projection VJP
- optimizer

Use existing trusted timing artifacts if source identity and checkpoint/camera semantics match exactly; otherwise remeasure.

Construct an OPTIMISTIC incremental lower-bound time for each f:

- active-only projection work scales by active visible fraction;
- active-only intersection work scales by exact active-intersection fraction;
- raster forward/backward work scales by `exact_repair_hit_fraction`;
- optimizer work scales by active Gaussian fraction;
- loss remains full cost;
- dynamic-tree maintenance cost is set to ZERO in DR0.

This intentionally overestimates possible speedup.

Define:

    S_oracle_upper = T_full / T_optimistic_lower_bound

It is an upper bound, NOT a speedup result.

## 8. Frozen verdict

The analyzer evaluates f=0.10 and f=0.20 only.

### IR0_STRONG_PROMOTE
Exists ONE fixed f, the same across all 4 cells, such that every cell satisfies:
- median utility_capture >= 0.80
- median exact_repair_hit_fraction <= 0.25
- p90 exact_repair_hit_fraction <= 0.40
- median R_loss >= 0.75
- p10 R_loss >= 0.50
- cache_GiB_32B_B32 <= 16
- S_oracle_upper >= 3.0
- NON_DESCENT <= 2/8 cameras

### IR0_PROMOTE
Exists ONE fixed f, same across all cells, such that every cell satisfies:
- median utility_capture >= 0.75
- median exact_repair_hit_fraction <= 0.40
- p90 exact_repair_hit_fraction <= 0.55
- median R_loss >= 0.65
- p10 R_loss >= 0.40
- cache_GiB_32B_B32 <= 16
- S_oracle_upper >= 2.0
- NON_DESCENT <= 2/8

### IR0_DROP
Otherwise.

No per-cell choice of f.
No threshold edits.
No new active-set score.
No production CUDA rescue.

## 9. Required artifacts

- `IR0_SOURCE_IDENTITY.json`
- `IR0_ENVIRONMENT.json`
- `IR0_ACTIVE_UTILITY.json`
- `IR0_FIXED_SUPPORT_LOCALITY.json`
- `IR0_CACHE_MEMORY.json`
- `IR0_DYNAMIC_REPAIR.json`
- `IR0_ONE_STEP_UTILITY.json`
- `IR0_PHASE_TIMING.json`
- `IR0_SUMMARY.json`
- `IR0_VERDICT.json`
- `IR0_REPORT.md`
- raw traces/logs
- SHA256 manifest

## 10. Interpretation boundary

A PROMOTE verdict proves only that an incremental rasterizer has enough *oracle opportunity* to justify IR1.

It does NOT prove:
- a deployable active-set predictor exists;
- a dynamic tree is fast on GPU;
- full training converges;
- the method is novel;
- 2x/3x speedup is realized.

Those questions belong to later gates.
