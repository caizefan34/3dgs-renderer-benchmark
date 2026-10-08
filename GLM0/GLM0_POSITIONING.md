# GLM0 Positioning - Novelty Audit vs Prior 3DGS Optimization Work

**GLM0 question:** Is Gaussian lifecycle (birth -> growth -> mature -> inactive -> death)
a new, exploitable 3DGS training-optimization dimension beyond visibility?
**Method:** pure oracle on frozen-C0 traces (room/garden/bicycle, seed 42, 30K). No
production code changed. All numbers measured on EPIC-05 (8x A100-SXM4-80GB).

## 0. Short form

GLM0 finds a stable, visibility-distinct lifecycle structure whose actionable lever -
reducing the optimizer-update frequency of visible-but-mature Gaussians - yields an
**additional 12.84% safe-skip headroom** beyond the visibility oracle (VOS), pooled over
3 scenes, robust to threshold choice. The lever is NOT visibility skipping (that is
already captured by VOS/GS-Scale/Sparse-Adam); it is the maturity axis that visibility
alone cannot see. **Verdict: GLM0_PROMISING** (enter GLM1 with a quality-gated
lifecycle scheduler).

## 1. Closest prior art (repo-verified)

| Work | Problem it solves | Lifecycle? | Relation to GLM0 |
|---|---|---|---|
| FastGS | Density/training acceleration via adaptive density control | NO | Different lever (densification schedule), not update scheduling. No collision. |
| GS-Scale (ASPLOS 2026) | Deferred optimizer update for zero-gradient (invisible) Gaussians; 4-bit defer counter (<=15), restores on reactivation | NO (visibility only) | Same class as VOS: defers invisible Gaussians. Does NOT reduce updates of visible Gaussians. The GLM0 lever (visible-mature frequency reduction) is outside its counter model. Partial overlap on the INACTIVE-skip sub-lever only. |
| Sparse Adam (AdamW-GS, ICLR 2026) | Parameter sparsity + visibility-aware optimizer; re-state / decoupled-attribute regularization | NO (visibility only) | Storage/sparsity + visibility. No lifecycle (maturity) axis. No collision on frequency reduction. |
| VOS-GS (this repo) | Visibility scheduling: virtualize optimizer state of invisible rows; exact dense-Adam replay on reactivation | PARTIAL | VOS = the visibility oracle GLM0 compares against. VOS skips invisible; GLM0 adds visible-but-mature reduction. VOS verdict KEEP (1.5-2.45x stage speedup, quality-gated). GLM0 is the lifecycle extension on top of the VOS visibility baseline. |
| ReproTopo (this repo) | Topology consistency: deterministic clone/split/prune/compaction | NO | Correctness of topology ops, not scheduling. GLM0 reads topology history (birth_type, parent lineage) as a feature but does not alter topology. No collision. |
| LAVOS (this repo) | Lineage-Aware Exact VOS: exact logical-time virtualization of dense Adam across a dynamic population with stable lineage | RELATED BUT DIFFERENT | LAVOS tracks lineage to virtualize optimizer state exactly (systems correctness). GLM0 uses lifecycle stage to schedule update frequency (optimization). Lineage is a prerequisite signal, not the contribution. No collision on the scheduling claim. |

## 2. What is actually new in GLM0

The headroom decomposes cleanly:

- **Visibility skipping (VOS / GS-Scale / Sparse Adam):** skip invisible Gaussians.
  Measured: 73.65% of all optimizer updates are on invisible Gaussians (pooled).
  This is the already-exploited majority - not new.
- **Lifecycle frequency reduction (GLM0):** among the 26.35% visible updates VOS still
  performs, reduce the mature subset update frequency (every 2nd iter) while keeping
  young (NEW) gaussians at full update. Measured additional: 12.84% of total updates
  (= 48.7% of the visible-update residual).

The novelty is the maturity axis: visibility cannot distinguish a young-but-visible
Gaussian (must update every iter to establish) from an old-but-visible Gaussian
(converged, safe to halve). Lifecycle (age + contribution history) provides exactly this
split. The GS-Scale counter defers only zero-gradient rows and restores on reactivation;
it never halves the update rate of a continuously visible, converged Gaussian. That is
the GLM0 opening.

## 3. Evidence that lifecycle != visibility (criterion 2)

- corr(stage, vis_frac) ~ 0.39-0.41 (moderate, not deterministic): stage is related to
  but not reducible to visibility.
- INACTIVE gaussians are sometimes visible (not always invisible): always-invisible
  fraction 26.5% (room), 14.1% (bicycle), 0.01% (garden). The visible-INACTIVE updates are
  pure lifecycle headroom VOS cannot see.
- NEW and MATURE have near-equal median vis_frac (~0.30 room/garden) but opposite
  update policy (full vs halved) - the split is driven by maturity (age), invisible to
  a visibility-only method.

## 4. Honest novelty caveats

1. The INACTIVE-skip sub-lever (~0.5% pooled at freq=1) is NOT novel - it overlaps
   GS-Scale/VOS/Sparse Adam. The novel and dominant lever is MATURE frequency reduction.
2. The 12.84% is a theoretical oracle assuming mature-Gaussian updates can be halved
   without quality loss. GLM1 must confirm this with a quality gate (PSNR/SSIM/LPIPS
   non-inferiority) exactly as VOS did. If mature reduction proves quality-unsafe, the
   realized headroom collapses toward the INACTIVE-only 0.5%.
3. Infrequent update of converged parameters has analogues in classical optimization
   (lazy/infrequent updates, event-triggered control). The GLM0 claim is scoped to
   3DGS optimizer scheduling on a dynamic population with clone/split/prune lineage -
   the combination, not the isolated idea.

## 5. Positioning statement (for paper)

> Gaussian lifecycle - not visibility - is the actionable scheduling signal for 3DGS
> optimizer work. Visibility methods (GS-Scale, Sparse Adam, VOS) already remove the
> 73.65% of updates spent on invisible Gaussians. The remaining, previously-unexploited
> headroom lives in the visible-but-mature majority: lifecycle-aware frequency reduction
> adds a further 12.84% safe-skip headroom (48.7% of the visible residual), a dimension
> orthogonal to visibility.
