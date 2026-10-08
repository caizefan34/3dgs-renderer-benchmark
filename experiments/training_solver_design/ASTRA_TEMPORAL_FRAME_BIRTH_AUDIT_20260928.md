# Historical densification statistics in moving Gaussian frames

2026-09-28. Read-only pinned-source and bounded literature audit. No CPU/GPU experiment, training run, or Luna-file edit.

## Decision

**The source allows the proposed cross-iteration frame/history mismatch. Actual rotation drift, child inflation and accepted-work opportunity have not been measured.** This is different from the rejected fresh-checkpoint twelve-view eigenframe hypothesis: it concerns how the *actual observation trajectory* is summarized, even when the final-view metric is full rank. A read-only historical trace audit is justified if those packets exist. A new GPU rerun or method arm is not authorized by this audit.

Do not call the behavior a proved bug or the correction an original algorithm. The three channels can be interpreted as persistent body-axis parameters, so a maximum over their historical demand is a meaningful heuristic. Re-evaluating historical world directions in the current frame implements a different heuristic unless a current-frame invariant is explicitly specified. SAD also argues that spare directional capacity can help immature geometry; eliminating historical demand could damage later quality. Matrix congruence transport itself is standard mathematics, and repairing stale statistics alone is a baseline correction rather than a sufficiently original contribution.

## Source facts

Pinned source: [SADGS 029814fb5ed773e4a794cec5115951a64f610ab8](https://github.com/LinjieLyu/SADGS/tree/029814fb5ed773e4a794cec5115951a64f610ab8).

- `train.py` calls `update_freq_stats_online` for each camera in the batch at iterations divisible by 10, while densification is active. Normal densification occurs every 500 iterations in the frozen recipes; successful normal events reset frequency accumulators.
- `utils/freq_utils.py:391–392` computes a componentwise maximum into `max_eta_3ch`. This is over valid observations, not only high observations. The three channels come from the Gaussian's **then-current** scales, rotation, center, projected footprint and sampled image-frequency tensor.
- The hybrid optimizer updates `_rotation` and `_scaling` every ordinary training iteration. The default rates are rotation .002 and scaling .01. Room's scale/rotation scheduler changes those positive rates from 3x initial to .5x final; it does not freeze them. Garden uses the unscheduled rates in the provided recipe.
- No coordinate transport of `max_eta_3ch` was found on optimizer updates. Frequency accumulation happens before the optimizer step in that iteration, and an event consumes the accumulated statistics before that iteration's optimizer step.
- Warmup topology edits may happen every 100 iterations between normal events. Pruning compacts rows and newborn statistics are reset. A trace must use stable primitive identities/lineage, not raw array rows. Opacity reset and masks can alter which observations are eligible, without resetting every historical direction maximum.

These facts prove simultaneous *opportunity* for frame movement and untransported history, not that any particular event parent actually rotated by a significant angle. The single event-1500 capture cannot reconstruct the prior rotation/scale trajectory or full frequency observations.

## Correct quantities and a bounded algebraic example

Let observation `t` use world covariance factor `L_t=R_t diag(s_t)` and world frequency metric `B_t=J_t^T F_t J_t`, including the world-to-camera transform in `J_t`. The local metric is `H_t=L_t^T B_t L_t`. In the native wavelength branch `F_t` is the scalar squared frequency times identity; in projection mode it is the sampled structure tensor. Preserve the original epsilon conventions and eta units when verifying executable parity.

The native observed demand is `m_i=max_t sqrt(e_i^T H_t e_i)` and its executable count is `ceil(sqrt(max(m_i,1)))`. A current-frame interpretation of the same historical **world** observations is instead

`H_(t→T)=L_T^T B_t L_T = L_T^T L_t^(-T) H_t L_t^(-1) L_T`,

`m_T(q)=max_t sqrt(q^T H_(t→T) q)`.

Keep the directional maximum; do not replace it with an average. A maximum over quadratic forms is generally not one quadratic form, and elementwise matrix maximum is not a conservative PSD-envelope operation. Exact historical observations can be retained for an audit; a compact conservative production envelope is an additional unsolved cost/design problem.

For a pure illustrative projection metric, take constant `B=diag(81,0,0)`, unit scales, and rotations that successively align world x with each local body axis, ending at identity. Each local axis attains native eta 9 at some time, giving `k=(3,3,3)` and 27 children. Evaluating the unchanged world metric in the final frame gives eta `(9,0,0)`, hence `(3,1,1)` and 3 children. For the native wavelength-style rank-two projection, `B=diag(81,81,0)` similarly permits a 27 versus 9 example. These are algebraic counterexamples to equivalence, **not synthetic experiment results or evidence about room/garden**.

Transporting `B_t` does not make it a fresh observation: its Jacobian, sampled structure tensor, visibility and footprint belong to the old center/state. Large translation or changing texture/occlusion may make it stale. This is why world-frame transport and a current-view rerender are different controls.

## Minimum falsifiable trace, before changing a birth operator

Use one previously fixed normal-event window, preferably the already selected room 1000→1500 interval. Do not choose a new scene/event after examining drift. If the necessary historical packets do not exist, explicitly report that the existing capture cannot answer the question. This document does not request a rerun.

For every observed primitive in a declared traced cohort, retain stable ID, observation time/camera, active/mask decision, center, normalized rotation, scales, sampled frequency input, projected Jacobian or complete symmetric world `B_t`, instantaneous native eta, and running maxima. Keep source-observed masks and the final native split mask fixed for the first analysis. Attribute newborns to their actual observation histories and resets. Record primitive survival and every raw-row remapping.

First reproduce the logged native maxima and child counts exactly. A mismatch means the trace is invalid, not that the new method saves children. Then compare four packet-matched summaries in the **same final native axis layout**, before trying an eigenframe:

1. Native: historical `R_t`, historical `s_t`, historical `B_t`.
2. Rotation-only current-frame control: final `R_T`, historical `s_t`, historical `B_t`.
3. Scale-only control: historical `R_t`, final `s_T`, historical `B_t`.
4. Full frame control: final `R_T`, final `s_T`, historical `B_t`.

All four retain the same valid-observation set and max operator. They separate changing axis directions from mere scale shrinkage; their effects need not add. Current-geometry/frequency re-evaluation is a separate, more costly control for position/footprint staleness and is not required to establish an initial rotation-history discrepancy.

Report axis-maximizer timestamps, rotation changes relative to event frame, scale changes, stable-ID coverage, native versus each control's child counts, and per-observation directional demand. A quaternion sign flip is the same rotation; covariance-symmetric axis permutations/signs also need careful interpretation. Large raw quaternion differences alone are not evidence of physically meaningful drift. Distinguish rotation change from scale degeneracy and report the actual world-direction quadratic forms.

**Falsification gate:** stop if source-level parity fails; if a material count reduction comes only from scale shrinkage/observation recency rather than rotation-only alignment; if maxima already come from nearly the same final directions; or if conservative re-evaluation restores nearly all native counts. Do not lower a spectral/coverage threshold to create inflation. If there is a broad rotation-specific count discrepancy, the next necessary gate is its *native accepted-work share*, not another count plot. Under the existing opportunity standard, even optimistically removing the affected families must cover at least 10% of whole-scene accepted contributions before a new layout/quality renderer test is considered. This is a proposed gate, not a measured result.

If accepted-work data are unavailable, the cost claim remains untested. A small prospective sample must report sampling/coverage and uncertainty; it cannot establish whole-scene prevalence. Saving all observations scales with `observations × active primitives × packet size`: even six FP32 matrix entries cost 24 bytes per primitive-observation before IDs and states. No production proposal may hide this memory, transfer, envelope maintenance or recomputation cost.

## Quality, cost and novelty boundaries

The potential quality hypothesis is narrower than “fewer births are better”: false coordinate-history inflation may create unnecessary overlapping offspring, alter alpha accumulation and increase future optimization burden; removing that inflation might reduce overfit/floaters while retaining real multiview detail. None of PSNR, SSIM, LPIPS or downstream convergence follows from matrix/count agreement. The opposing hypothesis is that the historical maximum preserves useful capacity for an unsettled parent, consistent with SAD's negative directional ablation.

The cost route requires actual fewer accepted contributions, sort entries, parameter/backward work and lower total training time **after** charging packet/envelope work and any quality-recovery iterations. A count fix with a slower high-dimensional history tracker is not success. A rotated final layout is an additional intervention and must not be used to hide that a simple stale-statistic repair explains the result.

Prior art: [SAD](https://arxiv.org/html/2604.28016v1) already has multiview directional frequency, historical maxima, analytic anisotropic births and a warning about immature rotations; [SteepGS](https://arxiv.org/html/2505.05587v1) already has matrix/eigenvector split directions; [BOGausS](https://hal.science/hal-05444229/document) already has camera precision and inherited optimization history. Transporting quadratic metrics between coordinate frames is standard. Bounded arXiv queries for densification with rotation/history/transport and coordinate/frame-invariant statistics did not identify this exact historical-body-axis issue, but did surface dynamic-scene temporal densification, which concerns time-varying scene reconstruction rather than static-training iteration history. Query responses are saved in `astra_temporal_frame_prior_20260928/`. This no-match is not a novelty certificate or exhaustive source-code review.

**Current status:** valid distinct diagnostic hypothesis; source opportunity confirmed; numerical prevalence and usefulness unknown; no original deployable mechanism established. Root should record these as source facts plus explicitly labeled hypotheses and coordinate the shared D:\学习/changelog update.

## Final originality decision on root's fast gate

**NO-GO as the next original mechanism, even if a future audit finds a large numerical discrepancy.** The concrete operation currently offered is standard congruence transport/current-frame re-evaluation of SAD's stored directional observations. It is a statistical-definition correction, not an independently established algorithmic contribution. Adding a conservative directional envelope or an eigenframe does not by itself distinguish it enough from standard transported quadratic metrics, camera-precision methods and spectral splitting. Nor does removing potentially stale children independently explain improvement in all three image metrics; SAD's own directional ablation supplies a countervailing capacity-risk argument.

Do not rerun to iteration 1500, replay a 500-step GPU window, instrument a new training prefix, or build a renderer merely to establish this discrepancy. The trace above is only a specification for inspecting **already-existing** historical packets if available. A single event snapshot is insufficient; absence of packets should be recorded as an unverified implementation concern, not converted into a new experiment request. Actual correctness repairs can be handled separately if needed for baseline fidelity, with no originality claim.
