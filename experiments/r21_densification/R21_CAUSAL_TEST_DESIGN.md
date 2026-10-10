# R21 falsifiable causal experiments

Status: DESIGN_COMPLETE; MX experiments NOT_EXECUTED. Date: 2026-10-09.

First capture both untouched R20 training trajectories through step 600, with
per-step exact hashes and full states at 1,100,599,600. Distinguish post-step
parameter drift, projection, statistics, hypothetical candidate masks and the
first executed topology difference. A first hash mismatch is not a root cause.
If it falls between full snapshots, rerun from the preceding exact checkpoint
with every-step capture. If it precedes step 100, rerun from initialization.

## Same-state replay

Choose ONE post-599 checkpoint (prefer the HiGS trajectory first). Every A–F
process restores the same raw parameters, native/stable IDs, Adam groups and
moments, active SH, historical A/D/M, scene extent, seed and Python/NumPy/CPU/
CUDA RNG states. Each selects the same precomputed camera 600, records actual
K/view/GT hashes and applies the original step-600 LR update. Compare the input
digests before accepting results. Repeat using the std trajectory checkpoint
to distinguish state dependence from backend effects.

Each arm runs in a fresh process because frozen and candidate CUDA operator
registries cannot safely be mixed. A supplies detached consumer-level tensors
and final masks, not a replacement renderer or graph.

| Arm | Only intervention relative to B | Supports / weakens hypotheses |
|---|---|---|
| A | Native std_ll whole path (reference) | Defines same-state target; compare A/A for nondeterminism |
| B | Validated R20 HiGS whole path | Difference from A isolates backend behavior on common state; compare B/B |
| C | Replace radius magnitudes consumed by max_radii/tmp_radii with A radii; retain B visibility mask | Isolates direct radius consumer effects. At 600, clone/split cannot change from this alone; such a result would expose a tool/state confound |
| D | Replace only the statistics visibility mask with A mask; retain B radii and True AbsGrad | Recovery supports visibility-gated A/D update as a proximal cause. A one-step failure does not refute historical visibility bias |
| E | Replace historical A and D before adding the current B contribution; retain M, B visibility/radii/gradients | With optional native-trajectory history, recovery supports historical A/D as a mediator. Trace earlier steps to identify its origin |
| F | Replace actual clone, split and post-topology prune masks at their consumption boundaries, aligned by lineage | Recovery of downstream IDs/children/Adam supports candidate selection as the proximal source; remaining differences motivate RNG/state migration checks. F is diagnostic only |

There are two explicitly labelled E modes. On a fully common state, A's
pre-step history equals B's, so E without `--history-reference` MUST be a no-op
control. It cannot test a history difference that was removed by common-state
restoration. Supply std's original post-599 trajectory checkpoint to inject
only its A/D into the common replay state for the history experiment. That
history arose on different parameters/cameras if the trajectories drifted;
recovery demonstrates mediation, not the original cause. IDs, scene, iteration,
recipe and seed must match; do not silently pair divergent lineages.

C deliberately holds the visibility gate fixed. Replacing radii and then
recomputing visibility would alter two named factors and make C indistinguish-
able from a combined C+D intervention. At step 600 only opacity pruning is
active; the source predicts C will not restore split masks by itself.

## Outcomes that change the next action

- If A/B differ in visibility but per-Gaussian AbsGrad matches, inspect zero
  contribution rows and denominator increments on the first such step. A/D
  recurrence predicts a lower average when extra zero-signal visits are counted.
  This is a falsifiable mechanism, not an assertion that it happened on MX.
- If A/B same-state candidates match but original trajectories disagree at 600,
  inspect the earliest raw-gradient/parameter/statistics difference. The
  current renderer comparison cannot identify an earlier accumulated cause.
- If E with historical input restores split selection, compare history deltas,
  denominator counts, gradient sums, camera IDs and visibility across the
  narrowed interval. Hold parameters fixed in per-step probes as necessary.
- If small score margins explain flips, compare the observed deltas with A/A
  and B/B repeatability. Threshold sensitivity is supported only with measured
  error scale, not because the masks happen to differ near .0008.
- If scale margins cross .01×extent, examine raw scale gradients and Adam
  moments; do not attribute every split difference to the grow score.
- If F fails to align downstream topology from identical parents and RNG,
  inspect split sampling order, stable-ID lineage, actual prune row domains,
  Parameter replacement and moment migration before changing CUDA.

Optional G (common projection) and H (common raw gradients) are reserved for a
second experiment design if A–F cannot resolve the ambiguity. They are not
implemented as undocumented runtime substitutions. G must fix means/conics/
radii/depth consistently and preserve VJP meaning; H must specify whether it
replaces pre-topology raw gradients or post-topology Adam inputs. Neither is
safe to improvise by overwriting one tensor without a new contract.

## Acceptance and evidence

Save original `.pt` snapshots and SHA sidecars, `steps.jsonl`, runtime source
copies and binary hashes, all arm logs and aligned comparisons. Preserve exact
and tolerant numerical metrics separately. Require identical input raw state,
optimizer, RNG and camera for A–F; consumer substitutions must be the only
listed difference. Use both scenes and at least two repeats per backend.

No candidate restoration alone proves 5K quality restoration. After causal
evidence supports a minimal production change, rerun native True AbsGrad,
signed/raw xyz/opacity/scale/quaternion/SH parity, visibility/radii contract,
clone/split/prune, dynamic counts, 1000-step smoke, 5K quality and renderer
performance. Quality gates: PSNR delta >= -.10 dB, SSIM delta >= -.002,
LPIPS delta <= +.005. MX execution belongs to DSH; local tests certify only
the instrumentation contract.
