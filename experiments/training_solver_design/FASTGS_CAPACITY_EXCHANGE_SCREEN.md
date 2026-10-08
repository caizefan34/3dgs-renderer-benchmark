# FastGS room capacity exchange: first real fixed-state opportunity screen

2026-09-27. **Verdict:** the tested native-child, local-deletion version fails
both the meaningful whole-image quality and compute-opportunity gates. This
does not test residual-triangulated new geometry, joint scene-wide exchange,
integrated GPU scoring, matched continuation, or official unseen-view quality.

## Frozen setup

Author FastGS commit `44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f`, room@4100
PLY SHA256 `64160970e5ea580aea4f79278a8fbe334d8d81ef6e304a903626843cc48f85bc`,
effective 1600x1066 full-resolution training input, GPU 2 on EPIC-05. The
birth geometry is the previous six-fit-view-selected native split child 1467
(parent 151525). Its previous no-deletion fit set constant RGB and opacity.
Here the only new choice is a deletion near that geometry. No model is saved
or optimized. Six specified training cameras choose an exchange; four other
training cameras are used only once afterward as internal checks.

Deletion candidates use only existing Gaussian geometry and opacity: within
40 screen pixels and 15% camera depth of the child in at least two of six fit
views, projected opacity between .005 and .2, and estimated alpha-support
area 20-10,000 pixels. Among 231 candidates, the top six summed-area scores
and the child's own parent are tested. This area is a proposal proxy, **not**
measured future pixel work. For each deletion, the old opacity logit is set
to -30 in an isolated model, the child opacity/RGB is re-fitted from the
post-deletion stable-support pixels, and all six fit images are rendered with
the unchanged author CUDA and original `0.8 L1 + 0.2(1-fusedSSIM)` loss.
The minimum mean fit loss chooses the exchange. The deleted parent is included
as a contrast, not a tuned final rule.

## Result

The fit-selected deletion is Gaussian 45461; its original opacity is .1233.
Re-fitting after deletion gives child opacity .148385, RGB
`[1, .750941, .265970]`. Relative to unedited room@4100, the mean six-fit-view
original-loss changes are:

| Arm | Mean original loss change; lower is better |
| --- | ---: |
| Delete only | `+1.24e-9` |
| Birth only | `-5.22e-7` |
| Delete then re-fit birth | `-5.29e-7` |

Across the four internal-check **training** views, the respective mean changes
are `-2.79e-9`, `-1.21e-7`, and `-1.20e-7`. Exchange is marginally worse
than birth alone on those check views. The differences are far too small for
a quality claim, and four internal training views are not an independent
scene/test set. Other tested deletions mostly had fit-view effects near
floating-point loss granularity; deleting the child's opaque parent did
not produce a stronger exchange. Raw data: `fastgs_exchange_room_result.json`.

The unchanged author rasterizer generated **7,615,103 Gaussian-tile
instances** across the ten views for the base model. Relative totals were
delete-only `-44`, birth-only `+21`, exchange `-23`, or a net structural
reduction of just `0.000302%`. This is not accepted pixel work, optimizer
cost, or measured GPU speed. It shows this particular pair has no plausible
large speed opportunity. Raw data: `fastgs_exchange_cost_result.json`.
The two independent offline processes took 57.69 and 50.79 seconds including
scene loading; neither is an integrated event-cost estimate.

## Decision and remaining scope

Stop this **native-child plus nearby-delete** formulation. It inherited the
known tiny single-child whole-image effect, and the tested deletion did not
change quality or tile work enough to help. Do not run 512-step or full 30K
branches for this pair, and do not describe the result as evidence that a
different residual-triangulated geometry or larger exchange program works.
The broader capacity-exchange design remains an unverified technical idea,
but a subsequent full-text prior-art check found that 3DGS-MCMC and
Metropolis-Hastings Sampling for 3D Gaussian Reconstruction already relocate
under-contributing Gaussians, with the latter also using multi-view residual
importance and approximate loss-aware proposal acceptance. Thus the broad
birth/death relocation framing is not an original strong-method candidate.
The narrower post-deletion exact visibility fit did not show a compelling
whole-image or cost effect here; stop this route as the current main line.

An author-protocol garden development checkpoint was also generated at 4100
steps in the isolated checkout under `output/garden_exchange_dev_4100`, with
4,111,216 Gaussians, PLY SHA256
`d396092ca139fa0e6aea4d17a085ca0239bf2ff0ec2bbbe0ee3c5ab539feb5fb`,
and reported training-loop time 52.2515 s. It has **not** been used to test an
exchange and is not a garden validation result. The 955 MiB PLY stays on
EPIC-05; no reason to run the failed room proposal there.
