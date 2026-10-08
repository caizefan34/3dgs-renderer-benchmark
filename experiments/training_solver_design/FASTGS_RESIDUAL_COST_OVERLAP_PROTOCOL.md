# Native residual–cost overlap gate

2026-09-27. Frozen before measurement. This diagnoses whether the pixels with
the largest *training-image L1 residual* also incur unusually large native
accepted Gaussian–pixel work. It does not propose pruning, birth, sorting,
color sharing, or a new loss. Co-location is a necessary opportunity for a
single local edit that targets high-residual pixels and hopes to remove a
large fraction of work *in those same pixels*. A future method could improve
quality in one region and save work elsewhere; this gate does not assess that.

## Fixed source, states, views

Use author FastGS commit `44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f` on
EPIC-05, physical A100 GPU 2 if it is idle. `train.py` SHA256
`1fd700c341d109f13f95ff3a3cdf1328445f27909119e7eacadb31913e34c963`.
Use existing PLYs, no parameter mutation:

| Scene / step | Model directory | Expected PLY SHA256 |
|---|---|---|
| room / 4100 | `room_big_capture4100` | `64160970e5ea580aea4f79278a8fbe334d8d81ef6e304a903626843cc48f85bc` |
| room / 30000 | `room_big_branch_base30k` | `c2afc615fd244d561bd790e7d7721af1b7a7515222bd5385650e7b886c58516a` |
| garden / 4100 | `garden_exchange_dev_4100` | `d396092ca139fa0e6aea4d17a085ca0239bf2ff0ec2bbbe0ee3c5ab539feb5fb` |
| garden / 30000 | `garden_native_stage_profile_20260927` | `ba12264768db1e09ebdb32c8c57a6be506300d4a354c459404b520d6a0977eaa` |

Use the *first four names in order* from each scene's saved
`fastgs_score_cost_state_result.json` or `fastgs_score_cost_garden_result.json`
10-camera list. Those lists were already sampled from the author `--eval`
training camera set, before this diagnostic. Room: `DSCF4696`, `DSCF4789`,
`DSCF4956`, `DSCF4830`. Garden: `DSC07973`, `DSC08133`, `DSC08126`,
`DSC08050`. Load only those four training image files for each state; never
open formal held-out images. Use the author's image loader, effective full
width 1600, native `render_fastgs(..., mult=.5)` and black background.

## Measurements and validity checks

For each state/view, render original RGB and obtain the matching original
train-image RGB. Define pixel L1 residual `E(x,y)=mean_c |I_c-GT_c|`; this is
the per-pixel term used by FastGS's error-score ranking. It is *not* a
per-pixel decomposition of fused SSIM or the full training loss. Select
exactly `ceil(0.25*H*W)` pixels with largest `E` using sorted `topk`, with no
tuned threshold. The first `ceil(0.10*H*W)` returned indices are the nested
diagnostic top-10% subset.

Rerender the *same* unchanged model with the author's `get_flag=True`
counter, first an all-ones metric map, then each fixed top-error mask. Assert
that all counter-render RGBs match the original render within max absolute
`1e-6`, each per-Gaussian count vector has model length, the top-10 mask is
contained in top-25, and all subset counts are `<=` all-ones counts. Compute
int64 sums of accepted Gaussian–pixel counts for all pixels, top-25 and
top-10. Record original `E` sums, pixel fractions, cost fractions, stage,
view name, PLY hash, GPU, source/probe hash, and any run failure. These are
native accepted-hit work counts, not backward milliseconds or complete
training wall time. Do not substitute tile instances or projected radii.
As a descriptive cross-view proxy, retain the four top-25 count vectors in
memory for a state; calculate what fraction of all top-25 accepted hits comes
from Gaussian IDs with a positive top-25 count in at least two of the four
views. This proxy does not prove that the same 3D surface or error is shared.

## Decision fixed in advance

For each of the four states, pool the four views by summing hit counts, then
compute `f25 = top25_hits/all_hits`. The co-location gate passes only if
`f25 >= 0.40` in *all four* states: the highest-residual quarter must consume
at least 40% of accepted-hit work, i.e. twice the hit density of the other
three quarters. If any state fails, stop this **high-residual-concentrated
work** direction before proposing an algorithm or touching formal test views.
Even a pass supplies only spatial overlap, not a safe way to remove hits,
better original full-image loss, faster CUDA, novelty, or unseen-view quality.
No intervention or 30K training is authorized by this gate alone. Preserve
all eight to sixteen per-view values and failures; do not tune the quarter or
0.40 threshold after seeing results.

This screen is not a rerun of the top-64 deletion screen: no Gaussian is
deleted or fitted, and its denominator is whole-image accepted-hit work.
Nor does it revive near-color runs or residual-guided births, which have their
own negative evidence/prior art.

## Frozen execution identity

The reviewed diagnostic script is `fastgs_residual_cost_overlap_probe.py`,
SHA256 `8f43bc60880c1efdb7d798dca4dc7c963cbb912d16070c3b5278f78f07fef86f`.
The script verifies this protocol's SHA256 from `FASTGS_PREREG_SHA256` before
any image/model measurement. The exact post-freeze protocol hash is supplied
at launch and copied into the result. If a dependency failure requires repair,
record the failed attempt and an explicit pre-measurement amendment; never
silently change the thresholds or successful-run evidence.
