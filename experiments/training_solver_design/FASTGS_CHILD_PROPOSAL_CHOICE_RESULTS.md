# FastGS child-position choice: one-event room pilot

## Verdict (2026-09-27)

The frozen ten-train-view, first-order per-family choice is **not a strong
speed-and-quality candidate**. In a 4,099→4,612 room continuation it kept the
same 9,345 native splits and 654,429 surviving Gaussians, and raised the mean
39-view development PSNR by only +0.013601 dB versus a matched fresh control.
The descriptive paired-view 95% interval was [−0.000092,+0.027860] dB.
SSIM changed +0.000054 and LPIPS −0.000069, with both intervals crossing zero.
Against the prior seed-20260927 control, the 512-step PSNR gain was only
+0.006384 dB; against the prior seed-20260928 control, LPIPS was worse by
+0.0000265. These room views were already used for development and cannot
confirm generalization.

The two branch process walls were 66.41 s (control) and 64.98 s (choice), but
the offline two-proposal, ten-view scoring was excluded from both. Its already
measured render/adjoint/gate components sum to 323.15 ms across two separate
diagnostic jobs; that excludes camera and proposal construction, startup and
data loading. The choice branch's built-in training time was slightly slower
(4.1492 vs 4.1373 s). Startup/load variation dominates these short process
walls. Thus **no net training speedup has been demonstrated**. Do not scale
this policy to a 30K run without a new, materially larger opportunity case.

## Frozen protocol and checks

- Author FastGS commit `44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f`, GPU 2
  on EPIC-05; same room 4,099 checkpoint, author loss/optimizer, effective
  1600×1066 training input, camera RNG, one native 4,100 event and then no
  further densification.
- The rule chooses, for each parent, both children from the proposal with
  smaller mean fixed-support loss VJP over the ten prespecified training views.
  It chooses seed 20260928 for 4,061 families and seed 20260927 for 5,284.
  It does not use the 39 development views for the choice.
- Frozen parent IDs and both complete sample bytes matched the preceding
  diagnostic. Proposal sample SHA256 values: `185bd364a5eae3638bbf45099bfc7a69b62d70d20d743364ff8b587be88b2b40`
  and `be02252c9a83e6f36f9cd67b47862e8662379190bb42e0296f4252d217327046`.
  Policy SHA256: `6db726c90e6636eb56dd678cabd557fa46fb37089a0b75ece7f48617fea1300f`.
- An initial attempted control was stopped by the split-parent assertion:
  the original `--highfeature_lr 0.02 --grad_abs_thresh 0.0004` parameters
  had been omitted. Both successful branches used these author-run parameters.
- The fresh control reproduced split IDs, full child-sample hash and point
  counts, but its PLY bytes differed from the earlier seed-20260927 control.
  Its 4,612-view mean PSNR was 29.813139 vs 29.820356 previously. Therefore
  the fresh pair is primary, with both historical proposals as sensitivity
  checks; a few thousandths of a dB must not be treated as robust benefit.

| Step | Arm | PSNR | SSIM | LPIPS |
| ---: | --- | ---: | ---: | ---: |
| 4,164 | fresh control | 28.725582 | .887585 | .268418 |
| 4,164 | choice | 28.733552 | .887655 | .268216 |
| 4,612 | fresh control | 29.813139 | .902481 | .247281 |
| 4,612 | choice | 29.826740 | .902535 | .247212 |

At 4,612, choice improved PSNR in 24/39 development views, SSIM in 22/39,
and LPIPS in 21/39. The paired-view intervals are descriptive only because
views are correlated, one scene/event was used, and there is no untouched
confirmation set. The better summed one-step VJP of the selected proposals
is not a measured long-horizon gain.

## Reproducible evidence

Local scripts and data: `fastgs_child_proposal_choice.py`,
`fastgs_child_proposal_branch.py`, `fastgs_child_proposal_eval.py`,
`fastgs_child_proposal_analysis.py`, their `*.json`, logs and `.time` files.
The four fresh PLY SHA256 values at 4,164/4,612 are control:
`4f9c649c…6630b29` / `ccd319b8…57bfab`; choice:
`e8f313c8…15dd45291` / `02e92906…41b2`.
Remote PLYs remain under `output/room_child_choice_{control,choice}_4612`
in `/root/codex_fastgs_audit_20260927`.
