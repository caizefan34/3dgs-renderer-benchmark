# FastGS-Big barrier audit: fixed-resolution quality and complete training cost

2026-09-27. Status: source audit, complete room baseline, live-selection
diagnostic, and paired 4,099 -> 30K capped-score trial completed. The cap32
rule failed the joint faster-and-better gate. No new method, quality gain, or
originality is proven.

## Reproducible baseline setup

- Author FastGS source: `44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f`.
- Isolated remote checkout: `/root/codex_fastgs_audit_20260927` on EPIC-05;
  `.venv` inherits PyTorch 2.9.1+cu128 and uses separately installed FastGS CUDA
  extensions. This differs from the author's Python 3.7 / PyTorch 1.12 / CUDA
  11.x environment, so published minutes are not directly comparable.
- Room data: `/root/3dgs-renderer-benchmark/data/datasets/mipnerf360/room`.
  Command follows the room row of `train_big.sh` with `--eval`, 30K updates,
  `--highfeature_lr 0.02` and `--grad_abs_thresh 0.0004`.
- FastGS default `-r -1` caps source images wider than 1600 pixels at 1600.
  Every training step uses this fixed effective resolution. A rival must use
  identical effective input resolution for a causal comparison; a separate
  `-r 1` arm is needed to study true native-resolution behavior.

## Source-level cost and semantics

`utils/fast_utils.py:45-105` samples ten cameras for each score call and runs
two extra forward renders per camera: one makes the residual map, one counts
high-error pixel/gaussian interactions. `train.py:127-158` calls the score on
each 100-step densification event from 600 to 14,900 inclusive (144 calls),
and at 18K/21K/24K/27K pruning (4 calls). Under default 30K settings this is
2960 additional forward-render calls. This is a call count, not a time ratio;
the first and second score render can have different work and no backward.

The actual densification gate is `floor(sum_v c_v / 10) > 5`, where `c_v` is
the number of high-error pixels touched by a Gaussian in sampled view `v`.
Thus a single view with at least 60 such pixels can pass, while ten views
with five each do not. This gate aggregates pixel evidence across views; it
does not require a minimum number of distinct supporting views. This is a
mathematical property of the pinned code, not evidence that FastGS makes bad
choices or that a stricter view vote improves generalization.

FastGS paper's own K-view ablation reports 10/20/50 sampled views with
Mip-NeRF 360 PSNR 27.56/27.55/27.55 and LPIPS .261/.261/.262. Merely raising
the number of scored views is not a supported quality strategy.

## Early timing gate

An unmodified author-code room-Big smoke ran 600 steps and reported 133,950
Gaussians and 3.8743 seconds of its built-in `Training time`. An isolated
700-step run wrapped only `compute_gaussian_score_fastgs` with CUDA synchronize
and Python wall timing. Its two events cost 0.05139 and 0.03438 seconds on
112,627 and 134,030 pre-event Gaussians; total reported training time was
4.56845 seconds. The measured score fraction was about 1.88% for this early
run. The wrapper adds synchronizations, and scene loading/saving are outside
FastGS's built-in timer; this is neither a full-training speed ratio nor a
projected savings claim. It rules out treating elimination of early score
renders alone as an already strong acceleration.

Diagnostics: `fastgs_score_profile.py` and `fastgs_count_concentration.py`.
Remote logs: `logs/room_big_smoke600.log`, `logs/room_big_profile700.log`.

## First complete baseline and real-state count distribution

The pinned author `train.py` completed room-Big 30K on an A100-SXM4-80GB:
reported `Training time: 231.7975746` seconds and 572,546 final Gaussians.
Its timer excludes scene loading, final file writing, rendering, and metric
evaluation. The author `render.py` produced 39 images at 1600x1066, and
`metrics.py` reported PSNR 32.042473 dB, SSIM 0.9298893, LPIPS 0.1887548.
Its first metric attempt failed because the LPIPS calibration weight was not
cached; a byte copy of the installed `lpips` package's v0.1 `vgg.pth` was
placed in the expected torch cache, a zero-vs-zero LPIPS smoke returned 0,
and the unchanged author metrics script then completed. One scene/seed cannot
establish cross-scene superiority or compare against an unmatched gsplat run.
Raw local evidence is under `fastgs_evidence/`: training log, author metric
JSON and per-view JSON, saved `cfg_args`, scoring logs, and image SHA manifest.
Remote originals remain in `/root/codex_fastgs_audit_20260927`.

Provenance: room COLMAP `cameras.bin` SHA256
`affcfac9e8af524b36c6acdffa6d0bb19a1c3c05833f40d2453e28083058b0ba`;
`images.bin` SHA256
`92833c78b3f0ddc41e7027b573e201f2527b9945a0d693f3e866f4769d6f03c4`;
the SHA256 manifest of 311 training/source images hashes to
`18c1ded03c4de72c22c2b8f0ad57e80adaa0c93ab4aae592a57924a06b281b09`.
LPIPS v0.1 VGG calibration weight SHA256 is
`a78928a0af1e5f0fcb1f3b9e8f8c3a2a5a3de244d830ad5c1feddc79b8432868`.

At the saved *post-event* room@600 state, the original score renderer was run
on ten training cameras sampled with seed 20260927. The metric gate
`sum_v c_v >= 60` passed 79,738 / 112,627 Gaussians. Of those, 63,293 had
positive projected radius in at least three sampled views; 10,348 / 63,293
(16.35%) had effective error-evidence view count below 1.5, where
`V_eff=(sum_v c_v)^2/sum_v c_v^2`. Among all metric-gate passers, 21,840 / 79,738
(27.39%) obtained at least 75% of their counted hits in one view. The median
`V_eff` was 2.09. These are candidate score distributions, not a claim that
single-view concentrated Gaussians are wrong. Projected radius is only a
frustum-footprint proxy; PLY lacks the live gradient qualifiers, and the
post-event state is not the pre-event selection state. Raw diagnostic:
remote `logs/room_big_600_count.jsonlog`.

The next diagnostic wrapped the **unchanged author scorer** during a fresh
room-Big trajectory through 5K. It captured ten per-view count arrays at
each of 45 real densification events and independently reproduced both live
gradient/scale qualifiers. The first event selected 21,424 split/clone
parents; the last selected 8,630. Across all events there were 738,697
*selection occurrences* (not unique Gaussians). Among 486,569 occurrences
with positive projected radius in at least three sampled cameras, 111,322
(22.88%) had `V_eff<1.5`; 321,215 / 738,697 (43.48%) had at least 75% of
high-error pixel hits from one camera. The late event group (31-45) rose to
27.60% concentrated among those projected in at least three cameras. This
proves the concentration survives the actual FastGS gradient filters on this
room trajectory. It does **not** prove those points are redundant or that a
different rule improves held-out quality. `radii>0` overcounts physical
visibility, and the instrumented run's extra capture/synchronization means
its 46.39-second 5K time must not be used as the unmodified baseline.
Evidence: local `fastgs_evidence/room_big_live5k.log` and
`analyze_fastgs_live_selection.py`.

## Preregistered falsification gate (completed below)

The source-level and live-selection opportunity gates now pass for room.
Next create an optimizer-complete checkpoint at 4,099, then compare paired
continuations that retain the author's cameras, per-step loss, renderer,
pruning score and **the exact number of clone and split parents at each event**.
The single preregistered test rule is `S_g=sum_v min(c_v(g),32)` with separate
top-K selection among gradient-eligible clone/split parents; the control is
the author's `sum_v c_v >= 60` gate. The per-view cap prevents one large
residual patch from dominating all ten-camera evidence; 32 is a fixed
research hypothesis, not a tuned optimum. Count its sorting/capture cost,
evaluate the same unseen cameras without choosing parameters on them, and
reject if quality does not improve or total training time is not competitive.
This rule is not yet established as original; FastGS already has multi-view
scoring and other work may use per-view robust aggregation.

## Paired 4,099 -> 10K pilot (development scene only)

The author trainer's `--checkpoint_iterations` is parsed but has no save call.
`fastgs_capture_checkpoint.py` therefore wrapped `GaussianModel.optimizer_step`
to save `capture("default")` immediately after step 4,099, before the step
4,100 densification event. The 450 MiB checkpoint SHA256 is
`d91ba09a5ab040373cc59a93b6864949cb3cc1b5d5b2dea17c79223fef7e927b`;
it contains 651,547 Gaussians and both optimizer state dicts. PyTorch 2.9's
default `weights_only=True` rejected the author's unqualified `torch.load`
because the author state contains a NumPy scalar. Both branches use the same
locally generated checkpoint with `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`, which
changes only trusted checkpoint deserialization. The failed first load log is
retained remotely and was not counted as a training result.

The original author score and fixed-cap score continued from this checkpoint
through step 10,000 on the same A100 and dataset. The candidate wrapper
checked **all 60 densification events**: at each candidate-branch state, its
clone and split parent counts matched what the original FastGS rule would
have selected **on that same state**. After the first edit the two branches
diverge, so their actual future event counts need not match each other. Across events,
342,426 parent selections were possible under the baseline rule and 239,163
overlapped the cap rule, so the candidate replaced 30.16% of selected parents
in event-occurrence terms. This does not imply identical future Gaussian
counts because the selected identities alter subsequent training and pruning.

| 4,099 -> 10K branch | built-in training s | full process wall s | N@10K | test PSNR dB | test SSIM | test LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| author score | 55.4928 | 118.60 | 689,326 | 30.355038 | .9134763 | .2176106 |
| cap32 score | 55.4278 | 117.86 | 673,453 | 30.439732 | .9140022 | .2167515 |

The 39 official held-out room views are **development data** for this pilot;
they must not be reused as an untouched confirmation set after seeing these
results. The candidate's aggregate deltas are PSNR +.084695 dB, SSIM +.000526,
LPIPS -.000859, with 24/39, 26/39 and 27/39 views respectively improving.
Median per-view PSNR delta is only +.031792 dB, and extremes are -1.046 to
+2.033 dB, so the average is sensitive to a few cameras. The timing difference
is sub-second and not a speed result. A no-edit replay and full 30K/multi-scene
validation are required before attributing even the quality delta to the rule.
Raw branch logs, time files, metrics JSON and per-view JSON are copied to
`fastgs_evidence/`.

An unmodified replay from the same checkpoint was not bitwise identical:
its 10K PLY SHA differed and it finished with 690,403 rather than 689,326
Gaussians. Its 39-view metrics were PSNR 30.346371, SSIM .9133779,
LPIPS .2180378, giving replay-minus-control differences of -.008667 dB,
-.0000984, and +.0004272. The capped rule's pilot deltas exceed this one
replay in magnitude, but a single replay is not a variance estimate or a
formal paired confidence interval. `fastgs_evidence/` retains both metric
sets and raw logs. Continue to 30K before deciding whether this candidate
is worth multi-scene confirmation.

After that, require complete 30K time/quality curves, official held-out views,
multiple scenes/seeds, same data/resolution/software, FastGS-Big and current
other strong baselines, and a full-text/source prior-art audit. No result here
satisfies those requirements.

## Paired 4,099 -> 30K cap32 verdict (room development scene)

Both continuations used the same optimizer-complete 4,099 checkpoint, FastGS
training loss and renderer, the same effective 1600x1066 resolution, and saved
10K/20K/30K models. The rule changed only parent selection at densification
events. At each candidate-branch event, clone and split parent counts were
matched against the author rule evaluated on that branch's current state;
the two diverged branches were not forced to have identical later point counts.

| Branch | Built-in training s | Full process wall s | N@30K |
|---|---:|---:|---:|
| Author score | 192.0766 | 262.40 | 566,856 |
| cap32 score | 197.9282 | 267.66 | 567,789 |

The cap rule took 5.852 s (3.05%) more built-in training time and 5.26 s
(2.00%) more full-process wall time. One run per branch does not estimate
timing variance, but neither measured timer shows a speed gain.

| Step | Branch | PSNR dB | SSIM | LPIPS |
|---:|---|---:|---:|---:|
| 10K | Author | 30.400068 | .9136015 | .2174812 |
| 10K | cap32 | 30.327240 | .9137385 | .2170839 |
| 20K | Author | 32.020439 | .9291240 | .1912480 |
| 20K | cap32 | 31.981611 | .9299193 | .1893839 |
| 30K | Author | 32.229950 | .9302623 | .1887598 |
| 30K | cap32 | 32.220028 | .9310054 | .1871286 |

At 30K cap32 is -0.009922 dB PSNR, +0.000743 SSIM, and -0.001631
LPIPS; 21/39, 27/39 and 32/39 views respectively improve. The median
per-view PSNR change is +0.006401 dB despite the aggregate PSNR decline,
so the image metrics have a real tradeoff rather than uniform improvement.
The earlier independently stopped 10K pilot (+0.084695 dB PSNR) did **not**
predict this complete trajectory's 10K result (-0.072828 dB). Its test views
were already examined during development, so these 39 views cannot serve as
an untouched confirmation set for cap32.

**Decision:** retire the simple `sum_v min(c_v,32)` parent ranking as a
joint speed-and-quality candidate. It increased training time and did not
improve all image-quality metrics; a modest SSIM/LPIPS gain does not satisfy
the user's combined target. This single-scene result does not establish that
all view-robust scoring is ineffective. Raw training logs, wall time files,
aggregate `results.json`, and three-step `per_view.json` copies are under
`fastgs_evidence/`; the source and checkpoint provenance are above. Future
claims require untouched scenes/views, multiple seeds, matched full-cost
curves and a prior-art audit.

For the strong-method audit, [ACE-GS](https://arxiv.org/abs/2606.21244)
(2026) is also relevant: its indexed abstract claims momentum-consistency
densification, sensitivity pruning and cross-dimensional residual-frequency
compensation, with up to 3.7x acceleration against Speedy-Splat and up to
0.89 dB PSNR improvement over original 3DGS in its own comparisons. This is
an author abstract, not a reproduced or protocol-matched result. Its full
method, code availability and metric protocol must be checked before any
novelty or competitive-performance claim for our candidate.
