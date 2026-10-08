# Full-resolution FastGS split-event VJP and exploratory veto pilot

2026-09-27. Status: a real-camera all-family diagnostic passed endpoint checks;
two room child-seed split-event veto pilots produced small development-view
quality signals but did **not** establish faster complete training, cross-scene
quality or novelty.
The half-resolution main-training route remains outside this project goal.

## Fixed event and all-family GPU check

The unchanged author FastGS commit is `44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f`.
The room@4100 pre-event model has 651,547 Gaussians. Its PLY SHA256 is
`64160970e5ea580aea4f79278a8fbe334d8d81ef6e304a903626843cc48f85bc`;
the native selected-split mask SHA256 is
`d2a393e0513f71bbd19996a9c5044da9dde7faf11740939f845b35c9b7c8b0a7`.
The same 9,345 selected parents and their 18,690 author-rule sampled children
were inserted into one full-resolution 1600×1066 camera's native projected,
depth-sorted tile list. `fastgs_event_gate_vjp_cuda.cu` computes a separate
fixed-support optical-depth gate VJP for every family; it is a diagnostic,
not a modified author renderer.

At `g=0`, maximum whole-image RGB error against the native pre-split render
was `1.788e-7`; at `g=1`, maximum error against native post-split was
`2.384e-7`. Zero pixels exceeded the predeclared `2e-5` endpoint threshold.
Three pixels' family VJPs differed from independent FP64 autograd by at most
`1.50e-14` absolute. The test does not independently check all 9,345 family
VJPs at all pixels. Of them, 4,556 had nonzero VJP on this view, split into
2,314 negative and 2,242 positive values.

The union list had 893,886 Gaussian–tile instances, versus 866,228 before
split and 872,713 after native split. It touched 4,320/6,700 tiles and
1,101,984/1,705,600 pixel slots (64.6%). In repeat runs the **separate**
all-family full-image diagnostic pass took 3.686–4.411 ms per invocation.
A same-phase native forward/loss/backward pair on this camera took median
6.166 ms before split and 6.296 ms when parent and children were all kept at
raw opacity. The latter is only a support-work comparison; it is not a
correct gated training step. One-shot child construction took 47–64 ms in
these three local runs, which is noisy and excludes repeated training.
Therefore adding a separate 3.7–4.4 ms gate pass on **every** probation step
is not a credible speed path without integration or measured later savings.
The author backward already forms per-layer `dL/dalpha`; a fused gate VJP
could reuse it, but its candidate-support, early-stop, parameter-gradient
and total-cost implementation remains unverified.

Raw event runs: `fastgs_event_gate_vjp_result.json`,
`fastgs_event_gate_vjp_v2_result.json`, and
`fastgs_event_gate_vjp_v3_result.json`, with matching `*.log` files.
The final local/remote event probe SHA256 is
`284485592cefe4923edd7056a1e44eb0a3a8df84813e6ddcf594565bd4534190`;
CUDA source SHA256 is
`9e5e774a26c8a347478f041fc53fd5ee41df656d4137aed44c44155041ea7496`.

## Ten training-view scores are inconsistent

Using fixed training-camera indices `0,30,...,270`, the same candidate child
sample was scored with the unchanged full-resolution L1 + fused D-SSIM
objective. All ten `g=0` images matched native RGB within `2.384e-7`.
The ten-view pass measured 54.0 ms for native base render, 25.0 ms for its
loss adjoints, 22.2 ms for union render, and 37.4 ms for the separate gate
kernel; these one-off wall timings exclude camera load and construction and
are not an optimized integrated policy cost.

At an absolute loss-VJP threshold of `1e-10`, 7,079 families had meaningful
signal in at least two views; 4,926 of those (69.6%) had opposite signs in
different views. Leaving out any one of the ten views changed the mean score
sign for 2,819/8,057 families with meaningful mean. In 1,000 resamplings of
these ten cameras, only 1,017 mean-negative and 1,831 mean-positive families
had at least 95% sign stability; 5,209 meaningful-mean families did not.
The opposite-sign count remains 4,379/6,669 at the stricter `1e-8` threshold.
This is evidence against deciding from one view, not proof of true split
utility or of cross-scene stability. Files:
`fastgs_event_gate_multiview_scores_20260927.pt`,
`fastgs_event_gate_multiview_result.json`,
`fastgs_event_gate_score_analysis.json`, and their scripts/log.

## Common-state 64/512-step exploratory veto

We froze a narrow hypothesis before training: veto the 1,831 families whose
ten-view mean VJP was positive above `1e-10` in at least 95% of the 1,000
camera resamples, while keeping all other native splits and clones. The
`control` and `veto` branches restarted the identical 4,099-step checkpoint.
At step 4,100 they selected the same 9,345 original split parents; both
sampled the entire child proposal tensor once with seed `20260927` and SHA256
`185bd364a5eae3638bbf45099bfc7a69b62d70d20d743364ff8b587be88b2b40`,
then the veto branch discarded only the prespecified families. Subsequent
densification was disabled to isolate this one event. Each training step
retained the author's 1600×1066 input, loss and optimizer, and the two runs
used the same camera RNG. Source: `fastgs_gate_score_4100_branch.py`.

On the same 39 room development test cameras, direct FP32 native renders
with standard whole-image PSNR and VGG LPIPS v0.1 gave:

| Step | Arm | Gaussians | PSNR | SSIM | LPIPS |
| ---: | --- | ---: | ---: | ---: | ---: |
| 4,164 | control | 654,429 | 28.725807 | .887588 | .268422 |
| 4,164 | veto | 652,539 | 28.733999 | .888034 | .267492 |
| 4,612 | control | 654,429 | 29.820356 | .902534 | .247214 |
| 4,612 | veto | 652,539 | 29.829924 | .902656 | .247111 |

At 64 steps the veto-minus-control mean differences were +.008192 dB PSNR,
+.000446 SSIM and −.000930 LPIPS; 27/39, 37/39, and 36/39 views improved.
At 512 steps they were +.009568 dB, +.000121 and −.000103; only 18/39,
24/39 and 27/39 improved respectively. A descriptive paired-view bootstrap
gave a 95% interval of [−.00731, +.02747] dB for the 512-step PSNR mean and
[−.000248, +.000054] for LPIPS. Views are correlated, one scene and one
child seed were used, and room views had already been consulted during
development: these intervals are **not** cross-scene confidence intervals.
The first evaluation accidentally averaged per-channel PSNR; its raw output
is retained as `fastgs_gate_branch_64_512_eval_per_channel.json`, but the
table and conclusions use the corrected whole-image PSNR in
`fastgs_gate_branch_64_512_eval.json`.

The two 512-step processes took 68.59 s (control) and 67.85 s (veto) complete
wall; their built-in training timers were 4.142 s and 4.291 s, respectively.
The process order, image-loading dominance and lack of repeats make the
subsecond complete-wall difference uninformative. **The offline ten-view
screening cost is excluded from both branch runs**, so even a stable wall
difference would not establish algorithmic speed. Raw logs/time files and
the four PLYs remain on EPIC-05 under
`output/room_gate4100_{control,veto}_4612`; local metric/analysis JSONs and
training logs preserve the paired evidence. The four PLY SHA256 values at
4,164/4,612 are, control:
`c72ef312cfd6bfda125e297fdb16da8905a7843c57d853058d39d84f7fd85024` /
`1bc820428b9f524b403381e8af138efef048c2d75ca8fe80ea23d00583d49280`;
veto:
`0d624f98d424fb5254de5459c94b73d3ae2c7a2cd0e40a55553a0449bfe79135` /
`574c74e3bbb49316f3104e30257a46d3463b7fb5addf5906e327d52a2c86c43a`.

## Decision

The separate per-step probation renderer fails to establish an affordable
path. A one-shot event score has a weak short-horizon quality signal, but its
multi-view signs disagree often and its 512-step quality differences are small.
The second child-seed audit below further constrains that conclusion.

## Frozen policy, second child seed

With seed `20260928`, the parent mask and ten cameras were unchanged. The
independent full proposal sample SHA256 was
`be02252c9a83e6f36f9cd67b47862e8662379190bb42e0296f4252d217327046`;
the scoring script and both training arms used exactly this sample, verified
by `fastgs_child_sample_identity.py`. Recomputing the same frozen 1,000-draw
camera-bootstrap policy vetoed 1,855/9,345 native splits. Candidate child
positions matter greatly: among 8,022 families with meaningful mean VJP in
both seeds, 3,783 changed score sign; the stable-positive lists intersected
in only 620 families (Jaccard .202). This does not imply the score is wrong,
because the child proposals themselves changed.

The same common-checkpoint, full-resolution, one-event-only 64/512-step
protocol gave the following standard test-image mean veto-minus-control
differences (the 39 room views remain development views):

| Child seed | Horizon | PSNR dB | SSIM | LPIPS | PSNR better views |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20260927 | 64 | +.008192 | +.000446 | −.000930 | 27/39 |
| 20260927 | 512 | +.009568 | +.000121 | −.000103 | 18/39 |
| 20260928 | 64 | +.025230 | +.000460 | −.000761 | 34/39 |
| 20260928 | 512 | +.003470 | +.000064 | −.000054 | 24/39 |

The seed-28 512-step descriptive paired-view bootstrap intervals crossed
zero for PSNR [−.00993,+.01710] dB, SSIM [−.0000108,+.0001391], and LPIPS
[−.0001968,+.0000943]. Complete processes took 64.58/70.83 s
(control/veto) in seed 28, with built-in training timers 4.143/4.174 s;
the opposite complete-wall ordering from seed 27 shows that startup/load
variation dominates this short test. Both seeds excluded offline policy
scoring time from branch wall. Raw second-seed evidence is in
`fastgs_event_gate_multiview_scores_20260928.pt`,
`fastgs_event_gate_seed_comparison.json`,
`fastgs_event_gate_score_analysis_20260928.json`,
`fastgs_gate_branch_64_512_eval_20260928.json`,
`fastgs_gate_branch_eval_analysis_20260928.json`, and their scripts/logs.
The second-seed four remote PLY SHA256 values (control 4,164/4,612, then
veto 4,164/4,612) are
`2f3a083bf27396fde056ce1c80bc04bdc362846467259e503809dc73bfec31b5`,
`a1a9459f4504a6c53e3c64a51bc8decd0566664c874b0863a1d2f63bb9f864da`,
`5442d1763b24674d343f019ae7d6610979db3856128b2668025a705141530ad0`,
`3f26c45e17d6e72f7a4575d5bf55ae25d74527f2af867c9d2cedd43469e5b825`.

Decision after two seeds: the veto policy has a reproducible **direction** of
small room short-horizon mean improvements, but the 512-step effect is too
small and uncertain, while no net speed benefit is measured or even accounted
for. Deprioritize this one-shot veto as a candidate for the user's strong
joint speed/quality target. Do not scale it to 30K or advertise it as novel
without a new mechanism that materially improves the quality-time margin.
The fused persistent-gate hypothesis is a separate, unimplemented avenue.
