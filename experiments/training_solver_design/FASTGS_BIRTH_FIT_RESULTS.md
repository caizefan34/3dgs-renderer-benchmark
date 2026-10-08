# Fixed-geometry analytic birth fit: full-resolution FastGS pilot

## Verdict (2026-09-27)

The initial one-child room diagnostic almost rejected its candidate: fitted
opacity 0.00024856 and sampled fit-view MSE reduction only 0.000153%. This
was a selection failure, not proof that all native split children lack useful
geometry. A subsequent search of all 24 eligible multi-view children found a
locally useful one, but its author-rendered whole-image loss improvement was
tiny (details below). Neither result supports promotion to a training method.
No model was optimized or continued, and the offline screening cost was not
integrated into a trainer. Stop this CPU-scanned, single-birth formulation as
a strong speed-and-quality candidate; do not run a full training claim from it.

## Mathematical and implementation scope

For fixed depth order and pixel support, inserting one Gaussian with screen
kernel `G`, opacity `q`, color `c`, preceding transmittance `T`, and suffix
color `B` changes a pixel by `T G(qc-qB)`. Writing `u=qc`, the RGB response is
linear in four unknowns `(u_R,u_G,u_B,q)` under `0<=q<=0.8` and `0<=u_j<=q`.
An ideal synthetic FP64 compositor passed 10,000 identity cases with maximum
error `4.44e-16`, and recovered a known two-view insertion; this establishes
only the fixed-support algebra. FastGS's alpha cutoff, tile support, and early
termination require separate checks.

The real test used author FastGS commit
`44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f` on EPIC-05 GPU 2,
room@4100, effective 1600x1066 full-resolution training images, and the
native split-child geometry. Six specified training views provided geometry
selection scores and fit residuals; four other specified training views were
reserved for internal checks. Geometry visibility in those four was used only
to require possible check coverage, not their GT pixels for scoring or fitting.
Of 18,690 children, 24 satisfied at least three fit-view and one check-view
visibility tests. The selected child yielded 72 usable pixels in three fit
views and 59 in one check view; it projected out of the image in the other six.

On usable pixels, the maximum author-CUDA insertion-versus-analytic RGB
difference was `3.61e-7`. The four design singular values were
`[2.3655, 2.0697, 2.0697, 0.4453]`. Thus this sampled problem was not rank
deficient, but its optimum still nearly disabled the child. Fit MSE changed
`0.001189639605 -> 0.001189637786`; check MSE changed
`0.001594327263 -> 0.001593059522`. The check is one training camera and a
small set of supported pixels; it is not an untouched scene, official test,
or statistical evidence of generalization.

The script took 54.61 seconds including scene loading and candidate scan.
That number is diagnostic process time, not a training overhead estimate.
Sources: `fastgs_birth_fit_probe.py`, `fastgs_birth_fit_multiview_result.json`,
`fastgs_birth_parity_probe.py`, `fastgs_birth_parity_v2b.json`, and
`rank_one_birth_oracle.py/json`.

Cross-view residual-based densification already appears in MVGS,
GaussianPro, MVG-Splatting, and MMGS. This test makes no originality claim.
Any revived approach would need a distinct geometry proposal, larger
pre-registered local benefit, native full-resolution loss, complete cost,
matched continuation, and untouched-scene strong-baseline evaluation.

## Follow-up: all eligible children and native whole-image loss

The 24 candidates were scanned using only the same six fit training views.
To qualify for selection a child needed at least 40 stable sampled pixels in
at least two fit views and minimum design singular value at least `1e-3`;
15 qualified. The frozen choice maximized the absolute reduction in sampled
fit squared error. Only **after** this choice were the other four training
views' residuals inspected. Their geometry visibility had been used for the
earlier eligibility check, but their GT values did not choose or fit a child.

Chosen child index 1467 (parent 151525) fit to opacity `0.14867364` and
constant RGB `[1, 0.75046184, 0.26474471]`. In 179 fit pixels over three
views, sampled squared error fell by `0.11367182 / 0.59795218 = 19.01%`.
One internal check view had 43 supported pixels and fell by
`0.02981465 / 0.17122650 = 17.41%`. The maximum author-CUDA versus analytic
sampled RGB discrepancy among all scanned candidates was `3.67e-7`. This
is a real positive **local proxy**, still within one developed scene/event.

The same fixed child was then inserted into an isolated model with its fitted
opacity, DC chosen for constant RGB, and zero higher-order SH. The unchanged
author FastGS rasterizer rendered all ten 1600x1066 images. Using the author's
`0.8*L1 + 0.2*(1-fused_ssim)` loss, the mean six fit-view loss changed by
`-5.22472e-7`; mean four internal check-view loss changed by `-1.21072e-7`.
Mean PSNR changed by `+0.0000744` and `+0.0000253` dB respectively. Only
three fit views and one check view showed any image change; the other six
projected out. The largest per-view loss decrease was `1.37463e-6`. These
are single-child, fixed-state changes and do not measure a joint many-child
edit, post-birth convergence, final test quality, or net training speed.

The candidate-search diagnostic took 54.88 s and the native loss diagnostic
54.27 s, each including scene loading. Neither is an integrated event cost;
it would be invalid to subtract them directly from a 30K training wall.
An instrumented repeat selected the same child and measured 54.48 s for
scene load/native proposal, 0.143 s for ten-view visibility selection,
**4.365 s for fitting all 24 candidates on six views**, and 0.239 s for the
post-selection internal check and output, 59.23 s total. Repeating the
current CPU-fit phase at FastGS's 144 densification events would by arithmetic
alone add about 629 s before changes in scene size/candidate count; this is
not a prediction of an integrated GPU implementation. It makes the current
offline implementation plainly unsuitable for the speed target.
Current CPU scanning requires a radically cheaper, validated implementation
and a substantially larger full-image benefit before it merits continuation.
Raw results: `fastgs_birth_multi_candidate_result.json` and
`fastgs_birth_native_loss_result.json`; executable scripts with the same stems.

## Joint fixed-state check

To test whether the one-child result was merely too small, all 14 candidates
with fit-view sampled SSE gain above `1e-8` and fitted opacity above `1e-3`
were inserted together. The rule was frozen from fit views before inspecting
the four internal-check GT images. Native FastGS full-image render and the
same L1+fused-SSIM loss gave a six-fit-view mean loss change of `-7.13393e-7`
and a four-internal-check-view mean change of **`+1.90921e-7`** (worse).
Mean PSNR changes were `+0.0001408` and `+0.0000634` dB. One check camera
(`DSCF4873`) worsened in original loss by `+2.00793e-6`, outweighing the
improvements on two other check cameras. This small fixed-state result does
not establish a statistically meaningful degradation, but it fails the
precondition for claiming a robust quality-positive edit. Positive individual
sampled-MSE proxies did not transfer to an improved check-view mean in the
author's loss after joint insertion. Stop this native-child CPU-scan family
at the opportunity gate. Source: `fastgs_birth_joint_native_probe.py` and
`fastgs_birth_joint_native_result.json`.
