# FastGS native trajectory stage profile

2026-09-27. Diagnostic, not a candidate or a speed claim.

The older multi-stage tables under `mechanism_discovery_v2` include gsplat
and estimated backward costs. They cannot supply a multi-stage FastGS cost
denominator. The pinned FastGS evidence has a 44-step room@4100 profile and
multi-state scorer timings, but lacks ordinary late-stage and garden profiles.

Run the unchanged author training computation from initialization for room and
garden, one currently idle A100 per run. Pin source commit
`44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f`; use each scene's `train_big.sh`
options, `--eval`, original effective width 1600, original loss, 30K iterations.
No lower-resolution phase, alternate camera order, parameter intervention or
evaluation-based selection. Save the final author PLY. The output paths are new.

Instrument iterations 4096–4223, 10048–10175, 16000–16127 and
24064–24191 (128 steps each). Record nested CUDA/CPU phase totals for native
forward, original loss, backward, optimizer and topology. Record all raw
profiler rows, actual optimizer call counts, N and synchronized iteration wall
times. At late stages a 128-step window covers four 32-view or two 64-view
optimizer cycles. Early windows deliberately include topology events; report
their costs separately. Keep profiler overhead explicit. Instrumentation can
perturb wall times; this is not an uninstrumented complete-training baseline.

Use disjoint named phase totals or device-kernel totals, never their sum.
Compare measured stage fractions before selecting a new mechanism. A candidate
must have a credible route to removing >=15% of complete ordinary step time
(after its own overhead), and a separate, testable full-image quality mechanism.
Merely eliminating a phase whose share exceeds 15% is not evidence it is
eliminable. Phase windows do not establish complete 30K savings, novelty or
formal unseen-view quality. Retain all previous failed screens.

Analysis accounting clarification after the first collected window: Kineto
emits both CPU and CUDA annotation ranges named `phase.*`. The CUDA ranges
are not kernels and include children/idle spans. Exclude device rows sharing
CPU event names from the leaf-kernel sum. The autograd engine runs on another
CPU thread, so `phase.backward` does not contain its full device work; use
the CPU `_RasterizeGaussiansBackward` operator's attributed device total and
report SSIM backward separately. The analyzer initially rejected duplicate
phase keys before any numerical conclusion; it now enforces these rules and
checks that the disjoint attributed phases do not exceed the leaf CUDA total.
This corrects measurement accounting, not a candidate acceptance threshold.
