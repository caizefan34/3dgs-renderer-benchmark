# P16-RB — 2026 Baseline Reset + Bottleneck Census

**Date:** 2026-09-27  
**Status:** preregistered before EPIC-05 execution.

## 0. Purpose

P13–P15 ruled out several optimization ideas on the frozen P12 C0 baseline. Before proposing another mechanism, this phase establishes what the **current 2026 strong baseline** actually costs on the same EPIC-05 A100 hardware.

P16-RB is a benchmark/census phase, not a new method.

No new optimization mechanism may be introduced during this phase.

## 1. Questions

Q1. On identical hardware and datasets, which modern public 3DGS systems are on the quality/time Pareto frontier?

Q2. For the strongest systems, is remaining training cost dominated by:
- per-iteration execution, or
- convergence / density-control trajectory / number of optimization steps?

Q3. What baseline should future “>3× strong-baseline” claims use?

## 2. Frozen methods

Use `BASELINE_PINS.json` exactly.

Methods:

1. `P12_C0`
2. `FasterGS_CVPR26_main`
3. `FasterGS_CVPR26_fused`
4. `FastGS_CVPR26`
5. `SADGS_SIGGRAPH26`

TurboGS is literature context only in this phase because no verified official code pin is frozen here. It may be added later under a new preregistration if an official implementation is identified.

## 3. Primary scenes

- room
- bicycle
- garden

No scene substitution after results.

## 4. Two comparison tracks

### Track N — native method

Run each public method using its official/native scene recipe at the pinned commit.

Do NOT force all methods to:
- the same optimizer,
- the same density-control schedule,
- the same iteration count,
- the same loss.

That would destroy method semantics.

Record all differences.

### Track C — common evaluation

Regardless of native training loss, final rendered test images are evaluated with one common metric implementation:
- PSNR
- SSIM
- LPIPS

Use the same test-camera split and the same GT images.

If the native method uses a different train/test split, stop and resolve the split before comparison.

## 5. Resolution comparability gate

For every method/scene record:
- source image dimensions,
- actual training/render dimensions,
- scale factor.

Direct wall-time ratios are marked `DIRECT_COMPARABLE=true` only if test render dimensions match the common P12 dimensions exactly, or differ by <=2% pixels due only to rounding.

Otherwise preserve native results but do not use them as a direct speed denominator until a semantics-preserving common-resolution rerun is made.

## 6. Stage 0 — build and semantic preflight

No full benchmark before each method passes:

- exact git commit pin;
- recursive submodule identities saved;
- environment manifest saved;
- one training scene loads correctly;
- one forward/backward/optimizer step succeeds;
- native render/eval succeeds;
- train/test split identity checked;
- output resolution recorded;
- no hidden multi-GPU use.

For Faster-GS, pin NeRFICG as specified in `BASELINE_PINS.json`.

Any source patch required merely to build on EPIC-05 must:
- be minimal,
- be stored as a diff,
- not alter algorithm/training semantics,
- be explicitly marked `BUILD_COMPAT_PATCH`.

Algorithmic patches are forbidden.

## 7. Stage 1 — room pilot

Run one clean full native `room` training for each external method.

P12 C0:
- reuse the existing frozen result only if an untimed 500-step timing replay on the same EPIC-05 A100 differs from the archived P12 median iteration time by <=5%;
- otherwise rerun room C0 fully.

Pilot records:
- training wall clock excluding evaluation/rendering/checkpoint I/O where the method itself already excludes them;
- separately record end-to-end command wall clock;
- native iteration count;
- final # Gaussians;
- peak VRAM;
- final common metrics;
- training resolution;
- crash/retry records.

If a method cannot reproduce a sensible native room result, stop that method before bicycle/garden and return `METHOD_PREFLIGHT_FAIL`.

## 8. Stage 2 — primary triad

Only methods passing room pilot run:
- room
- bicycle
- garden

One clean seed/config per method is enough for this **baseline reset**. This is not the final publication statistics phase.

Use one physical A100-SXM4-80GB GPU per compared run.
No concurrent user workload on that GPU.

## 9. Native recipes

### FastGS-CVPR26

Use official `train_base.sh` scene arguments at the frozen commit.

For the primary triad:

room:
- `--densification_interval 500`
- `--optimizer_type default`
- `--test_iterations 30000`
- `--highfeature_lr 0.02`
- `--grad_abs_thresh 0.0008`

bicycle:
- `--densification_interval 500`
- `--optimizer_type default`
- `--test_iterations 30000`
- `--grad_abs_thresh 0.0012`

garden:
- `--densification_interval 500`
- `--optimizer_type default`
- `--test_iterations 30000`
- `--highfeature_lr 0.02`
- `--loss_thresh 0.06`
- `--grad_abs_thresh 0.0008`

Do not replace these by one unified config.

### SADGS-SIGGRAPH26

Use official `run_train.sh` semantics at the frozen commit.

For room:
- Mip-NeRF360 indoor configuration
- 3000 iterations
- batch_size 2
- hybrid optimizer
- warmup/structure-aware densification flags exactly as upstream

For bicycle:
- Mip-NeRF360 outdoor configuration
- 3000 iterations
- hybrid optimizer
- structure-aware/warmup densification exactly as upstream

For garden:
- official special garden configuration
- 3000 iterations
- hybrid optimizer
- exact upstream garden flags

### Faster-GS-CVPR26

Use official NeRFICG/Faster-GS native Mip-NeRF360 configuration.

The repository currently ships a `fastergs_garden.yaml`; use it as the canonical schema and create room/bicycle configs ONLY by changing dataset/model-name fields unless upstream documented scene-specific differences are discovered.

Any other field change requires:
- evidence from the official paper/code/config,
- documentation in `P16_CONFIG_PROVENANCE.json`.

For the fused branch, use the same method-level configuration as main; only the upstream fused branch is changed.

## 10. Common final quality gate

For direct “same-quality” comparison against P12 C0, a method/scene is quality-admissible only if:

\[
PSNR \ge PSNR_{C0} - 0.10\,\mathrm{dB}
\]

AND

\[
SSIM \ge SSIM_{C0} - 0.003
\]

AND

\[
LPIPS \le LPIPS_{C0} + 0.005.
\]

All three are preregistered.

Do not rescue a failed metric using an aggregate mean.

Also report every method’s native final quality even if it fails this strict C0-matched gate.

## 11. Time-to-quality curve

For every external method passing Stage 1, produce checkpoints/renders at approximately:

- 20%
- 40%
- 60%
- 80%
- 100%

of the method’s native training schedule.

Checkpoint/evaluation I/O must be outside the primary training timer.

If the native code cannot save these without altering training:
- perform separate replay runs for curve collection;
- label them `CURVE_REPLAY`;
- never mix their wall time with the primary clean run.

For each checkpoint record elapsed training time and common PSNR/SSIM/LPIPS.

Compute time to first reach the C0-admissible quality box.

If never reached: `TTQ = null`.

## 12. Bottleneck census

Do not instrument the primary timing run.

After native runs are complete, take a cloned checkpoint near:
- 60% of native schedule,
- final state.

Run a separate 200-step profiling replay where valid.

Measure, with CUDA events or upstream timing hooks:

- data/camera preparation
- projection / preprocessing / culling
- intersection / sort
- raster forward
- loss
- raster backward
- optimizer
- densification / pruning / structure analysis
- other Python/GPU overhead

If backward and optimizer are fused, report:
`backward_optimizer_fused_ms`
rather than inventing a split.

If SADGS/FastGS perform major control work only at sparse iterations, additionally report amortized density-control cost over a representative window.

Profiling runs are not used for headline wall time.

## 13. Derived metrics

For each scene/method:

- native training wall time
- common final quality
- # iterations
- # Gaussians
- peak VRAM
- ms/iteration median
- time-to-quality (TTQ)
- quality-admissible boolean
- direct-comparable-resolution boolean

For qualifying methods:

\[
S_{C0} = T_{C0}/T_{method}
\]

and:

\[
S_{TTQ} = TTQ_{C0}/TTQ_{method}.
\]

Do not compute a speedup against a method that failed the quality or resolution comparability gate.

## 14. Baseline-reset verdict

The aggregator emits one of:

### `RESET_STRONG_BASELINE_FOUND`
At least one modern method is quality-admissible on all 3 scenes and is faster than P12 C0 on all 3.

### `RESET_SCENE_DEPENDENT`
No single modern method is admissible+faster on all three, but one or more are on the Pareto front by scene.

### `RESET_P12_REMAINS_STRONG`
No modern pinned method is both admissible and faster than P12 C0 on the triad.

These are benchmark findings, not method endorsements.

## 15. Bottleneck classification

For each modern Pareto method classify:

### `EXECUTION_DOMINATED`
if >=70% of profiled iteration time remains in renderer + backward/optimizer execution and native iteration count is not materially lower than P12 (>=70% of P12 iterations).

### `CONVERGENCE_DOMINATED`
if native method uses <=50% of P12 iteration count OR reaches the common quality box in <=50% of P12 wall time primarily through density-control/optimization trajectory changes.

### `MIXED`
otherwise.

This classification is descriptive.

## 16. Stop rule

P16-RB ends after benchmark + census.

Do NOT:
- invent P16 optimization,
- port P13–P15 tricks,
- tune external methods after seeing results,
- run the full 13-scene publication campaign.

The next research mechanism is designed only after this reset report exists.

## 17. Required outputs

- `P16_SOURCE_IDENTITY.json`
- `P16_CONFIG_PROVENANCE.json`
- `P16_ENVIRONMENTS.json`
- `P16_RESOLUTION_SPLIT_AUDIT.json`
- `runs/<method>/<scene>/P16_RUN.json`
- `curves/<method>/<scene>.json`
- `profiles/<method>/<scene>_{mid,final}.json`
- `P16_COMMON_METRICS.json`
- `P16_BASELINE_TABLE.json`
- `P16_BOTTLENECK_CENSUS.json`
- `P16_VERDICT.json`
- `P16_REPORT.md`
- `P16_CONTAMINATION.json`
- `MANIFEST_P16.json`

No git commit.
