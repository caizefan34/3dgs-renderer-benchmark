# Research Readiness Audit — 2026-08-05

## Scope and standard

This audit evaluates whether the repository can support a defensible top-tier
graphics or vision submission. It does not claim that engineering quality can
guarantee acceptance. The target is narrower and testable: every headline
claim must be novel, pre-specified, statistically supported, reproducible from
published artifacts, and bounded by explicit limitations.

## Executive assessment

The repository is already stronger than a typical project release in raw
artifact retention, renderer identity pinning, quality gating, and negative
result reporting. It is not yet submission-ready. The largest gap is not code
volume; it is the confirmatory experimental design around the novel HiGS
training contribution.

| priority | gap | why reviewers will care | status / required action |
| --- | --- | --- | --- |
| P0 | No formal manuscript or frozen claim set | The contribution, hypotheses, and evidence are spread across long engineering reports | Open: create a paper source and a claim-to-artifact manifest |
| P0 | Adaptive search across 65 rounds | Selecting the best configuration on the same scenes/seeds creates winner's-curse and multiple-comparison risk | Open: freeze the method, then run untouched confirmatory scenes/seeds |
| P0 | Most HiGS training evidence uses 3,000 steps | Short-horizon gains may not survive the standard 30k convergence regime | Open: run full convergence for all five scenes and strong baselines |
| P0 | Single hardware family | A100-only conclusions do not establish consumer, workstation, or newer datacenter behavior | Open: add at least one consumer and one second datacenter architecture |
| P0 | Missing modern end-to-end competitors | FlashGS, Local-GS/TiCoGS, GEMM-GS, and related training accelerators are not in the same-checkpoint matrix | Open or justify incompatibility in a separate-track table |
| P1 | TC-GS heavy-tail latency | Mean FPS is unstable across independent launches | Partially closed: three-pass evidence and honest limits are published |
| P1 | Statistical analysis is mostly mean +/- SD | Reviewers need paired effects, confidence intervals, and a declared experimental unit | Open: add a pre-registered analysis script and block bootstrap |
| P1 | No untouched held-out confirmation set | The same five scenes informed method development and final claims | Open: reserve scenes/datasets before final tuning |
| P1 | GPU CI is manual | CPU correctness is automated; performance reproducibility is not continuously exercised | Open: scheduled pinned-host smoke and artifact publication |
| P1 | Published wheel was incomplete | The installed CLI omitted runtime packages and could not locate the checkout | Fixed in this pass with wheel and installed-CLI tests |
| P2 | Artifact validation in CI is selective | A stale or malformed committed result can escape the current smoke checks | Open: validate all canonical result manifests and referenced hashes |
| P2 | Dataset/licensing statement is fragmented | Paper artifact evaluation needs one clear redistribution and download policy | Open: consolidate licenses, checksums, and non-redistributed assets |
| P2 | No independent replication | Authority-host results are internally reproduced only | Open: publish a replication recipe and solicit a second host/institution |

## Paper-worthy contribution boundary

The benchmark platform is useful infrastructure, but infrastructure alone is
unlikely to be the strongest top-conference story. The most credible research
claim is the trainable HiGS system combining sparse training, masked optimizer
updates, projection-only visibility refresh, and a progressive-resolution
schedule. The paper must distinguish:

- established upstream components;
- this repository's implementation contribution;
- the final frozen method after exploration;
- exploratory negative results that motivated the method; and
- confirmatory results collected only after the method was frozen.

Compression and renderer benchmarking should support the system evaluation,
not compete with the central contribution unless the paper is explicitly
positioned as a benchmark paper.

## Minimum confirmatory experiment package

1. Freeze one speed-oriented and one quality-oriented configuration before
   running any new test scene or seed.
2. Train to the standard convergence horizon, not only 3,000 steps.
3. Use paired seeds and randomized in-wave baseline/method execution blocks.
4. Report wall time, time to target quality, PSNR, SSIM, LPIPS, final Gaussian
   count, peak VRAM, and failure rate.
5. Publish per-seed values, paired deltas, 95% block-bootstrap intervals, and
   effect sizes; avoid declaring significance from overlap heuristics.
6. Evaluate all five current scenes plus a held-out dataset family.
7. Repeat the frozen protocol on at least three GPU classes.
8. Include upstream gsplat, Original 3DGS, and the strongest compatible modern
   training baseline under the same checkpoints and evaluation code.
9. Profile the claimed mechanism and show where time is removed, not only the
   final wall-clock delta.
10. Run the full artifact pipeline from a clean checkout and publish a single
    manifest mapping every paper table cell to immutable JSON.

## Claim policy for the current repository

- "Measured on EPIC-05" is acceptable when linked to raw artifacts.
- "Faster and better" is only acceptable for the exact scene, horizon,
  hardware, seed set, and baseline shown.
- "General" or "state of the art" is not supported by the current evidence.
- Host contention, kernel faults, and mechanism explanations are hypotheses
  unless isolated by a controlled experiment or profiler evidence.
- Negative results remain valuable and should stay visible; they reduce
  researcher degrees of freedom when accompanied by the pre-specified gate.

## Immediate engineering gates

- `python -m unittest discover -s tests`
- `python -m pytest tests -q`
- build a wheel without network/build isolation and inspect its contents;
- install that wheel into a clean environment and run the CLI from outside the
  checkout with `GSBENCH_ROOT` set;
- validate the canonical suite/protocol hash and all newly committed TC-GS
  artifacts;
- render-check README Mermaid, HTML, SVG, and relative links on GitHub Pages.

The repository should not label itself "submission-ready" until every P0 item
has direct evidence and the confirmatory experiment package is complete.

## Status updates — 2026-08-06

Progress made after the audit date; the table above remains the original
08-05 snapshot.

- **P0 manuscript / claim set: in progress.** `paper/main.tex` is a draft
  manuscript skeleton and `paper/claims.yaml` is a schema-v1 claim-to-artifact
  manifest (10 claims, measured/exploratory statuses, evidence paths, and the
  required freeze gates before `freeze_status` may become `frozen`).
  `tests/test_paper_claims_manifest.py` validates the schema, the protocol
  hash against `benchmark/suite.json`, and that every evidence path exists.
- **P0 adaptive search: in progress.** The round-65 sweep is committed and the
  `--higs-quality-max` preset gives one reproducible frozen-configuration
  entry point; the untouched confirmatory scenes/seeds remain open.
- **P1 statistical analysis: closed for the script, open for the frozen
  protocol.** `src/scripts/bootstrap_analysis.py` implements paired
  per-scene/per-seed deltas with a scene-level block bootstrap, 95%
  percentile intervals, and a paired effect size (11 unit tests,
  `tests/test_bootstrap_analysis.py`); `paper/tables/` holds a generated
  example table. Pre-registered stopping rules for the confirmatory protocol
  still need to be declared.
- **P2 artifact validation: improved.** The claims-manifest test now runs in
  CI, so committed claim evidence paths and the canonical protocol hash are
  continuously checked.
- **Engineering gates re-verified 2026-08-06:** `unittest discover` = 180
  tests OK (1 skip); `pytest tests -q` = 228 passed; README Mermaid fences
  verified balanced and all evidence paths referenced by claims.yaml exist.


## Confirmatory status 2026-08-06 (evening)

The pre-registered confirmatory protocol (`paper/confirmatory-protocol.md`)
has been executed on the primary host and the held-out family. The table
above remains the original 08-05 snapshot; this section records the frozen-
protocol evidence.

- **P0 30k convergence (audit "Open"): closed.** The frozen ctrl/pd pair ran
  to 30,000 steps on all five canonical scenes, 3 seeds x 2 arms = 30 runs,
  zero failures (`results/confirmatory-matrix/`).
- **P0 held-out confirmation (audit "Open"): closed.** Deep Blending
  (playroom, drjohnson) was prepared from the official Graphdeco iteration-
  30000 checkpoints and never touched exploratory rounds; 12/12 runs
  completed (`results/confirmatory-db/`).
- **P1 paired statistics (audit "Open"): closed for the primary matrix.**
  Per-seed values and scene-level block-bootstrap 95% CIs are published for
  train_ms, total_wall_s, PSNR, SSIM, LPIPS (`paper/tables/`); the primary
  outcome (pd train_ms delta) is +0.739 ms/step (CI [+0.197, +1.511]) with
  0/5 strict-dominance scenes, so the 1080p/30k speed hypothesis is a
  pre-registered negative, while final quality improves (PSNR +0.612 dB,
  LPIPS -0.014). On Deep Blending the pd cell is faster and better on average
  (train_ms -0.261, PSNR +0.748), with strict dominance on drjohnson.
- **P0 hardware breadth (audit "Open"): partial.** The 720p resolution leg
  is complete (30/30 on EPIC-05 A100, zero failures, within-resolution
  analysis; protocol section 9 addendum documents why the local RTX 5070
  Laptop 8GB cannot fit the frozen workload). The genuine consumer-GPU and
  second-datacenter-GPU sub-gates remain open with the launcher recipe
  published (protocol section 7).
- **P1 manuscript / claim set: in progress.** `paper/claims.yaml` now carries
  the three measured confirmatory claims C-011/C-012/C-013 and per-gate
  statuses (G-1..G-5); `freeze_status` remains "draft" until the
  consumer-GPU and second-datacenter sub-gates close.
- **P2 artifact validation: improved.** `build_confirmatory_tables.py`
  generates per-seed JSON + markdown tables so every paper table cell traces
  to an immutable artifact; the claims test still runs in CI.
- **Launcher fix:** `run_confirmatory_matrix.py` now raises a clear error when
  a pd arm is requested without a coarse res-schedule (was a silent None-crash
  on custom scene specs); covered by a new unit test.

Primary result in one line: at 1080p/30k the frozen pd cell is a quality win
(PSNR +0.61 dB, LPIPS -0.014, ~4x fewer Gaussians, lower VRAM) and not a
training-speed win on the canonical five; the held-out family shows the speed
win too (train_ms -0.26). The completed 720p/30k leg (C-013) shows the
exploratory 720p speed result does not survive the 30k horizon at 720p
either; the genuine consumer-GPU test remains open.
