# Confirmatory Results 2026-08-06

Pre-registered confirmatory evaluation of the frozen HiGS speed/quality
configurations. Design, frozen flags, seeds, stopping rules, and success
criteria were locked before any confirmatory run started; see
`paper/confirmatory-protocol.md`.

## Hardware legs

| leg | host | GPU | resolution | status |
| --- | --- | --- | --- | --- |
| Primary matrix | EPIC-05 | 8x NVIDIA A100-SXM4-80GB | 1920x1080 | complete, 30/30 runs OK |
| Held-out family | EPIC-05 | A100 (same pool) | 1920x1080 | complete, 12/12 runs OK |
| Consumer | local RTX 5070 Laptop 8GB | 1x GPU | 960x540 (720p) | running (protocol section 7) |
| Second datacenter GPU | none available | - | - | open gate, documented in the audit |

Software: torch 2.7.0+cu128 / gsplat 1.5.3 fork on EPIC-05; torch
2.12.1+cu130 / same gsplat fork locally. Resolution-specific analysis only:
A100 cells are compared within 1080p, consumer cells within 720p, never mixed.

## 1. Canonical five scenes (A100, 1080p, 30k steps)

- Runs: 5 scenes x 3 seeds x 2 arms = 30; failures: 0; failure rate: 0%.
- Raw per-run values: `paper/tables/confirmatory-matrix-per-seed.json`; rendered
  tables: `paper/tables/confirmatory-matrix-table.md`.
- Primary outcome (pre-registered): paired `train_ms` delta, pd - ctrl.

| metric | mean paired delta (pd - ctrl) | 95% block-bootstrap CI |
| --- | ---: | --- |
| train_ms (ms/step) | +0.739 | [+0.197, +1.511] |
| total wall (s) | +22.33 | [+6.06, +45.46] |
| PSNR (dB) | +0.612 | [+0.286, +1.027] |
| SSIM | +0.003 | [-0.001, +0.006] |
| LPIPS | -0.014 | [-0.026, -0.002] |

Reading: lower is better for `train_ms`, `total_wall_s`, `lpips`; higher is
better for `psnr`, `ssim`.

- Strict per-scene dominance (negative `train_ms` delta on all three seeds and
  quality guardrail): 0 of 5 scenes. The 1080p/30k training-speed hypothesis
  is NOT confirmed on the canonical five.
- Quality guardrail (PSNR mean delta >= -0.05 dB, LPIPS mean delta <= +0.005):
  passes overall. Per-scene PSNR deltas are positive on all five scenes; the
  only per-scene LPIPS regression is truck (+0.006), which fails the per-scene
  LPIPS guardrail and is reported as a regression on that metric, not a win.
- Time to target quality (pre-registered): pd reaches the paired ctrl arm's
  final 30k PSNR at the first eval point (step 300) on all 15 pairs; median
  wall time 2.7-7.4 s per scene. Quality is therefore delivered within the
  first 1% of the horizon.
- Mechanism: pd finishes with far fewer Gaussians (mean final_n ctrl 1.55M vs
  pd 0.41M; bicycle 3.12M -> 398K, garden 2.57M -> 873K) and lower peak VRAM
  on every scene (bicycle 14.4 -> 12.7 GB, garden 14.1 -> 12.8 GB, bonsai
  5.6 -> 4.4 GB, truck 7.7 -> 5.8 GB, train 5.2 -> 4.5 GB).

Interpretation: on the canonical five at full 1080p resolution and the full
30k convergence horizon, the progressive-resolution + union-decay cell does
not speed up training; its robust benefit is final quality (PSNR +0.61 dB,
LPIPS -0.014) with a large Gaussian-count reduction. This matches the
pre-registered expectation that the exploratory 720p speed result (rounds
56/58/60, claims C-005/C-006) does not transfer to the 1080p regime.

## 2. Held-out family: Deep Blending (A100, 1080p, 30k steps)

- Family not used in any exploratory round (playroom, drjohnson).
- Runs: 2 scenes x 3 seeds x 2 arms = 12; failures: 0; failure rate: 0%.
- Raw values: `paper/tables/confirmatory-db-per-seed.json`; rendered tables:
  `paper/tables/confirmatory-db-table.md`.

| metric | mean paired delta (pd - ctrl) | 95% block-bootstrap CI |
| --- | ---: | --- |
| train_ms (ms/step) | -0.261 | [-0.473, -0.049] |
| total wall (s) | -7.67 | [-14.04, -1.31] |
| PSNR (dB) | +0.748 | [+0.026, +1.470] |
| SSIM | +0.011 | [+0.000, +0.022] |
| LPIPS | -0.003 | [-0.008, +0.001] |

- Strict per-scene dominance: 1 of 2 (drjohnson yes; playroom mean `train_ms`
  delta -0.049 but not negative on all three seeds).
- Time to target quality: step 300 on all pairs; median wall time 2.4-2.6 s.

Interpretation: on a held-out family, the pd cell is faster AND better on
average, and generalizes the quality benefit. Per-scene strict dominance is
not universal (playroom is a near-tie on speed), so claims must stay
family-level and per-scene.

## 3. Consumer leg (RTX 5070 Laptop, 720p) - in progress

The full 5-scene x 3-seed x 2-arm matrix at 720p (960x540) is running with the
identical frozen flags (protocol section 7). 1080p cells exceed 8 GB VRAM on
several scenes, so the consumer leg is 720p by pre-registration and is
analyzed within resolution only. Results will be committed under
`results/confirmatory-consumer-720p/` and analyzed in a follow-up section of
this document.

## 4. Open gates and limitations

- Second datacenter GPU: no second datacenter GPU is available in this
  environment; the reproducible recipe is the launcher
  (`src/scripts/run_confirmatory_matrix.py`) and this protocol, documented as
  an open gate in `docs/research-readiness-audit-2026-08-05.md`.
- Single-host effects (clock/power policy) are bounded by the paired
  same-GPU design but not removed for absolute numbers.
- Quality metrics use the pinned 3 held-out cameras; PSNR levels are lower
  than full-100-view training reports by design (n_train=4).
- The 1080p result is a negative for the speed hypothesis on the canonical
  five and positive for quality; no claim of general speed dominance is made
  from it.

## 5. Artifacts

- `results/confirmatory-matrix/` - manifest, 30 per-run JSON + logs
- `results/confirmatory-db/` - manifest, 12 per-run JSON + logs
- `results/confirmatory-matrix/summary.json`, `results/confirmatory-db/summary.json`
- `paper/tables/confirmatory-{matrix,db}-per-seed.json`
- `paper/tables/confirmatory-{matrix,db}-table.md`
- `paper/tables/confirmatory-{matrix,db}-<metric>-bootstrap.json` (5 metrics each)
- `src/scripts/collect_confirmatory_results.py`, `src/scripts/bootstrap_analysis.py`,
  `src/scripts/build_confirmatory_tables.py`