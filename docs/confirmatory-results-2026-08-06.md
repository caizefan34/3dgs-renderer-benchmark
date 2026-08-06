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
| 720p resolution leg | EPIC-05 | A100 (same pool) | 960x540 (720p) | complete, 30/30 runs OK; within-resolution analysis only |
| Consumer GPU (RTX 5070 Laptop 8GB) | local | 1x GPU | 960x540 (720p) | open gate: 8 GB VRAM cannot fit the frozen HiGS workload (protocol section 9) |
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

## 3. 720p resolution leg (EPIC-05 A100, 30k steps) - complete

Runs: 5 scenes x 3 seeds x 2 arms = 30; failures: 0; failure rate: 0%.
Raw per-run values: `paper/tables/confirmatory-consumer-720p-per-seed.json`;
rendered tables: `paper/tables/confirmatory-consumer-720p-table.md`.

| metric | mean paired delta (pd - ctrl) | 95% block-bootstrap CI |
| --- | ---: | --- |
| train_ms (ms/step) | +0.902 | [+0.481, +1.433] |
| total wall (s) | +27.20 | [+14.58, +43.12] |
| PSNR (dB) | +0.531 | [+0.134, +0.968] |
| SSIM | -0.004 | [-0.013, +0.003] |
| LPIPS | -0.013 | [-0.029, +0.000] |

- Strict per-scene dominance: 0 of 5 scenes; the pd cell is not faster at 720p
  at the 30k horizon either.
- Quality guardrail passes on the mean (PSNR +0.53 dB, LPIPS -0.013), but
  quality is not universally positive: truck delta PSNR -0.002 / SSIM -0.021
  and bonsai delta LPIPS +0.006 are per-scene regressions on those metrics.
- Time to target quality: 15/15 pd runs reach the paired ctrl final PSNR at
  the first eval point (step 300).
- Hardware note (protocol addendum, section 9): the pre-registered consumer
  leg was to run on the local RTX 5070 Laptop 8GB at 720p. The local GPU
  cannot fit the frozen HiGS workload: even 200-step smoke runs on the
  5.83M-Gaussian garden scene pinned 7.6 GB of 8.15 GB and dropped to ~10
  s/step under WDDM paging (A100 runs at ~13 ms/step), and 720p peak VRAM in
  this matrix reaches 12.6 GB (bicycle ctrl), so no 30k cell is feasible
  locally. The 720p leg was therefore executed on EPIC-05 A100 with the
  identical frozen configs and seeds and is analyzed within resolution only;
  absolute timings are never compared across the two hardware pools.

Interpretation: the exploratory 720p/3k speed win (C-005/C-006) does not
survive the 30k horizon at 720p either; at 720p/30k the pd cell is slower on
average (+0.90 ms/step) with a robust average quality gain (PSNR +0.53 dB),
the same qualitative pattern as the 1080p/30k leg (C-011). The genuine
consumer-GPU sub-gate (RTX 5070 Laptop) remains open with the reproducible
launcher recipe.

## 4. Open gates and limitations

- Consumer GPU (RTX 5070 Laptop 8GB): the 8 GB VRAM floor of the frozen HiGS
  workload (720p peak VRAM 2.7-12.6 GB per cell) makes the 30k matrix
  infeasible on the local consumer GPU; this sub-gate stays open with the
  reproducible launcher recipe. The completed 720p leg on EPIC-05 A100 is
  evidence for the resolution regime, not for consumer hardware.
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
- `results/confirmatory-consumer-720p/` - manifest, 30 per-run JSON + logs
- `results/confirmatory-{matrix,db,consumer-720p}/summary.json`
- `paper/tables/confirmatory-{matrix,db,consumer-720p}-per-seed.json`
- `paper/tables/confirmatory-{matrix,db,consumer-720p}-table.md`
- `paper/tables/confirmatory-{matrix,db,consumer-720p}-<metric>-bootstrap.json` (5 metrics each)
- `src/scripts/collect_confirmatory_results.py`, `src/scripts/bootstrap_analysis.py`,
  `src/scripts/build_confirmatory_tables.py`

Consumer-family analysis recipe (same commands reproduce the A100 matrix and
DB legs by substituting the family name and arm regexes):

```bash
python src/scripts/collect_confirmatory_results.py \
  --in-dir results/confirmatory-consumer-720p --out results/confirmatory-consumer-720p/summary.json
for m in train_ms total_wall_s psnr ssim lpips; do
  python src/scripts/bootstrap_analysis.py \
    --baseline results/confirmatory-consumer-720p/summary.json \
    --method results/confirmatory-consumer-720p/summary.json \
    --baseline-arm ctrl --method-arm pd \
    --metric $m \
    --key-regex '(?P<scene>garden|bicycle|bonsai|train|truck)_(?:ctrl|pd)_s(?P<seed>\d+)$' \
    --strict --out paper/tables/confirmatory-consumer-720p-$m-bootstrap.json
done
python src/scripts/build_confirmatory_tables.py --family confirmatory-consumer-720p
```
