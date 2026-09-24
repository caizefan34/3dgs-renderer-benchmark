# Improvement Review — 2026-08-05

This review covers README readability, rendering correctness, test health,
packaging, leaderboard consistency, and the TC-GS variance investigation.
Claims below are tied to committed data or reproducible commands.

## Completed fixes

### README and documentation

- Added a visual project overview, plain-language introduction, glossary, and
  collapsible research history so new readers see the purpose before details.
- Closed the broken Mermaid fence, normalized line endings, and repaired
  mojibake in the README and report generators.
- Corrected the test badge and TL;DR test count.
- Replaced misleading single compression ratios with the measured SPZ 8/8
  per-scene range: `5.57-6.07x` (median `5.78x`). The `5.73x` value remains
  valid only as the byte-weighted five-scene aggregate and is now labeled.
- Clarified the HiGS strict-dominance result: bicycle, truck, and bonsai use
  the 0.5x schedule; garden uses 0.75x; low-N train is not recommended.

### Test and CI health

- Fixed three benchmark-module tests that could not import
  `benchmark/higs_masked_adam.py` under unittest or pytest collection.
- Verified the historical baseline after that fix:
  `155 tests OK (1 skip)` under unittest and `203 passed` under pytest.

### Packaging and installed CLI

- Reproduced a release defect: the 0.2.0 wheel contained only
  `benchmark_cli.py` and `benchmark_matrix.py`.
- Added complete `src/` package discovery and all top-level runtime modules.
- Added an installed entry point that locates the source checkout through
  `GSBENCH_ROOT` or a parent of the current directory. It now fails with an
  actionable message instead of searching the Python installation directory.
- Added wheel-content and outside-checkout CLI regression tests. A clean
  isolated install successfully ran `benchmark list renderers`.

## Leaderboard verification

The Tier A table is reproducible from `results/measured/**/metrics.json`.
It intentionally uses the first complete 2026-07-20 batch for all five cases
and geometric-mean speed aggregation:

| renderer | aggregate FPS | frame time ms |
| --- | ---: | ---: |
| gsplat | 241.60 | 4.139 |
| gsplat HiGS | 696.91 | 1.435 |
| Original 3DGS | 122.88 | 8.138 |
| Speedy-Splat | 293.03 | 3.413 |
| TC-GS | 251.62 | 3.974 |

Later runs remain additional evidence. They do not replace the first batch
after outcome inspection.

## TC-GS variance re-test

Three complete EPIC-05 passes used the same A100 GPU UUID, TC-GS commit,
assets, cameras, and 500 measured frames per scene:

| pass | warm-up | five-scene geometric-mean FPS | scenes with >100 ms max |
| --- | ---: | ---: | ---: |
| A | 30 | 217.70 | 5/5 |
| B | 150 | 312.56 | 1/5 |
| C | 30 | 275.46 | 1/5 |

Median latency stays at 1.48-3.99 ms, while isolated frames reach 229-423 ms.
Quality is exactly repeatable for every scene. Longer warm-up coincides with
fewer stalls but does not eliminate them, and Pass C improves with the
original warm-up. Warm-up and changing host load are therefore confounded;
neither a kernel defect nor host contention is proven.

The 105 JSON artifacts are isolated under
`results/diagnostics/tcgs-variance-20260805/` and cannot enter the leaderboard.
Full results: `reports/epic05-tcgs-variance-2026-08-05.md`.

## Remaining gaps

- `fast-gaussian-rasterization` still needs a validated Linux EGL environment
  and framebuffer readback path.
- GPU performance CI remains manual rather than scheduled and pinned.
- A paper submission still needs a frozen claim set, full-convergence runs,
  held-out confirmation data, stronger compatible baselines, multi-GPU-
  architecture evidence, and paired confidence intervals.
- Independent replication on a second host or institution is not available.

The submission-facing gap analysis and minimum confirmatory matrix are in
`docs/research-readiness-audit-2026-08-05.md`. The paper workspace is under
`paper/` and must not be called submission-ready until every P0 item is closed.

## Verification gates

```text
python -m unittest discover -s tests
python -m pytest tests -q
python src/scripts/validate_benchmark_suite.py
```

The packaging test also builds a wheel without build isolation, inspects its
contents, and executes its entry point from outside the checkout.
