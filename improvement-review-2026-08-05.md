# Improvement Review — 2026-08-05

Scope: beginner-friendliness and correctness review of the README + full-repo
quality audit. Every claim below was verified against committed data or
reproducible commands (not memory).

## 1. What was fixed

### README (beginner-friendly + rendering errors)

- Replaced the old text-only intro with a visual banner, a mermaid
  architecture diagram, a plain-language "What is this?" section, a glossary,
  and collapsible research-history blocks so new readers get the big picture
  before any numbers.
- Closed the first mermaid code fence (was unclosed, breaking GitHub
  rendering of that diagram) and normalized line endings to LF.
- Repaired mojibake em-dashes/arrows in the README and report scripts.
- Fixed the tests badge (`155_passing` -> `155_tests_OK`; the count text in
  TL;DR is now `154 pass + 1 skip`, matching the real suite).
- Compression numbers now match the per-scene evidence exactly:
  - Per-scene SPZ 8/8 ratios are `5.57-6.07x` (bicycle 5.78x, bonsai 6.07x,
    garden 5.57x, train 5.74x, truck 5.83x), so the README now says
    `5.57-6.07x (median 5.78x)` instead of a single `5.73x` that matched no
    scene.  Source: `reports/epic05-spz-qualification-2026-07-24.md`
    (byte counts verify every row) and `results/measured-compression/spz/`.
  - `docs/epic05-higs-ablation-results-2026-07-25.md`: Truck compression was
    `~5.732x` (that figure is the byte-weighted **aggregate** across scenes);
    corrected to `~5.83x`, which is Truck's actual measured ratio.
  - The README roadmap bullet still said `5.73x`; aligned to the per-scene
    range as well.
- Corrected `4/5` strict-dominance statement in the HiGS research report:
  the original text was confusingly worded; it now names each scene and its
  config (bicycle/truck/bonsai 0.5x, garden 0.75x; low-N train stays
  not-recommended).

### Report generator scripts

- `src/scripts/gen_report.py`: `??` placeholders (mojibake of an emoji) in
  the HTML header / fastest-renderer tag replaced with `🚀` / `🏆`.
- `src/scripts/generate_report.py`: `? Active` -> `✅ Active`,
  `?? Wrapper` -> `🔌 Wrapper`, bare `?` -> `🚫`.

### Test suite — CI was red on master (fixed)

- `tests/test_benchmark_progressive_res.py`,
  `tests/test_benchmark_tile_sampling.py`,
  `tests/test_higs_sparse_pixel_raster.py` could not be collected:
  they load `benchmark/run_higs_train_benchmark.py` via importlib, and that
  module does `from higs_masked_adam import masked_adam_step`, which requires
  `benchmark/` on `sys.path`.  Neither `unittest discover` nor pytest adds it.
- Fix: each of the three test files inserts its repo `benchmark/` directory
  into `sys.path` before executing the module (CPU-safe; the CUDA extension is
  lazy-loaded only when CUDA is actually present).
- Verified: `python -m unittest discover -s tests` -> `155 tests OK (1 skip)`;
  `python -m pytest tests -q` -> `203 passed`.  These are the exact same
  green numbers as the historical baseline, so the fix restores CI parity.

## 2. Leaderboard — verified, no change needed

`docs/leaderboard/ranking.md` (Tier A) is fully reproducible from
`results/measured/**/metrics.json`:

- Each renderer row uses the **first-run batch (2026-07-20)** for all 5 cases
  (bicycle/bonsai/garden/train/truck) and geometric-mean aggregation, exactly
  as `_aggregate_renderer` in `src/benchmark_matrix.py` defines:
  - gsplat 241.60 FPS / 4.139 ms, gsplat_higs 696.91 / 1.435,
    original_3dgs 122.88 / 8.138, speedy_splat 293.03 / 3.413,
    tcgs 251.62 / 3.974 — all match to the printed 2-3 decimals.
- PSNR / SSIM / LPIPS (arithmetic mean) and VRAM (per-case max) also match.
- The later 2026-07-23 re-runs are additional evidence batches; the published
  table intentionally reports the first complete Linux Tier A matrix.  This is
  worth a one-line note in `docs/leaderboard/README.md` (see recommendations).

## 3. What is still open (honest list)

- **Ranking.md aggregation note**: the committed table does not state which
  batch it aggregates.  Recommend adding "first complete run of 2026-07-20,
  geometric mean over 5 cases" to the leaderboard doc so future readers do not
  re-derive it.
- **fast-gaussian-rasterization**: skipped on this Windows/CPU environment
  (requires EGL/GL); it is GPU/CI-only.  The CI manual benchmark workflow
  (`.github/workflows/benchmark-regression.yml`) is self-hosted GPU and not
  run in the standard test job.
- **tcgs variance**: TC-GS FPS varies wildly between runs (64-365 FPS across
  batches; CI width 153-396 in the published table).  The geometric mean of
  the first batch is what is published; if TC-GS numbers matter, a
  dedicated warm-up/steady-state protocol would tighten the CI.
- **`reports/final-conclusions.md`**: still uses `5.73x` for SPZ 8/8.  That
  value is the byte-weighted aggregate and is not wrong, but it differs from
  the per-scene range now used in the README; left untouched as a dated
  report artifact.
- Not changed (out of scope): GPU kernel code, benchmark protocol JSON, and
  the CSV/JSON evidence artifacts under `results/` — those are raw data.

## 4. Recommendations for the next pass

1. Add the batch + aggregation note to `docs/leaderboard/README.md`
   (one paragraph, prevents future re-derivation).
2. Publish `docs/improvement-review-2026-08-05.md` (this file) link from the
   README repository-layout section so the audit is findable.
3. If TC-GS is a headline number, add a per-renderer warm-up knob to the
   protocol (e.g., `--warmup-frames 100`) and re-measure; otherwise keep the
   published CI as-is and document the variance.
4. Consider a `conftest.py` at the repo root adding `benchmark/` to
   `sys.path`, which would make future benchmark-module tests import-clean
   without repeating the three-file patch (kept minimal for now).
5. Add a `make check` / `scripts/check.sh` alias that runs both
   `unittest discover` and `pytest` so contributors run the same gate CI does.
