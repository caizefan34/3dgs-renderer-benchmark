# P16-RB return package — 2026 baseline reset + bottleneck census

**Box:** EPIC-05 (8.130.30.251), deployed tree `/root/epic05_p16rb`
**Date:** 2026-09-28
**Verdict:** `RESET_SCENE_DEPENDENT` (`artifacts/P16_VERDICT.json`); `strong_methods = []`
**Bottom line:** exactly one of twelve modern cells is quality-admissible and faster on the triad — FasterGSFused/room at 1.7319x (PSNR 31.7282, inside the prereg §10 C0 box). No admissible method exceeds 1.73x on any scene, so **P12 C0 remains the conservative strong baseline** for future ">3x" claims.

## Where to look

- `artifacts/P16_REPORT.md` — full report (primary document).
- `artifacts/P16_BASELINE_TABLE.json` — 15 canonical rows (walls + common metrics + box status); aggregator input for the verdict.
- `artifacts/P16_VERDICT.json` — verdict, per-scene admissible Pareto vs C0, census classes, TTQ.
- `artifacts/P16_BOTTLENECK_CENSUS.json` — 12 profile cells (mid/final component medians, classification per prereg §15).
- `artifacts/P16_COMMON_METRICS.json` — common metric pipeline outputs (PSNR/SSIM/LPIPS) per cell.
- `artifacts/P16_CURVE_XAXIS_CORRECTION.json` — v1→v2 curve x-axis forensics + graphdeco alpha calibration.
- `artifacts/P16_CONTAMINATION.json` — per-run GPU hygiene captures (UUID, foreign PIDs, peak VRAM).
- `artifacts/P16_DATA_HEAL_garden.json`, `artifacts/P16_DATA_FIX_garden_points3D.json` — garden dataset corruption, heal, and duplicate-ID fix records.
- `artifacts/P16_P12_REPLAY_VERIFICATION.json` — P12 C0 archived-timing revalidation.
- `artifacts/P16_RESOLUTION_SPLIT_AUDIT.json`, `artifacts/P16_ENVIRONMENTS.json`, `artifacts/P16_SOURCE_IDENTITY.json`, `artifacts/P16_CONFIG_PROVENANCE.json` — fairness/provenance gates.
- `artifacts/MANIFEST_P16.json` — sha256 of every file in this package (62 entries).
- `runs/<method>/<scene>/P16_RUN.json` — 15 canonical per-cell records (native walls, contamination captures, metrics pointers); garden_flatdata_corrupt and replay500 records retained alongside.
- `curves/<method>/<scene>.json` — v2 TTQ curves. Note: TTQ is null in all 12 pairs — no replay ever enters the C0 box (see report §7).
- `curves_v1_archive/` — superseded v1 curves, retained by rename (rename-only archive policy).
- `profiles/<method>/<scene>_{mid,final}.json` — instrumented profile windows; instrumentation diff in `diffs/`.
- `envs/` — environment exports (conda/pip) per method env; `diffs/` — source/build/instrumentation diffs.
- `scripts/` — all drivers/builders used as executed; `logs/` — full logs (stage queues, curve rebuild, finalize chain).

## Headline table (primary walls, native timing semantics per report §2)

| Method | room | bicycle | garden | quality box |
|---|---:|---:|---:|---|
| P12_C0 | 471.64 s / 31.7179 | 828.79 s / 27.3999 | 778.76 s / 28.4505 | reference |
| FasterGS | 266.96 s / 31.6069 | 439.38 s / 24.7239 | 485.72 s / 26.8588 | miss |
| FasterGSFused | **272.32 s / 31.7282 (PASS, 1.73x)** | 464.82 s / 24.7168 | 528.24 s / 26.8485 | 1 of 3 pass |
| FastGS | 194.00 s / 31.8014 | 210.57 s / 24.4129 | 267.82 s / 26.6986 | miss (SSIM/LPIPS or PSNR) |
| SADGS | 85.48 s / 31.1446 | 92.18 s / 24.7394 | 105.81 s / 26.6465 | miss (PSNR) |

Cells show wall (s) / common-metric PSNR; full metrics in the report table. Raw-wall speedups outside the box are not valid headline numbers (prereg §13).

## Policy notes

- No git commit; the campaign stops after this report (prereg §16/§17).
- No optimization mechanism was introduced or tuned after seeing results.
- Archive operations were renames only; superseded artifacts (v1 curves, flat-data records) are retained verbatim in this package or on the box.
