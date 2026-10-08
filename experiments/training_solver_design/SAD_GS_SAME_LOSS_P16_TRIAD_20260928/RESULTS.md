# SAD-GS original-loss triad on P16 data

Date: 2026-09-28. This is a baseline calibration, not a new mechanism experiment. The frozen SAD-GS source is commit `029814fb5ed773e4a794cec5115951a64f610ab8`; each scene used its recorded author configuration for 3,000 iterations, with `lambda_l2=0.0`, so the training objective is `0.8*L1 + 0.2*(1-SSIM)`. The runs used the P16 common datasets and test splits at 1920×1080. All three records report `complete_validated`; common evaluation covered all expected test views (room 39/39, bicycle 25/25, garden 24/24).

| Scene | PSNR | SSIM | LPIPS | Train phase wall (s) | Author CUDA-event training (s) |
|---|---:|---:|---:|---:|---:|
| room | 31.1373 | 0.92374 | 0.27595 | 137.204 | 87.063 |
| bicycle | 24.6237 | 0.75152 | 0.24972 | 146.996 | 85.050 |
| garden | 26.5097 | 0.84157 | 0.14526 | 168.794 | 101.700 |

The train-phase wall includes the training process; it is distinct from the author CUDA-event timer. It is also distinct from complete train+render+metric wall. Room records 268.988 s for those phases; the bicycle and garden recovery records retain per-phase timings but do not populate that aggregate field, so no complete-wall ranking is reported here. The runs were isolated on GPU 3 while the P16 GPU 2 queue proceeded; comparisons across those concurrent runs are descriptive.

The P16 native SAD-GS reference used the author's default extra L2 term (`lambda_l2=2.0`), so it is not the same-loss control. Its native training event totals were 85.48/92.18/105.81 s for room/bicycle/garden and common-metric PSNR was 31.1446/24.7394/26.6465. The original-loss PSNR differences are −0.0073/−0.1157/−0.1368 dB. These are single-seed baseline results and do not establish an optimization-method quality or full-wall advantage.

## P16 curve audit

The returned P16 v2 curve bundle has 12 JSON files (four methods × three scenes), each with five strictly increasing time points and successful common metrics; the rebuild log reports `VERIFY_CURVES_BAD 0` and the execution chain ends with `FINAL_RERUN3_DONE`. The graphdeco SAD-GS checkpoints are 600/1200/1800/2400/3000 iterations. Their calibrated endpoint x values are 87.457/91.389/106.084 s for room/bicycle/garden, within +2.31%/−0.86%/+0.26% of the primary native event totals. This x axis estimates native `Training time:` semantics by subtracting measured checkpoint-save time and scaling loop wall by the per-run event/wall factor; it is not complete command wall.

All 12 v2 curves remain outside the preregistered C0 quality box at every checkpoint, so TTQ is null throughout. For SAD-GS the replay endpoints are 31.2726/24.7260/26.6003 PSNR, respectively, still below the room/bicycle/garden box edges by 0.345/2.574/1.750 dB. Do not interpret these curve x values as full-process speed claims.

## Evidence pointers

- `evidence/execution.meta.json` — triad status, metrics and train-phase wall.
- `evidence/{room,bicycle,garden}/run.meta.json` and `metrics_common.json` — per-scene run/evaluation provenance.
- `p16_rb_return/return_p16rb_20260928/curves/SADGS/{room,bicycle,garden}.json` — v2 curve points and elapsed-time evidence.
- `p16_rb_return/return_p16rb_20260928/artifacts/P16_CURVE_XAXIS_CORRECTION.json` and `artifacts/P16_REPORT.md` — curve calibration, TTQ interpretation and full P16 conclusions.

The live EPIC-05 host did not accept a direct SSH check from this workspace (connection timed out). Root's concurrent operator status reported the curve phase complete at 12/12 with no failures; this report also checks the returned files and logs.
