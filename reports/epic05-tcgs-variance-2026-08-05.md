# EPIC-05 TC-GS Variance Re-test — 2026-08-05

## Question

Why does the published TC-GS aggregate have a wide 95% confidence interval,
and does a longer warm-up remove the instability?

## Controlled setup

- Host: EPIC-05, NVIDIA A100-SXM4-80GB GPU 0
  (`GPU-ab2c0112-a9c5-8a5e-90ed-a9145a7150a0`).
- TC-GS source: `DeepLink-org/3DGSTensorCore` commit
  `0bb82f88fde211c34b42e1497f0fc7265461592b`.
- Benchmark source: commit `eb5217cb9d23f0a12a43632d625d614afae1d90f`.
- Runtime: PyTorch 2.9.1+cu128, CUDA 12.8, driver 580.105.08.
- Workload: the five canonical 1920x1080 cases, 100 frames per repeat,
  five repeats, synchronized wall-clock throughput and CUDA-event latency.
- Pass A: 30 warm-up frames; Pass B: 150 warm-up frames; Pass C: a second
  30-frame warm-up run.

The renderer, scene assets, camera order, output quality checks, GPU UUID,
runner, and measured-frame count were held fixed. Other jobs used different
GPUs on the same multi-GPU host during parts of the experiment. Host-level
contention was observed but not controlled, so it is a plausible confounder,
not an established cause.

## Results

| pass | scene | warm-up | mean FPS | 95% CI | median ms | P99 ms | max ms | PSNR dB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | bicycle | 30 | 204.04 | 155.1-273.7 | 3.09 | 14.42 | 341.50 | 24.335 |
| A | bonsai | 30 | 295.59 | 134.4-599.5 | 1.55 | 13.38 | 229.45 | 32.774 |
| A | garden | 30 | 113.38 | 65.6-183.2 | 3.99 | 47.13 | 384.28 | 25.845 |
| A | train | 30 | 283.90 | 186.1-475.4 | 2.08 | 10.86 | 338.13 | 23.543 |
| A | truck | 30 | 251.86 | 135.8-502.3 | 2.20 | 12.81 | 422.92 | 24.153 |
| B | bicycle | 150 | 205.74 | 131.9-340.5 | 3.04 | 15.57 | 338.78 | 24.335 |
| B | bonsai | 150 | 511.35 | 370.8-732.5 | 1.48 | 6.08 | 30.16 | 32.774 |
| B | garden | 150 | 209.27 | 189.6-231.0 | 3.36 | 16.50 | 52.44 | 25.845 |
| B | train | 150 | 437.73 | 339.9-579.7 | 1.85 | 8.47 | 17.52 | 23.543 |
| B | truck | 150 | 309.55 | 219.5-440.6 | 2.16 | 16.66 | 36.55 | 24.153 |
| C | bicycle | 30 | 209.49 | 171.5-256.8 | 3.15 | 16.54 | 78.79 | 24.335 |
| C | bonsai | 30 | 275.74 | 104.2-595.3 | 1.58 | 19.04 | 228.53 | 32.774 |
| C | garden | 30 | 231.64 | 195.2-275.5 | 3.14 | 14.59 | 26.02 | 25.845 |
| C | train | 30 | 382.98 | 288.5-504.7 | 1.95 | 9.95 | 14.46 | 23.543 |
| C | truck | 30 | 309.49 | 218.1-436.4 | 2.21 | 13.98 | 20.33 | 24.153 |

Geometric-mean FPS across the five scenes was 217.70 (A), 312.56 (B), and
275.46 (C). A >100 ms frame appeared in 5/5 A cases, 1/5 B cases, and 1/5 C
cases.

## Interpretation

1. The quality path is deterministic for this experiment. Every scene's PSNR
   is identical across all three passes and the historical batches.
2. TC-GS steady-state latency is much more stable than mean FPS suggests:
   medians stay within 1.48-3.99 ms while isolated frames reach 229-423 ms.
3. A 150-frame warm-up coincides with fewer stalls, but it does not eliminate
   them: bicycle still reaches 338.78 ms. Pass C also improves despite using
   the original 30-frame warm-up, so this experiment cannot identify warm-up
   as the sole cause of the improvement.
4. The published first complete batch remains the pre-declared leaderboard
   estimate. Replacing it after inspecting later runs would introduce
   selection bias. The new runs are variance evidence, not replacement rows.

## Reporting and protocol decision

- Keep mean end-to-end FPS as the primary throughput metric, but always expose
  median, P95, P99, maximum latency, and repeat-level confidence intervals.
- Do not describe the stalls as a proven TC-GS kernel defect or a proven
  host-contention effect without a controlled idle-host experiment and a
  profiler trace around a stalled frame.
- For a paper claim, run process-level randomized blocks on an otherwise-idle
  host, collect at least 10 independent process launches per renderer/case,
  and report a paired effect size with a block bootstrap confidence interval.

## Evidence and ranking isolation

The 105 committed JSON artifacts are under
`results/diagnostics/tcgs-variance-20260805/`. Each run contains the same
artifact classes as the historical Tier A records: `metrics.json`,
`raw_samples.json`, speed samples/NVML samples, and per-view quality evidence.

The directory is deliberately outside `results/measured/`, so these post-hoc
diagnostic runs cannot replace or enter the canonical leaderboard. Pass B
changes warm-up and its convenience `metrics.json` retains the canonical
protocol identity; it must not be treated as a canonical protocol record.
The authoritative warm-up is in each speed artifact. See the evidence
directory README for the complete policy. Rendered PNGs are excluded from git,
matching the existing repository policy.
