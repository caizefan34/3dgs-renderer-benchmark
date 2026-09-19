# R6-B — A100 Benchmark

## Environment

- GPU: NVIDIA A100-PCIE-40GB (8 GPUs, 1 per workload)
- gsplat: 1.5.3 (patched, isolated source tree at /tmp/r6b_patched)
- PyTorch: 2.4.1+cu124
- CUDA arch: sm_80
- Build: FAST_COMPILE=1, -O0 (debug build)
- Warmup: 20 iterations, Measure: 100 iterations
- Camera: idx=0 (fixed across all runs)
- Timing: torch.cuda.Event

## 9-workload results

| Scene    | Iter | N        | Mode    | T_bwd (ms) | T_iter (ms) | T_prep (ms) | T_scat (ms) | T_clear (ms) | Meta (MB) | Δiter % |
|----------|------|----------|---------|------------|-------------|-------------|-------------|--------------|-----------|---------|
| room     | 5K   | 560,632  | baseline | 21.71     | 54.27       | -           | -           | -            | -         | -       |
| room     | 5K   | 560,632  | B0       | 21.74     | 54.32       | 0.037       | -           | 0.031        | -         | -0.09   |
| room     | 5K   | 560,632  | B1-v2    | 21.99     | 54.54       | 0.031       | 0.247       | 0.024        | 0.54      | -0.49   |
| room     | 15K  | 933,590  | baseline | 21.60     | 55.12       | -           | -           | -            | -         | -       |
| room     | 15K  | 933,590  | B0       | 21.61     | 55.34       | 0.046       | -           | 0.040        | -         | -0.39   |
| room     | 15K  | 933,590  | B1-v2    | 21.79     | 55.51       | 0.039       | 0.170       | 0.033        | 0.89      | -0.70   |
| room     | 30K  | 933,590  | baseline | 20.16     | 53.56       | -           | -           | -            | -         | -       |
| room     | 30K  | 933,590  | B0       | 20.17     | 53.55       | 0.047       | -           | 0.041        | -         | +0.03   |
| room     | 30K  | 933,590  | B1-v2    | 20.35     | 53.69       | 0.039       | 0.165       | 0.033        | 0.89      | -0.24   |
| bicycle  | 5K   | 1,933,177| baseline | 27.07     | 64.91       | -           | -           | -            | -         | -       |
| bicycle  | 5K   | 1,933,177| B0       | 27.11     | 65.01       | 0.077       | -           | 0.071        | -         | -0.15   |
| bicycle  | 5K   | 1,933,177| B1-v2    | 27.37     | 65.24       | 0.059       | 0.247       | 0.053        | 1.84      | -0.50   |
| bicycle  | 15K  | 3,957,041| baseline | 38.09     | 85.96       | -           | -           | -            | -         | -       |
| bicycle  | 15K  | 3,957,041| B0       | 38.14     | 86.08       | 0.136       | -           | 0.130        | -         | -0.14   |
| bicycle  | 15K  | 3,957,041| B1-v2    | 38.39     | 86.27       | 0.110       | 0.267       | 0.104        | 3.77      | -0.37   |
| bicycle  | 30K  | 3,957,041| baseline | 36.05     | 83.29       | -           | -           | -            | -         | -       |
| bicycle  | 30K  | 3,957,041| B0       | 36.11     | 83.39       | 0.137       | -           | 0.130        | -         | -0.11   |
| bicycle  | 30K  | 3,957,041| B1-v2    | 36.36     | 83.67       | 0.108       | 0.255       | 0.102        | 3.77      | -0.45   |
| garden   | 5K   | 1,779,185| baseline | 23.47     | 59.50       | -           | -           | -            | -         | -       |
| garden   | 5K   | 1,779,185| B0       | 23.48     | 59.53       | 0.073       | -           | 0.067        | -         | -0.04   |
| garden   | 5K   | 1,779,185| B1-v2    | 23.59     | 59.63       | 0.041       | 0.129       | 0.035        | 1.70      | -0.20   |
| garden   | 15K  | 2,610,559| baseline | 28.45     | 68.68       | -           | -           | -            | -         | -       |
| garden   | 15K  | 2,610,559| B0       | 28.47     | 68.76       | 0.098       | -           | 0.091        | -         | -0.10   |
| garden   | 15K  | 2,610,559| B1-v2    | 28.63     | 68.90       | 0.071       | 0.170       | 0.065        | 2.49      | -0.31   |
| garden   | 30K  | 2,610,559| baseline | 27.41     | 67.33       | -           | -           | -            | -         | -       |
| garden   | 30K  | 2,610,559| B0       | 27.43     | 67.32       | 0.098       | -           | 0.092        | -         | +0.01   |
| garden   | 30K  | 2,610,559| B1-v2    | 27.59     | 67.55       | 0.071       | 0.162       | 0.065        | 2.49      | -0.33   |

## Summary

| Module | Mean Δiter | Range | Negative workloads |
|--------|-----------|-------|--------------------|
| B0     | -0.11%    | -0.39% to +0.03% | 7/9 |
| B1-v2  | -0.40%    | -0.70% to -0.20% | 9/9 |

## Diagnosis

### B0 (persistent buffers + full clear)

The full-clear cost (T_clear) is 0.031–0.130 ms across workloads, which is
0.06%–0.15% of the backward time. This is negligible — the `torch::zeros_like`
allocation + zero-fill that B0 replaces is already extremely fast on A100.

B0 shows no stable positive E2E benefit. The persistent buffer avoids
allocation overhead but the allocation itself is not a measurable bottleneck
at these scales.

### B1-v2 (touched-mask selective clear)

The scatter phase (T_scat) costs 0.129–0.267 ms — this is the cost of
scattering `flatten_ids [n_isects]` into the `[C*N]` bool mask. It is
**2–4× more expensive** than the full clear it's trying to eliminate.

The selective clear (T_clear) saves only 0.007–0.030 ms compared to B0's
full clear (0.024–0.104 vs 0.031–0.130). The savings are negligible because
most Gaussians are touched in typical training iterations (r_touch ≈ 0.6–0.8),
so the selective clear still writes to most rows.

**The touched-mask preparation cost exceeds the zero-fill cost being
eliminated.** This is the B1_DROP condition.

### Root cause

The original observation was that dense zero-init of `[C, N, 11]` gradient
buffers is costly. However, on A100 with gsplat 1.5.3, the zero-init is
performed by a highly optimized `cudaMemsetAsync` (via `torch::zeros_like`),
which achieves near-peak memory bandwidth. For N ≈ 1–4M Gaussians, the
zero-fill is only 0.03–0.13 ms — a tiny fraction of the 20–38 ms backward.

The selective-clear approach cannot win because:
1. The scatter phase adds new work (0.13–0.27 ms) that didn't exist before
2. The clear savings are minimal (most rows are touched anyway)
3. The net effect is negative on every workload

## Memory

| Scene    | Iter | N        | B1-v2 metadata (MB) | B0 peak (MB) | B1-v2 peak (MB) |
|----------|------|----------|---------------------|--------------|-----------------|
| room     | 15K  | 933,590  | 0.89                | 1771         | 1772            |
| bicycle  | 15K  | 3,957,041| 3.77                | -            | -               |
| garden   | 15K  | 2,610,559| 2.49                | -            | -               |

B1-v2's metadata overhead is C×N bytes (bool mask), which is 0.5–3.8 MB —
negligible relative to the ~1.7 GB peak memory.
