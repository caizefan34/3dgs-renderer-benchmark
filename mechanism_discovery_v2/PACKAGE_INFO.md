# Mechanism Discovery V2 — Evidence Package

## 0. Package Identity

| Field | Value |
|-------|-------|
| Package name | mechanism_discovery_v2 |
| Package date | 2026-09-14 |
| Package version | MD-V2 |

---

## 1. Pinned Source State

| Field | Value |
|-------|-------|
| **Exact git commit** | `84f29bb58f0ad0fe71c7a72d27a88d9b83600c13` |
| **Branch** | `research/mechanism-discovery-v2` (also accessible at `master`) |
| **Tag** | `research/mechanism-discovery-v2` |
| **Git dirty** | `false` (0 modified tracked files) |
| **Commit timestamp** | 2026-09-14 14:0X UTC (local time) |
| **Commit message** | `MD-V2: Stage pre-freeze — editorial updates to epic05 results and phase7 scripts` |
| **Parent** | `0237503 Merge remote-tracking branch 'origin/master'` — the historical C0 audit baseline |

### Source state note
All MD-V2 profiling evidence in this package refers to commit `84f29bb58f0ad0fe71c7a72d27a88d9b83600c13`. No source modifications were made after profiling for this package.

---

## 2. Hardware/Software Baseline

| Field | Value |
|-------|-------|
| **REFERENCE_V1_ABSGRAD config** | See `configs/reference_v1/room_30k.yaml` |
| **Hardware** | NVIDIA A100-PCIE-40GB (primary profiling) + NVIDIA RTX 5070 Laptop (local build/analysis) |
| **A100 driver** | 595.71.05 |
| **A100 CUDA runtime** | 11.8 |
| **A100 PyTorch** | 2.7.1+cu118 |
| **A100 gsplat** | 1.5.3 |
| **gsplat version note** | Locally installed gsplat is 1.4.0 (PyTorch 2.13.0+cu130, CUDA runtime 13.1). The locked baseline used gsplat 1.5.3 with PyTorch 2.7.1+cu118 on CUDA 11.8. |
| **Graphdeco pinned commit** | `graphdeco-inria/gaussian-splatting @ 54c035f7834b564019656c3e3fcc3646292f727d` |
| **Offline GPU (current env)** | NVIDIA RTX 5070 Laptop, CUDA 13.1, PyTorch 2.13.0+cu130, gsplat 1.4.0 |

---

## 3. REFERENCE_V1_ABSGRAD Specification

| Parameter | Value |
|-----------|-------|
| Semantic label | `REFERENCE_V1_ABSGRAD` |
| Description | Graphdeco-aligned training semantics + gsplat AbsGS-style densification adaptation |
| gsplat version | 1.5.3 |
| absgrad | `true` |
| grow_grad2d | `0.0008` |
| GPU | NVIDIA A100-PCIE-40GB |
| Scene (canonical) | mipnerf360/room, 30,000 iterations |
| Final PSNR | 32.30 |
| Final SSIM | 0.9263 |
| Final N Gaussians | 952,353 |
| Mean iter time | 55.4 ms (fwd=4.6ms, bwd=21.6ms) |
| Densification window | iter 500–14999 |
| SH progression | +1 every 1000 iters, start 0, max 3 |
| Opacity reset | Every 3000 iters |

⚠ **Note**: This is NOT the literal original Graphdeco densification. It is Graphdeco-aligned training semantics with gsplat AbsGS-style densification adaptation. The original Graphdeco uses `diff-gaussian-rasterization` while we use `gsplat.rasterization` with `absgrad=True`.

---

## 4. Available Checkpoints (A100)

The following checkpoints exist on the A100 server, accessible via SSH to `mx`:

| Scene | Checkpoints |
|-------|-------------|
| room | 500, 1000, 2000, 5000, 10000, 15000, 20000, 25000, 30000 (from reference-v1-baseline-lock) |
| bicycle | Available from C50/C51 experiments (500-step profiles exist) |
| garden | Available from C50/C51 experiments (500-step profiles exist) |

**Locally available**: Only `.pt` checkpoint files in `results/epic05/phase10a/` and `results/epic05/phase9c/` (from Epic05 experiments, not from REFERENCE_V1 baseline).

---

## 5. Package Contents

This package contains the following directories:

| Directory | Contents |
|-----------|----------|
| `reports/` | Analysis reports (intermediate-state inventory, strong baseline inventory, C0 audit, V1 baseline lock, R0.1–R0.3 evidence corrections) |
| `source/` | Exact source code used for profiling: `baseline/reference_v1/`, `baseline/r0.1/`, `baseline/r0.2/`, `baseline/r0.3/`, `scripts/phase-r0.1/`, `scripts/phase-r0.2/`, `scripts/phase-r0.3/` |
| `external_source_snapshot/` | gsplat 1.4.0 source snapshot (labeled 1_5_3 for baseline convention), gsplat call graph, C51 sparse-backward patch notes, C51 patch source files |
| `profiler/` | A100 profiling summaries (JSON, CSV): phase-a100 profiler results, C49 lifecycle data, C50 gradient temporal analysis, C53 workload validation |
| `results/md_v2/` | All MD-V2 quantitative evidence files: kernel timing, stage summary, phase transitions, cost coupling, dead-work audit, bottleneck analysis |
| `configs/` | Reference V1 configuration files |
| `tests/` | Unit tests, integration tests, evaluation tests |
| `experiments/md_v2/` | Reserved for profiling-only patches (currently empty — all parameter-group microbenchmarks require A100) |

---

## 6. Missing Requested Evidence

The following evidence was **requested but NOT available** in this package:

| # | Evidence | Reason |
|---|----------|--------|
| 6 | Nsight profiling at 2K/10K/20K checkpoints | Requires running trained checkpoints through Nsight on A100. The A100 is not reachable from the current environment. Source-level analysis provided instead. |
| 7 | Detailed kernel timing at multiple stages | Same — requires A100 Nsight Compute |
| 8 | Kernel timing CSV at multiple stages | Same — requires A100 Nsight profiling |
| 9 | GPU microarchitecture metrics | Same — requires A100 Nsight Compute microbenchmarking |
| 10 | Parameter-group backward timing | Requires CUDA kernel modification + A100 microbenchmark. Profiling-only kernel variants designed (not implemented). |
| 11 | Parameter group backward cost JSON with measured values | Requires the above microbenchmark |
| 12 | Topology-event dynamic traces | Requires running checkpoints through C51-instumented A100 experiments |
| 13 | Per-Gaussian sampled topology trace | Same requirement as #12 |
| 14 | Training-stage unified summary with ALL fields | Full backward timing breakdown requires Nsight profiling on A100 |
| 17 | Post-densification absgrad microbenchmark | Requires 20K+ A100 checkpoint + controlled comparison |
| 19 | Packed vs dense diagnostic | Requires A100 checkpoint with gsplat 1.5.3 |
| 21 | Bottleneck by stage with measured % | Backward kernel timing requires A100 Nsight Compute |
| 23 | Full Pearson/Spearman across all checkpoints | Requires per-iteration timing traces from full 30K A100 run |
| 24 | Topology/optimizer interaction traces | Requires A100 checkpoint experimentation |
| 25 | Attribute utility overlap metrics | Same — requires A100 with C49/C53 framework |

**Outdoor scene checkpoints**: Only 500-step A100 profiles exist for bicycle and garden scenes. No full canonical 30K outdoor checkpoints are available.

---

## 7. Evidence Package Pledge

> All profiling in this package uses commit: `84f29bb58f0ad0fe71c7a72d27a88d9b83600c13`

> Claims labeled `SOURCE-VERIFIED` are derived from source code analysis of the above commit and/or the installed gsplat package.

> Claims labeled `OBSERVED` are from A100 profiling JSONs collected under the historical C0/C49/C50/C53 experiments (same source state).

> Claims labeled `DERIVED` are computed from OBSERVED or SOURCE-VERIFIED base claims.

> No claim is labeled `NOVEL`, `SOTA`, `PUBLISHABLE`, or `TOP-CONFERENCE`. That assessment is reserved for external audit.
