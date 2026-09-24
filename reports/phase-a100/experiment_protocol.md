# A100 Experiment Protocol — M1 Evidence Chain Completion

**Last Updated:** 2026-09-04  
**Host:** mx (bms-39468022-001) — A100-PCIE-40GB  
**Repo Commit:** `0237503` master

---

## 1. Pre-flight Checklist (Before Any GPU Experiment)

### 1.1 GPU Availability

```bash
nvidia-smi
```

Expected: At least **1 free GPU** with >3 GB available (500-step) or >10 GB (30K).  
If GPUs still occupied by vLLM, wait or coordinate with server admin.

### 1.2 Conda Environment

```bash
export PATH="$HOME/miniforge3/bin:$PATH"
conda activate anysplat
python -c "import torch; print(torch.__version__); import gsplat; print(gsplat.__version__)"
```

Expected: `torch 2.4.1+cu124`, `gsplat 1.5.3+pt24cu124`

### 1.3 Dataset Verification

```bash
# Check dataset exists
ls -la repo/data/official/mipnerf360/{room,bicycle,garden}/point_cloud.ply
ls -la repo/data/datasets/mipnerf360/{room,bicycle,garden}/images/ | head -5
```

Datasets must be placed at the expected paths. If not present, sync them.

### 1.4 Dataset Sync Commands

```bash
# Sync from local Windows to A100 via tar+ssh
# Data is at C:\Users\36570\3dgs-renderer-benchmark\data\
```

### 1.5 Build Verification

gsplat is already installed in the `anysplat` conda env. No build needed.

---

## 2. Evidence Chain Execution Order

### Phase A: New A100 Baseline Validation (GPU ≤ 10 min each)

| Step | Command | Expected |
|:-----|:--------|:---------|
| **Smoke** | `python -c "import torch, gsplat; ..."` | Render OK |
| **Forward** | Run Phase 8B snapshot fwd at tile16 (room) | Matches old A100 pattern |
| **Fwd+Bwd** | Same with backward | Stable, no NaN |

### Phase B: 500-Step Short Training (~15 min × 9 = ~2.5 hours)

Run sequentially (or parallel across GPUs if multiple available):

| # | Scene | tile_size | Command | Est. Time |
|:-:|:-----:|:---------:|:--------|:---------:|
| 1 | room | 16 | `run_full.py --scene room --tile-size 16 --steps 500` | ~5 min |
| 2 | room | 20 | same | ~5 min |
| 3 | room | 32 | same | ~5 min |
| 4 | bicycle | 16 | `--scene bicycle --tile-size 16 --steps 500` | ~15 min |
| 5 | bicycle | 20 | same | ~15 min |
| 6 | bicycle | 32 | same | ~15 min |
| 7 | garden | 16 | `--scene garden --tile-size 16 --steps 500` | ~15 min |
| 8 | garden | 20 | same | ~15 min |
| 9 | garden | 32 | same | ~15 min |

**Total:** ~2.5 hours (can be parallelized to ~30 min with 4+ GPUs)

**Key metrics to record per run:**
- Total wall time
- Average/median iteration time
- Forward/backward/optimizer time
- Peak VRAM
- Best PSNR
- Final Gaussian count
- NaN/Inf
- Densification events
- Checkpoint validity

### Phase C: 30K Full Training (2-8 hours each)

**Selection Criteria** (based on 500-step results):
- **Must run:** tile16 (baseline) and tile32 (candidate) for both bicycle and garden
- **Optionally run:** tile20 if 500-step shows stable AND promising performance
- **Skip tile20 on A100 if:** 500-step shows topology spillover (like RTX5070 Phase 14A)

| # | Scene | tile_size | Priority | Est. Time (A100) |
|:-:|:-----:|:---------:|:--------:|:----------------:|
| 1 | bicycle | 16 | **CRITICAL** | ~6-8 hours |
| 2 | bicycle | 32 | **CRITICAL** | ~4-6 hours (if tile32 is faster) |
| 3 | garden | 16 | **CRITICAL** | ~5-7 hours |
| 4 | garden | 32 | **CRITICAL** | ~3-5 hours |
| 5 | room | 20 | LOW | ~2-3 hours (only if tile20 resolves anomaly) |

**Optimal parallel strategy (4 GPUs):**
```
GPU 0: bicycle tile16 30K
GPU 1: bicycle tile32 30K  
GPU 2: garden tile16 30K
GPU 3: garden tile32 30K
```

---

## 3. Data Collection Protocol

### Per Experiment JSON Schema

```json
{
  "experiment": {
    "label": "a100_30k_bicycle_t16",
    "scene": "bicycle",
    "tile_size": 16,
    "steps": 30000,
    "seed": 42,
    "timestamp_start": "ISO8601",
    "timestamp_end": "ISO8601"
  },
  "environment": {
    "host": "mx",
    "gpu": "A100-PCIE-40GB",
    "driver": "595.71.05",
    "cuda": "12.4",
    "pytorch": "2.4.1+cu124",
    "gsplat": "1.5.3+pt24cu124",
    "repo_commit": "0237503"
  },
  "results": {
    "total_wall_s": 0.0,
    "avg_iter_ms": 0.0,
    "median_iter_ms": 0.0,
    "p95_iter_ms": 0.0,
    "p99_iter_ms": 0.0,
    "fwd_ms": 0.0,
    "bwd_ms": 0.0,
    "opt_ms": 0.0,
    "peak_vram_mb": 0,
    "initial_gaussian_count": 0,
    "final_gaussian_count": 0,
    "best_psnr": 0.0,
    "final_psnr": 0.0,
    "has_nan": false,
    "has_inf": false
  },
  "evidence_class": "OBSERVED"
}
```

### Output Location

```
/mnt/storage_pool/3dgs-renderer-benchmark/results/training/
├── a100_500step_room_t16_results.json
├── a100_500step_room_t20_results.json
├── ...
├── a100_30k_bicycle_t16_results.json
├── a100_30k_bicycle_t32_results.json
├── a100_30k_garden_t16_results.json
└── a100_30k_garden_t32_results.json
```

---

## 4. Decision Rules During Execution

### Stopping Conditions

| Condition | Action |
|:----------|:-------|
| **500-step shows NaN/Inf** | Stop immediately, investigate, document as BLOCKED |
| **500-step OOM** | Document as BLOCKED with VRAM usage |
| **500-step tile20 topology spillover** | Drop tile20 from 30K selection, document anomaly |
| **30K training crashes >2 times** | Document as FAILED with crash details |
| **tile16 and tile32 both complete** | Proceed to scene×hardware analysis |
| **Only one tile completes** | Document partial evidence |

### Classification Tags

All results must include one of:
- `OBSERVED`: Directly measured
- `REPRODUCED`: Confirms previous finding on new hardware
- `SUPPORTED`: Consistent with multiple measurements
- `BLOCKED`: Cannot be executed
- `FAILED`: Execution failed

---

## 5. Dataset Sync Quick Reference

### Local → Remote Sync

From Windows (PowerShell):
```powershell
# Create tar pipe through SSH (must run from repo root parent)
tar -c -C "C:\Users\36570\3dgs-renderer-benchmark\data" . | `
  ssh -o ConnectTimeout=10 -o BatchMode=yes mx `
  "cd /mnt/storage_pool/3dgs-renderer-benchmark/repo/data && tar xf -"
```

### Sync Only Official Datasets

```powershell
# Copy just Mip-NeRF 360 dataset (room, bicycle, garden)
tar -c -C "C:\Users\36570\3dgs-renderer-benchmark\data" `
  "official/mipnerf360/room" "official/mipnerf360/bicycle" "official/mipnerf360/garden" `
  "datasets/mipnerf360/room" "datasets/mipnerf360/bicycle" "datasets/mipnerf360/garden" | `
  ssh -o ConnectTimeout=10 -o BatchMode=yes mx `
  "cd /mnt/storage_pool/3dgs-renderer-benchmark/repo/data && tar xf -"
```
