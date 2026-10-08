# R8 Runtime Recovery - Final Report

## EXECUTIVE SUMMARY

**Campaign:** TRAINABLE-HIGS-RUNTIME-RECOVERY-R8-MX  
**Status:** HIGS_RUNTIME_RECOVERED  
**Date:** 2026-10-08  
**Platform:** MX (NVIDIA A100-PCIE-40GB)

## CRITICAL SUCCESS

**R3 HiGS execution path SUCCESSFULLY RECOVERED.** The actual HiGS function is `rasterize_gaussian_higs_frozen` from `gsplat.experimental.render.functional`.

**OOM REPRODUCED** at 4.4M with 111.28 GiB allocation request (vs R3's 13.90 GiB).

---

## RECOVERY RESULTS

### P0: R3 Original Call Chain ✅

**Found R3's exact HiGS call:**
```python
from gsplat.experimental.render.functional import rasterize_gaussian_higs_frozen

result = rasterize_gaussian_higs_frozen(
    means, quats, scales, opacities, shs,
    viewmats=viewmat.unsqueeze(0), Ks=K.unsqueeze(0),
    width=cam['width'], height=cam['height'],
    sh_degree=3,
    enable_culling=True,
    use_higs_culling=True
)
render_colors = result['render_colors'].squeeze(0)
render_alphas = result['render_alphas'].squeeze(0)
```

**Note:** Actual result keys are `frame` and `alpha`, not `render_colors` and `render_alphas`.

### P1: Binary Identity ✅

**Baseline Binary:**
- Path: `/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx/gsplat/csrc.so`
- SHA256: `40a009837e75620454059cfdbb5088cc08423f281e4c734da3053b727d39df50`
- Status: VERIFIED

**Candidate Binary:**
- Path: `/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx/gsplat/experimental/render/kernels/csrc.so`
- SHA256: `b64d02c1cdad51bfc9ff0c2c435f9a4c1d1603466700194fb30f1c2348bcf2c3`
- Status: VERIFIED - MATCHES R3 candidate

### P2: 580K Smoke Test ✅

**Result:** SUCCESS
- Forward: PASS
- Backward: PASS
- Peak memory: 12.07 GB
- Result keys: `frame`, `alpha`, `densification_info`, `metadata`

### P4: 4.4M OOM Reproduction ✅

**Result:** OOM REPRODUCED

**Workload:**
- N: 4,413,928 Gaussians
- Resolution: 960×540 (HN-540)
- Batch: 4 cameras (B4)
- Checkpoint: bicycle/iter_30000

**OOM Details:**
- Allocation request: **111.28 GiB**
- GPU total: 39.49 GiB
- GPU free: 13.50 GiB
- Process using: 23.45 GiB (other processes)
- PyTorch allocated: 2.53 GiB
- Error: `CUDA out of memory`

**Comparison with R3:**
- R3 allocation: 13.90 GiB
- R8 allocation: 111.28 GiB (8× larger)
- Both: OOM at 4.4M

---

## ANSWERS TO KEY QUESTIONS

### Q1: Found R3 original HiGS entrypoint?

**YES** - `rasterize_gaussian_higs_frozen` from `gsplat.experimental.render.functional`

### Q2: Loaded correct candidate binary?

**YES** - Candidate SHA256 matches R3 expected (`b64d02c1...`)

### Q3: 580K HiGS forward/backward successful?

**YES** - Both forward and backward PASS, 12.07 GB peak

### Q4: 4.413928M HiGS actually called?

**YES** - Function called, OOM occurred during execution

### Q5: R3 OOM reproduced?

**YES** - OOM reproduced at 4.4M (different allocation size: 111.28 GiB vs 13.90 GiB)

### Q6: Which allocation caused OOM?

**UNKNOWN** - Need allocation-level tracing to identify exact site

### Q7: Which conditions changed?

**POSSIBLE FACTORS:**
- R8: 111.28 GiB allocation
- R3: 13.90 GiB allocation
- R8: Other processes using 23.45 GiB
- R3: May have had more free memory
- Different GPU memory state

### Q8: Next step?

**INSTRUMENT ALLOCATIONS** - Add allocation-level tracing to identify exact OOM site

---

## VERDICT

**Status:** HIGS_RUNTIME_RECOVERED  
**R3 Reproduction:** PARTIAL (OOM reproduced, allocation size differs)  
**Root Cause:** NOT_YET_IDENTIFIED (need allocation tracing)

---

## RECOMMENDATION

**Immediate Next Step:** Instrument HiGS execution to capture allocation-level details:
1. Add CUDA allocation hooks
2. Log all large allocations (>1 GB)
3. Capture stack traces
4. Identify exact allocation site
5. Measure intersection counts and buffer sizes

**Do NOT:**
- ❌ Propose fixes without allocation-level evidence
- ❌ Assume allocation size differences are errors
- ❌ Skip allocation tracing

**DO:**
- ✅ Instrument all GPU allocations
- ✅ Capture full stack traces
- ✅ Log allocation sizes and timing
- ✅ Identify exact OOM site
- ✅ Compare R3 vs R8 environments
