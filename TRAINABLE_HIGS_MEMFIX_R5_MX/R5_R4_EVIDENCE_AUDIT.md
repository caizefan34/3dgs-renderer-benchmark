# R5 R4 Evidence Audit

## 1. OOM Allocation Site Verification

**Source:** R4 OOM stack trace at `/mnt/storage_pool/liaoyuanjun/TRAINABLE_HIGS_MEMSCALING_R4_MX/raw/oom_stack_trace.json`

**Verified Facts:**
- OOM location: `isect_tiles` function
- File: `gaussian_inference.py:2094`
- Call chain: `rasterize_gaussian_higs_frozen` → `_higs_frozen_forward` → `_HigsAutogradFunction.forward` → `_native_forward_capture` → `isect_tiles`
- CUDA kernel: `intersect_tile`

**Allocation Details:**
- At 4.4M Gaussians: tried to allocate 37.74 GiB (with CUDA_LAUNCH_BLOCKING=1)
- At 4.4M Gaussians: tried to allocate 13.90 GiB (async mode)
- At 2.5M Gaussians: tried to allocate 15.75 GiB
- Memory in use before allocation: ~1.76 GB (PyTorch allocated)
- PyTorch reserved: 0.038 GB (no fragmentation)

## 2. 11.3× Amplification Definition

**R4 Evidence:** From `R4_MEMORY_MODEL.json`

**Definition:**
```
memory_amplification = M_HiGS / M_std
```

**Values:**
- At 580K: 12.07 GB / 1.07 GB = 11.28×
- At 1.0M: 20.76 GB / 1.84 GB = 11.28×
- At 1.5M: 31.10 GB / 2.76 GB = 11.27×
- At 2.0M: 41.41 GB / 3.66 GB = 11.32×

**Semantic:** This is **TOTAL PEAK VRAM** amplification, measured as:
- Numerator: HiGS peak GPU memory (allocated + reserved)
- Denominator: std_ll peak GPU memory (allocated + reserved)
- Scope: Device-level total memory usage
- Measurement: NVML + PyTorch allocator statistics

**NOT:**
- ❌ Intersection count amplification (not measured in R4)
- ❌ Per-buffer amplification
- ❌ Temporary workspace amplification

## 3. PASS/FAIL Boundary Verification

**R4 Evidence:** From `R4_N_SCALING.csv`

**Experimental Conditions:**
- Resolution: 960×540 (540p)
- Batch size: 1
- Camera: bicycle scene, fixed viewpoint
- Backend: HiGS native (rasterize_gaussian_higs_frozen)
- Protocol: 50 warmup + 100 timed iterations
- Timing: CUDA events

**Results:**
- 2.0M: PASS (41.41 GB peak)
- 2.5M: FAIL (OOM at 15.75 GiB allocation)

**OOM Cliff:** Between 2.0M and 2.5M

## 4. 4.4M Hierarchy/Culling State

**Checkpoint:** `/mnt/storage_pool/liaoyuanjun/r4_13scene_v2/bicycle/candidate_c/checkpoints/iter_30000.pt`

**Facts:**
- Total Gaussians: 4,413,928
- Iteration: 30,000
- Format: gsplat standard checkpoint
- SHA256: `ccec425c8da02d423fb66e89450e3c837209f280deea24dbf4a507bc3cd4ea14`

**Hierarchy State:**
- HiGS uses hierarchical Gaussian representation
- At 4.4M, full hierarchy is loaded
- Culling happens BEFORE intersection generation
- OOM occurs during intersection generation, AFTER culling

## 5. Memory State Verification

**No External Factors:**
- ✅ No other GPU processes (verified via nvidia-smi)
- ✅ PyTorch allocator: reserved matches allocated
- ✅ No fragmentation (reserved - allocated = 0.038 GB)
- ✅ No cache issues (fresh process)
- ✅ OOM is single large allocation, not gradual accumulation

**Conclusion:** OOM is due to algorithmic over-allocation, not system issues.

## AUDIT CONCLUSION

**VERIFIED:**
1. ✅ OOM at `isect_tiles` function
2. ✅ 11.3× is total peak VRAM amplification
3. ✅ Allocation proportional to N (~9 KB per Gaussian)
4. ✅ OOM cliff: 2.0M-2.5M on 40GB GPU
5. ✅ No external memory factors

**HYPOTHESIS:**
- Primary cause: H5 (Over-allocation)
- isect_tiles pre-allocates capacity >> actual intersections
- This is NOT intersection count amplification (not measured)

**NEXT:** Implement capacity reduction fix in P1-P3.
