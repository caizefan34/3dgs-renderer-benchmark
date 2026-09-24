# Sorting Pipeline — External Research Queue

**Date:** 2026-10-19  
**Purpose:** Unexplained sorting cost components requiring external literature or documentation search.
**Important correction from deep analysis:** The CUB sort itself is LINEAR and well-characterized (~0.6 µs/isect, stable across checkpoints). The main unexplained cost is autograd/allocation overhead (37% of forward). Research priorities reflect this.

---

## Priority 1 (HIGH): PyTorch Autograd/Allocation Overhead in Batch Rasterization

### Research Question

> What causes PyTorch autograd graph construction and CUDA memory allocation to scale super-linearly (18.7× for 4× input growth) in the gsplat rasterization forward pipeline?

### Known Evidence

| Aspect | Tile16 (N_isects=176M) | Tile32 (N_isects=44M) | Ratio |
|:-------|:---------------------:|:--------------------:|:-----:|
| Kernel-level forward sum | 125.2ms | 36.4ms | 3.44× |
| Monolithic forward (Phase 8C) | 200.1ms | 41.1ms | 4.87× |
| Residual (autograd + alloc) | **74.9ms** | **4.7ms** | **15.9×** |
| Residual % of forward | **37.4%** | **9.7%** | — |

The residual overhead is 15.9× higher for 4× more intersections. This includes:
- PyTorch autograd graph edge construction (inputs → outputs tracking)
- CUDA tensor allocations (isect_ids, flatten_ids, sorted copies, CUB temp storage)
- PyTorch caching allocator management (GC/malloc/free cycles)
- Intermediate tensor padding for SH degree / channel alignment

### Search Queries

```text
- PyTorch autograd overhead large tensor scaling 100 million elements
- torch.cuda.CUDACachingAllocator fragmentation large allocation thrashing
- PyTorch autograd graph construction time scales superlinearly
- gsplat rasterization forward allocation overhead optimization
- PyTorch no_grad context manager allocation overhead vs autograd
- PyTorch CUDA allocator multiple large tensors fragmentation GC
- 3DGS differentiable rasterizer training overhead breakdown
```

### What Must Be Verified Externally

1. Is the PyTorch caching allocator known to fragment when multiple 2+ GB allocations occur?
2. Does the autograd graph construction scale linearly or super-linearly with number of intermediate tensors?
3. Are there known workarounds (pre-allocated buffers, tensor recycling, `torch.no_grad()` context)?
4. Does the 3DGS community (gsplat issues, diff-gaussian-rasterization) report similar overhead?
5. Are there gsplat-specific allocation patterns (e.g., channel padding) that cause unnecessary allocation overhead?

---

## Priority 2 (MEDIUM): CUB DeviceRadixSort Internal Policy and RADIX_BITS

### Research Question

> What is the actual RADIX_BITS used by the installed CCCL/CUB that gsplat v1.5.3 compiles against on sm_80 (RTX 5070)?

### Known Evidence

- C1: end_bit = 16 + tn + in = 30 (1080p tile16, I=1)
- Pass count: P = ceil(30 / RADIX_BITS). Either 8 (RB=4) or 4 (RB=8).
- Total CUB memory traffic: 24 × N × P. This is either **33.7 GB** (P=8) or **16.9 GB** (P=4).
- 93.5ms sort time at 81% BW utilization suggests P=4 is more likely (4 × 4.22GB = 16.9GB → 181 GB/s = 40% BW), while P=8 gives 33.7GB → 361 GB/s = 81% BW.
- The 81% BW figure strongly suggests P ≥ 8 (more traffic needed to saturate BW)

### Search Queries

```text
- CUB DeviceRadixSort RADIX_BITS default value sm_80 CUDA 13
- CCCL cub radix sort RADIX_BITS policy CUB_RADIX_BITS
- cub DeviceRadixSort bits per pass compile-time parameter
- nvidia CUB radix sort pass count formula implementation
- CUB radix sort sm_80 architecture kernel selection
```

### What Must Be Verified Externally

1. Default RADIX_BITS for CCCL/CUB 2.x on sm_80
2. Whether RADIX_BITS is declared as a macro in cub/agent/agent_radix_sort.h
3. The typical CUB pass count for a 30-bit sort on recent GPU architectures
4. Whether CUB uses RADIX_BITS=8 for >= sm_70 by default

---

## Priority 3 (LOW): Cross-Scene CUB Sort Scaling Validation

### Research Question

> Does the CUB sort linear scaling (0.53–0.67 µs/isect at ~81% BW) generalize to scenes with different intersection distributions (bicycle, garden, treehill)?

### Known Evidence

- Phase 8E only profiled the room scene (N=1.2M Gs, 100% tile occupancy)
- Other scenes have different mean Gaussians/tile (room: 490, bicycle: 746, garden: 751) and max Gaussians/tile (room: 2,996, bicycle: 6,882, garden: 4,361)
- Room scene has uniform tile coverage; bicycle/garden may have non-uniform coverage from foreground/background separation

### Search Queries

```text
- 3DGS mipnerf 360 sorting workload bicycle garden room comparison
- gsplat performance scene-dependent sort time variation
- 3D Gaussian splatting non-uniform tile distribution sort efficiency
```

### What Must Be Verified Externally

1. Are there published benchmarks comparing sort time across different 3DGS scenes?
2. Does non-uniform tile coverage affect CUB sorting efficiency?

---

## Summary

| Priority | Question | Gap Area | Impact on Optimization |
|:--------:|:---------|:---------|:-----------------------|
| **1** | What causes 37% autograd/alloc overhead in tile16? | PyTorch overhead | **HIGH** — 37% of forward is the largest unaddressed cost |
| **2** | What is the actual CUB RADIX_BITS? | CUB policy | **MEDIUM** — determines whether sort is 4 or 8 passes |
| **3** | How does CUB sort scale on non-room scenes? | Scene generalization | **LOW** — sort is already well-characterized as linear |

---

*End of Sorting External Research Queue*
