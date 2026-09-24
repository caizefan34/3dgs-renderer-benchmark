# C41: GPU Utilization Bottleneck Discovery — Final Report

## 1. GPU Utilization Table

| Component | Time % | ms/iter | Kernels/iter | BW (GB/s) | BW % of peak | Classification |
|-----------|--------|---------|-------------|-----------|-------------|----------------|
| **D-SSIM loss** | **42.5%** | 37.99 | 8.7 | 21.0 | **1.3%** | Algorithmic excessive-work bound |
| **Adam optimizer** | **18.9%** | 16.94 | 40.7 | 83.6 | **5.4%** | Kernel fragmentation bound |
| **Elementwise misc** | **17.6%** | 15.71 | 160.2 | 5.1 | **0.3%** | Kernel fragmentation (launch + small-tensor) |
| **rasterize_bwd** | **11.9%** | 10.66 | 1.1 | 25.2 | **1.6%** | Memory bound (AI=0.48 < ridge=12.5) |

**A100-PCIE-40GB reference**: Peak BW = 1555 GB/s, Peak FP32 = 19.5 TFLOPS, Peak TF32 = 156 TFLOPS, 108 SMs.

**Critical observation**: No component exceeds 6% of peak memory bandwidth. The GPU is severely underutilized across ALL components.

---

## 2. Bottleneck Ranking

### Rank 1: D-SSIM Loss (42.5%, 38.0ms/iter)

**Evidence:**
- 8.7 kernel calls/iter, mean duration 4.37ms each
- Achieved BW: 21.0 GB/s (**1.3% of peak**) — far from memory bound
- Achieved compute: 0.07 TFLOPS (**0.05% of TF32 peak, 0.36% of FP32 peak**)
- Neither memory nor compute bound → **algorithmic excessive-work bound**

**Root cause:**
cuDNN convolution kernels for D-SSIM are doing excessive algorithmic work relative to the actual data size. The D-SSIM loss computes:
- Gaussian blur via 2D convolution (3×3 or 5×5 kernel) on 1920×1080×3 image
- Multiple convolutions: blur rendered, blur GT, compute mean/variance/covariance maps
- Backward: gradient through all convolutions (dgrad + wgrad)

The data is only ~796 MB/iter (8 images × ~100 MB), but 38ms of compute is spent. This is because cuDNN optimizes for large-batch large-channel convolutions, but D-SSIM uses 3-channel (RGB) images with 3×3 kernels — a regime where cuDNN's tiling and parallelism strategies are suboptimal.

**Top kernels:**
```
cudnn::dgrad2d_grouped_direct_kernel: 20.8ms/iter (3.3 calls)  ← backward gradient
cudnn::conv2d_grouped_direct_kernel: 17.2ms/iter (5.4 calls)  ← forward blur
```

**Possible optimization mechanism:**
1. **Separable blur**: Replace 2D conv with two 1D convs → 3× fewer FLOPs for 3×3 kernel
2. **Downsampled SSIM**: Compute SSIM at 1/2 or 1/4 resolution → 4× to 16× savings
3. **Box filter approximation**: Replace Gaussian blur with box filter → no convolution needed
4. **Skip D-SSIM after convergence**: D-SSIM gradient diminishes after ~15K iters

**Theoretical speedup upper bound:**
- Separable blur: ~2-3× on the convolution itself → **~25ms saved (29% of total)**
- Downsampled SSIM (1/2): ~4× → **~28ms saved (33% of total)**
- Combined: could reduce D-SSIM from 38ms to ~10ms → **~28ms saved (33% of total)**

**Implementation difficulty:** LOW — Python-level loss function change, no CUDA modification.

**Research value:** MEDIUM — D-SSIM approximation is a known technique but measuring exact quality/speed tradeoff in 3DGS training context is novel evidence.

---

### Rank 2: Adam Optimizer (18.9%, 16.9ms/iter)

**Evidence:**
- 40.7 kernel calls/iter, mean duration 0.42ms, median 0.051ms
- Achieved BW: 83.6 GB/s (**5.4% of peak**)
- Launch overhead: 0.285ms (1.7% of Adam time) — NOT launch bound
- SH parameters dominate: **81.4% of total Adam bytes** (1152.8 MB vs 72 MB for xyz)

**Root cause:**
**Kernel fragmentation bound** — 41 separate kernels process 5 parameter groups with different shapes and learning rates. The largest group (SH: 1M × 16 × 3 = 48M elements) requires multiple kernels for the Adam update (m update, v update, bias correction, param update).

The achieved bandwidth (83.6 GB/s) is 5.4% of peak — not memory bandwidth limited. The issue is that 41 small-to-medium kernels each launch with insufficient parallelism for the A100's 108 SMs.

**Per-group breakdown:**
| Group | Size (MB) | % of total |
|-------|-----------|------------|
| xyz | 72.0 | 5.1% |
| rotations | 96.1 | 6.8% |
| scales | 72.0 | 5.1% |
| opacity | 24.0 | 1.7% |
| **shs** | **1152.8** | **81.4%** |

**Possible optimization mechanism:**
1. **Fused Adam**: Combine all 5 groups into a single kernel launch (PyTorch's `foreach=True` already does partially)
2. **Custom fused kernel**: Single CUDA kernel that processes all params with per-group lr
3. **SH parameter packing**: Reshape SH from [N, 16, 3] to [N, 48] for better memory coalescing

**Theoretical speedup upper bound:**
- Full fusion (1 kernel instead of 41): eliminate fragmentation → **~8ms saved (9% of total)**
- Realistic: 2× speedup on Adam → **~8.5ms saved (10% of total)**

**Implementation difficulty:** MEDIUM — requires custom CUDA kernel or careful PyTorch foreach optimization.

**Research value:** LOW — Adam fusion is well-known; the 81% SH dominance is a data layout insight but not novel.

---

### Rank 3: Elementwise Misc (17.6%, 15.7ms/iter)

**Evidence:**
- **160.2 kernel calls/iter** (highest kernel count of any category)
- Mean duration: 98.1μs, median: 45.8μs, p90: 218.4μs
- Achieved BW: 5.1 GB/s (**0.3% of peak**)
- Launch overhead estimate: 1.12ms (7.1% of elem time)

**Root cause:**
**Kernel fragmentation (launch + small-tensor)** — 160 tiny kernels each processing small tensors. The GPU spends most time on kernel launch overhead and underutilized SMs.

Sources of fragmentation:
- **Autograd graph**: Each primitive op (add, mul, div, clamp, sigmoid, exp) generates a separate kernel
- **GaussianModel.forward()**: normalize, exp, sigmoid, contiguous each launch kernels
- **L1 loss**: abs, sub, mean = 3 kernels
- **Gradient clipping**: norm computation + scaling = 4+ kernels
- **SH evaluation**: helper ops in Python autograd

**Top kernels:**
```
vectorized_elementwise_kernel (add):    3.16ms/iter  (16.4 calls)
elementwise_kernel (div):               3.00ms/iter  ( 9.8 calls)
vectorized_elementwise_kernel (mul):    1.91ms/iter  ( 9.9 calls)
vectorized_elementwise_kernel (fill):   1.10ms/iter  (23.1 calls)
AUnaryFunctor:                          1.10ms/iter  (12.1 calls)
```

**Possible optimization mechanism:**
1. **torch.compile**: Compile the training step to fuse elementwise ops
2. **Manual fusion**: Combine activation functions (sigmoid+exp+normalize) into one kernel
3. **Custom loss**: Fuse L1 + D-SSIM into single kernel

**Theoretical speedup upper bound:**
- torch.compile fusion: ~3× on elementwise → **~10ms saved (12% of total)**
- Realistic: 2× → **~8ms saved (9% of total)**

**Implementation difficulty:** LOW-MEDIUM — torch.compile is a one-line change, but may have compatibility issues with gsplat custom autograd.

**Research value:** LOW — kernel fusion via torch.compile is standard practice.

---

### Rank 4: rasterize_bwd (11.9%, 10.7ms/iter)

**Evidence:**
- 1.1 kernel calls/iter (single kernel)
- Mean duration: 9.69ms — the longest single kernel in the pipeline
- Achieved BW: 25.2 GB/s (**1.6% of peak**)
- Arithmetic intensity: **0.48 FLOPs/byte**
- Ridge point: 12.5 FLOPs/byte
- AI << ridge → **Memory bound**

**Root cause:**
The rasterize_bwd kernel processes ~3.2M intersections, each doing ~40 FLOPs of gradient computation while reading ~84 bytes of data. The arithmetic intensity (0.48) is 26× below the ridge point, meaning the kernel is severely memory-bound.

However, the achieved BW is only 1.6% of peak — the kernel is NOT saturating memory bandwidth. This suggests the bottleneck is **memory access pattern inefficiency** (random access to per-Gaussian state via flatten_ids) rather than raw bandwidth.

**Roofline analysis:**
```
Arithmetic Intensity = 0.48 FLOPs/byte
Ridge Point = 12.5 FLOPs/byte (FP32)
→ 26× below ridge → deeply memory bound
→ But only 1.6% of peak BW achieved → access pattern limited, not BW limited
```

The kernel's memory accesses are indirect (flatten_ids → gaussian_ids → per-Gaussian state), causing:
- L2 cache misses (random access pattern)
- Uncoalesced global memory reads
- Warp divergence in tile traversal

**Possible optimization mechanism:**
1. **Prefetching**: Preload next batch of GS state into shared memory
2. **Gradient accumulation**: Accumulate gradients in shared memory, write once
3. **Warp-level cooperation**: Share GS state across threads in same warp

**Theoretical speedup upper bound:**
- If BW utilization improved from 1.6% to 10%: ~6× speedup → **~9ms saved (10% of total)**
- Realistic: 2× → **~5ms saved (6% of total)**

**Implementation difficulty:** HIGH — requires CUDA kernel modification, deep understanding of memory access patterns.

**Research value:** MEDIUM — memory access pattern optimization for tile-based Gaussian rasterization backward is a real optimization problem, but requires CUDA expertise.

---

## 3. Candidate Decision

| Rank | Candidate | Time % | Max Speedup | Difficulty | Research Value | Decision |
|------|-----------|--------|-------------|------------|----------------|----------|
| 1 | **D-SSIM optimization** | 42.5% | ~28ms (33%) | LOW | MEDIUM | **KEEP** |
| 2 | Adam fusion/layout | 18.9% | ~8ms (10%) | MEDIUM | LOW | NEED MORE EVIDENCE |
| 3 | Elementwise fusion | 17.6% | ~8ms (9%) | LOW-MED | LOW | NEED MORE EVIDENCE |
| 4 | rasterize_bwd CUDA | 11.9% | ~5ms (6%) | HIGH | MEDIUM | DROP (for C42) |

### Decision: **KEEP D-SSIM optimization as C42 primary candidate**

**Rationale:**

1. **D-SSIM is the single largest bottleneck** at 42.5% of GPU time
2. **Algorithmic excessive-work bound** — the work itself is excessive, not a hardware limitation
3. **LOW implementation difficulty** — Python-level loss function change
4. **High theoretical speedup** — up to 33% of total training time
5. **Measurable quality tradeoff** — PSNR/SSIM comparison with different D-SSIM approximations provides clear research evidence

### Secondary candidate: Elementwise fusion via torch.compile

**Rationale:**
- 17.6% of GPU time, 160 kernels/iter
- torch.compile is a one-line change with potential 2× speedup
- If successful, combines with D-SSIM optimization for **~36ms total savings (42% of total)**
- Worth a quick test in C42

### Why NOT Adam or rasterize_bwd for C42:

- **Adam**: 5.4% BW utilization means the GPU is underutilized but not for a fixable reason — it's inherent to processing 5 separate parameter groups with different shapes. Custom CUDA fusion is medium difficulty for only 10% savings.

- **rasterize_bwd**: Memory bound at 1.6% BW utilization, but the root cause is random access patterns in the tile-based algorithm. Fixing this requires CUDA kernel modification (HIGH difficulty) for only 6% savings. The gsplat kernel is already well-optimized.

---

## Key Insight: GPU is Severely Underutilized

| Component | BW % of peak | Compute % of peak | Verdict |
|-----------|-------------|-------------------|---------|
| D-SSIM | 1.3% | 0.05% (TF32) | Algorithmic work, not HW limited |
| Adam | 5.4% | N/A (memory op) | Fragmentation, not BW limited |
| Elementwise | 0.3% | N/A | Fragmentation, not BW limited |
| rasterize_bwd | 1.6% | 0.06% (FP32) | Access pattern, not BW limited |

**No component is hardware-bound.** The GPU's peak capabilities (1555 GB/s, 19.5 TFLOPS) are nowhere near saturated. All bottlenecks are software-level: algorithmic excess, kernel fragmentation, and access pattern inefficiency.

This means **software-level optimizations** (loss function changes, kernel fusion, torch.compile) can yield significant speedups without any hardware changes or CUDA kernel modification.

---

## Artifacts

- Experiment script: `scripts/phase-c31/c41_gpu_utilization.py`
- Raw data: `results/phase-c31/c41_gpu_utilization.json`
- Chrome trace: `results/phase-c31/c41_trace.json`
