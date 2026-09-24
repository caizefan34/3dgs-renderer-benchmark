# C42: D-SSIM Optimization Hypothesis Matrix

## Context

D-SSIM loss consumes 42.5% of GPU training time (38.0 ms/iter). The implementation audit (`reports/phase-c31/c42_dssim_audit.md`) identified three root causes:

1. **2 of 5 forward convolutions compute constant terms** (lines 47, 53: `blur(target)`, `blur(target²)`)
2. **5.5× FLOP inflation** from applying a separable kernel as full 2D convolution (line 37 constructs `kernel_1d ⊗ kernel_1d`, line 44 applies it as 2D)
3. **~30 unfused elementwise kernels** for SSIM map arithmetic (lines 48-60)

**Critical context discovered during audit**: The training loop (`train_3dgs.py` line 272) uses round-robin camera selection (`cam_idx = iteration % num_cameras`, 311 cameras). The target image is **NOT constant across iterations** — it changes every iteration. The "constant target" observation from C40/C41 profiling applies only to the fixed-camera profiling setup, not to actual training.

This distinction is fundamental: it changes the caching strategy from trivial (fixed-camera) to non-trivial (per-camera caching with memory implications).

---

## Hypothesis Matrix

| Candidate | Mechanism | Prior-art status | Expected speedup | Quality risk | Research value | Decision |
|---|---|---|---|---|---|---|
| **A: Cache constant terms** | Cache Gaussian kernel (always constant); cache blur(target)/blur(target²) per-camera with LRU | Known optimization | D-SSIM: ~15-18% (kernel always; target terms depend on cache hit rate) Total: ~6-8% | Mathematically identical | LOW | **KEEP — safe baseline** |
| **B: Separable convolution** | Replace 1×F.conv2d(11×11) with 2×F.conv2d(1×11)+(11×1); FLOPs 121→22 per pixel | Known optimization (textbook DSP) | D-SSIM: ~50-70% if cuDNN cooperates; ~20-30% if it doesn't Total: ~20-30% | Mathematically identical (bit-exact for float32) | LOW (proving root cause) | **KEEP — primary candidate** |
| **C: torch.compile fusion** | Compile loss function; fuses ~30 elementwise kernels into ~5; may optimize conv dispatch | Known optimization (PyTorch 2.x) | D-SSIM: ~10-20% (elementwise portion) Total: ~5-10% | Mathematically identical | LOW | **KEEP — additive** |
| **D: Downsampled D-SSIM** | Downsample pred+target to 1/2 or 1/4 before SSIM; 4× or 16× fewer pixels | Engineering application | D-SSIM: 4× (1/2 res) to 16× (1/4 res) Total: ~30-38% | Training behavior change (less high-freq structural detail) | MEDIUM | **MAYBE — needs quality evidence** |
| **E: Adaptive D-SSIM scheduling** | Reduce λ_dssim over training phases; or compute D-SSIM every N iterations | Potentially novel training strategy | D-SSIM: 2× (every-other) to 10× (phase-out) Total: ~20-38% | Training behavior change (different optimization trajectory) | HIGH | **KEEP — research contribution** |
| **F: Hybrid A+B+E** | Safe engineering floor (A+B) + training schedule (E) on optimized baseline | Hybrid: known + potentially novel | Total: ~35-50% | A+B identical; E adds behavior change | MEDIUM-HIGH | **KEEP — final synthesis** |

---

## Detailed Candidate Analysis

### Candidate A: Cache Constant SSIM Terms

**1. Mechanism:**

What changes in the computation graph:
- **Gaussian kernel** (lines 34-38): Currently rebuilt every iteration via `arange → exp → sum → div → mul → expand → contiguous` (~5 small kernels). Cache as a registered buffer — built once, reused forever. Removes ~5 kernel launches/iter.
- **blur(target)** (line 47): Currently recomputed every iteration. Cache the result.
- **blur(target²)** (line 53): Currently recomputed every iteration. Cache the result.
- **mu_target_sq** (line 49): Currently recomputed. Cache.
- **sigma_target_sq** (line 53): Currently recomputed. Cache.

Which redundant work is removed:
- 2 forward conv2d calls on constant data (lines 47, 53) → removes 2 of 5 forward cuDNN kernels
- ~5 kernel-construction kernels (arange, exp, sum, div, mul) → removes ~5 small kernels
- ~3 elementwise ops on cached target terms (mu_target_sq, sigma_target_sq) → removes ~3 elementwise kernels

**Critical caveat**: In actual training (round-robin, 311 cameras), `target` changes every iteration. Caching requires per-camera storage:
- Per camera: blur(target) [1,3,1080,1920] + blur(target²) [1,3,1080,1920] = ~50 MB
- Full 311-camera cache: ~15.5 GB — **memory-prohibitive** on 40GB GPU with 1M Gaussians loaded
- **Practical solution**: LRU cache with limited size (e.g., 32 cameras = 1.6 GB), or cache only for the profiling setup (fixed camera)

In the fixed-camera profiling context: 100% cache hit rate, trivially saves ~7 ms/iter.
In round-robin training: cache hit only when same camera revisited (every 311 iters) — hit rate ~0.3% without LRU. With LRU-32: hit rate ~0% (cameras don't repeat within 32 iterations of a 311-camera cycle).

**Revised assessment**: Candidate A's target-term caching is **only effective in fixed-camera profiling**. In actual training, only the **kernel construction caching** is universally applicable (~0.1 ms savings — negligible).

| Metric | Fixed-camera profiling | Round-robin training |
|--------|----------------------|---------------------|
| Kernel cache savings | ~0.1 ms/iter | ~0.1 ms/iter |
| Target-term cache savings | ~7 ms/iter | ~0 ms/iter (no reuse) |
| **Total A savings** | **~7 ms/iter (18% of D-SSIM)** | **~0.1 ms/iter (negligible)** |

**2. Prior-art classification:** Known optimization (constant subexpression elimination)

**3. Expected impact:**
- Fixed-camera: D-SSIM -18%, total -8%
- Round-robin training: D-SSIM -0.3%, total -0.1% (kernel construction only)
- **Practical impact in actual training: NEGLIGIBLE**

**4. Quality risk:** Mathematically identical (caching produces identical results)

**5. Evidence required before KEEP:**
- Measure per-camera cache hit rate in actual round-robin training
- If hit rate <5%: DROP target-term caching, KEEP only kernel construction (negligible savings)
- If profiling-only: report as profiling optimization, not training optimization

---

### Candidate B: Separable Gaussian Convolution

**1. Mechanism:**

What changes in the computation graph:
- Replace `F.conv2d(x, kernel_2d [3,1,11,11], padding=5, groups=3)` (line 44) with:
  ```
  # Horizontal pass: [3,1,1,11] kernel
  x_h = F.conv2d(x, kernel_h, padding=(0,5), groups=3)
  # Vertical pass: [3,1,11,1] kernel
  x_blur = F.conv2d(x_h, kernel_v, padding=(5,0), groups=3)
  ```
- Each blur() call changes from 1 kernel launch (2D conv) to 2 kernel launches (1D convs)
- Forward: 5 blur calls → 10 conv2d calls (but each is 1D, ~5.5× cheaper per call)
- Backward: 3 dgrad → 6 dgrad (but each is 1D)

Which redundant work is removed:
- 5.5× FLOP reduction on all 6 gradient-requiring convolutions (3 fwd + 3 bwd)
- 121 MACs/pixel → 22 MACs/pixel (11+11)

**Critical uncertainty**: cuDNN's `conv2d_grouped_direct_kernel` is the current algorithm for 3-group depthwise 2D conv. For 1D convolutions ([3,1,1,11] or [3,1,11,1]), cuDNN may:
- Use a different, potentially faster algorithm → expected 3-5× speedup
- Use the same grouped direct algorithm with less work per thread → expected 2-3× speedup
- Use a worse algorithm (e.g., im2col + matmul) → no speedup or regression

This **must be measured empirically** — it is the single highest-impact uncertainty in the entire C42 investigation.

**2. Prior-art classification:** Known optimization (separable filtering is textbook DSP; applying it to SSIM is standard image processing)

**3. Expected impact:**
| Scenario | D-SSIM reduction | Total speedup |
|----------|-----------------|---------------|
| Best case (cuDNN 1D is 5× faster) | 70% (38→11 ms) | 30% (89→62 ms) |
| Likely case (cuDNN 1D is 3× faster) | 50% (38→19 ms) | 21% (89→70 ms) |
| Worst case (cuDNN 1D is 1.5× faster) | 25% (38→28 ms) | 11% (89→79 ms) |
| Failure (cuDNN 1D is same speed) | 0% | 0% |

**4. Quality risk:** Mathematically identical — separable convolution produces the same result as 2D convolution when the kernel is separable (which it is by construction on line 37). Floating-point differences are at the level of summation order (~1e-7 relative error), which is below the threshold that affects SSIM values.

**5. Evidence required before KEEP:**
- Measure 1D conv2d vs 2D conv2d wall time for [1,3,1080,1920] input with [3,1,1,11] vs [3,1,11,11] kernel
- Verify numerical equivalence: `max(|separable_result - 2d_result|) < 1e-5`
- Measure backward (dgrad) speedup: 1D dgrad vs 2D dgrad
- If 1D dgrad is NOT faster: the backward dominates (54.8% of D-SSIM), so candidate may fail
- **Decision gate**: D-SSIM time reduction >30% → KEEP; <15% → DROP

---

### Candidate C: torch.compile / Operator Fusion

**1. Mechanism:**

What changes in the computation graph:
- `torch.compile(combined_loss)` applies Triton-based fusion to the autograd graph
- Fuses ~30 elementwise kernels (mul, add, sub, div, pow) into ~5 fused kernels
- May also optimize cuDNN conv dispatch (unlikely to change algorithm, but may reduce launch overhead)
- Does NOT change the convolution computation itself — conv2d remains cuDNN

Which redundant work is removed:
- Kernel launch overhead: ~30 launches → ~5 launches (saves ~25 × 7μs = 175μs)
- Memory traffic: fused kernels read each input once and write each output once, vs. unfused kernels that materialize intermediates → estimated 3-5× less memory traffic for elementwise portion

**Scope limitation**: torch.compile fuses elementwise ops but does NOT fuse convolutions. The 8 cuDNN kernels (5 fwd + 3 bwd) remain unchanged. Since cuDNN convs are 74.7% of D-SSIM time, the maximum speedup from C alone is on the remaining 25.3% (elementwise).

**2. Prior-art classification:** Known optimization (PyTorch 2.x standard feature)

**3. Expected impact:**
| Component | Current | After compile | Savings |
|-----------|---------|--------------|--------|
| Elementwise (25.3% of D-SSIM) | ~9.6 ms | ~3-5 ms | ~5-7 ms |
| cuDNN conv (74.7%) | ~28.4 ms | ~28.4 ms | 0 ms |
| **D-SSIM total** | ~38 ms | ~32-33 ms | ~5-6 ms (15%) |
| **Total training** | ~89 ms | ~83-84 ms | ~5-6 ms (6%) |

**4. Quality risk:** Mathematically identical — torch.compile preserves numerical semantics by construction.

**5. Evidence required before KEEP:**
- Measure combined_loss compilation time (first-call overhead)
- Measure compiled vs uncompiled D-SSIM time after warmup
- Verify no autograd graph breakage with gsplat custom autograd functions
- **Decision gate**: D-SSIM reduction >10% AND no autograd errors → KEEP; autograd errors → DROP

---

### Candidate D: Downsampled D-SSIM

**1. Mechanism:**

What changes in the computation graph:
- Before SSIM computation, downsample both pred and target:
  ```
  pred_down = F.avg_pool2d(pred, kernel_size=2, stride=2)  # 1080→540, 1920→960
  target_down = F.avg_pool2d(target, kernel_size=2, stride=2)
  dsim = d_ssim_loss(pred_down, target_down)
  ```
- All convolution and elementwise operations run on 4× fewer pixels (1/2 res) or 16× fewer (1/4 res)
- The gradient flows through the downsampling (avg_pool is differentiable)
- L1 loss remains at full resolution (captures pixel-level detail)

Which redundant work is removed:
- Not "redundant" work — this removes *necessary* work at the cost of spatial precision
- SSIM at lower resolution captures structural similarity at coarser scale
- The 11×11 Gaussian window at 1/2 resolution covers an equivalent 22×22 area at full resolution

**2. Prior-art classification:** Engineering application (downsampled SSIM is used in video quality assessment and real-time rendering; not standard in 3DGS training)

**3. Expected impact:**
| Resolution | Pixels | D-SSIM time | D-SSIM reduction | Total speedup |
|-----------|--------|-------------|-----------------|---------------|
| Full (1080p) | 2.07M | 38.0 ms | baseline | baseline |
| 1/2 (540p) | 518K | ~9.5 ms | 75% | ~32% (89→61 ms) |
| 1/4 (270p) | 130K | ~2.4 ms | 94% | ~40% (89→53 ms) |

**4. Quality risk:** Training behavior change — SSIM at lower resolution provides structural similarity gradients at coarser spatial scales. Effects:
- High-frequency detail (textures, edges) gets less SSIM gradient → may rely more on L1 for fine detail
- Large-scale structure still captured → structural coherence preserved
- Expected PSNR impact: 0.1-0.5 dB (1/2 res), 0.3-1.0 dB (1/4 res)
- Expected SSIM impact: 0.002-0.01 (1/2 res), 0.005-0.02 (1/4 res)
- **The quality impact is scene-dependent and must be measured**

**5. Evidence required before KEEP:**
- Train 500 iterations with 1/2-res D-SSIM from same checkpoint
- Measure PSNR, SSIM vs baseline at iterations 100, 250, 500
- Compare final Gaussian count (should be identical — no densification in test)
- **Decision gate**: PSNR drop <0.1 dB AND SSIM drop <0.005 AND speedup >20% → KEEP
- If PSNR drop 0.1-0.3 dB: MAYBE — investigate scene dependence
- If PSNR drop >0.3 dB: DROP

---

### Candidate E: Adaptive D-SSIM Scheduling

**1. Mechanism:**

What changes in the computation graph:
Two sub-strategies:

**E1: Frequency reduction** — compute D-SSIM every N iterations, use L1-only otherwise:
```
if iteration % N == 0:
    loss = (1-λ) * L1 + λ * D-SSIM  # full loss
else:
    loss = L1  # L1 only, no D-SSIM computation
```
- Eliminates D-SSIM entirely on (N-1)/N of iterations
- D-SSIM gradient is applied every N-th step → structural guidance is intermittent

**E2: Phase-out scheduling** — reduce λ_dssim over training phases:
```
if iteration < 5000:    λ_dssim = 0.2  # full D-SSIM (structural initialization)
elif iteration < 15000: λ_dssim = 0.1  # half D-SSIM (refinement)
else:                   λ_dssim = 0.0  # L1 only (fine detail)
```
- D-SSIM is always computed but with diminishing weight
- Total D-SSIM time unchanged per iteration, but effective contribution decreases

**E1 is the speedup mechanism** (computes D-SSIM less often). E2 does not save time directly but may justify E1 (if D-SSIM is less important later, computing it less often is safe).

Which redundant work is removed:
- E1: D-SSIM is not computed at all on (N-1)/N iterations → saves 38 ms × (N-1)/N per iteration on average
- E2: No time savings directly, but provides the theoretical basis for E1

**2. Prior-art classification:** Potentially novel training strategy — D-SSIM scheduling specific to 3DGS training dynamics. The original 3DGS paper uses fixed λ=0.2 for all iterations. No published work (to our knowledge, given search was unavailable) explores adaptive D-SSIM scheduling in 3DGS training.

**3. Expected impact:**
| Schedule | D-SSIM iters | Avg D-SSIM time | Total speedup |
|----------|-------------|-----------------|---------------|
| Every iter (baseline) | 100% | 38.0 ms | baseline |
| Every 2nd iter (N=2) | 50% | 19.0 ms | 21% (89→70 ms) |
| Every 3rd iter (N=3) | 33% | 12.7 ms | 28% (89→64 ms) |
| Phase-out (E2+E1) | 33% (first 5K) → 0% (after 15K) | ~5 ms avg | 37% (89→56 ms) |

**4. Quality risk:** Training behavior change — D-SSIM provides structural similarity gradients that guide Gaussian placement toward perceptually important regions. Reducing D-SSIM frequency:
- May slow structural convergence in early training
- May miss structural corrections that L1 alone doesn't capture
- Risk is phase-dependent: early training (Gaussians finding structure) is more sensitive than late training (refinement)
- E2 (phase-out) is theoretically safer than E1 (frequency reduction) because it preserves early-training structural guidance

**5. Evidence required before KEEP:**
- Train 500 iterations with N=2 (every-other) from same checkpoint at iter 5000
- Measure PSNR, SSIM trajectory vs baseline
- Test E2 phase-out: λ=0.2 for first 250, λ=0.1 for next 250
- Measure whether D-SSIM gradient magnitude decreases over training (if yes, phase-out is justified)
- **Decision gate**: PSNR drop <0.1 dB AND SSIM drop <0.005 AND speedup >20% → KEEP
- If quality drop is acceptable only with phase-out (not with uniform frequency reduction): KEEP E2+E1 combined
- **Research evidence**: D-SSIM gradient magnitude vs iteration curve — if it decays, this is strong evidence for scheduling

---

### Candidate F: Hybrid Strategy

**1. Mechanism:**

What changes in the computation graph:
- **Layer 1 (Engineering floor)**: Apply B (separable) + C (torch.compile) — mathematically identical, zero quality risk
- **Layer 2 (Training strategy)**: Apply E (adaptive scheduling) on top of the optimized D-SSIM
- The optimized D-SSIM from Layer 1 is what gets scheduled in Layer 2 — so when D-SSIM IS computed, it's faster

Combined computation graph:
```
# Layer 1: Optimized D-SSIM (mathematically identical to original)
def fast_d_ssim(pred, target, cached_kernel_1d):
    def blur_separable(x):
        x_h = F.conv2d(x, kernel_1d_h, padding=(0,5), groups=3)
        return F.conv2d(x_h, kernel_1d_v, padding=(5,0), groups=3)
    # ... same SSIM formula, but with separable blur ...
    
fast_loss = torch.compile(fast_combined_loss)

# Layer 2: Scheduling
if iteration % N == 0 or iteration < phase_out_iter:
    loss = fast_loss(rendered, gt, lambda_dssim=scheduled_lambda)
else:
    loss = l1_loss(rendered, gt)  # L1 only
```

Which redundant work is removed:
- Layer 1: 5.5× FLOP waste (B) + elementwise fragmentation (C) + kernel reconstruction (A)
- Layer 2: D-SSIM computation entirely on (N-1)/N iterations

**2. Prior-art classification:** Hybrid: known optimization (B, C) + potentially novel training strategy (E)

**3. Expected impact:**
| Configuration | D-SSIM when computed | D-SSIM frequency | Avg D-SSIM time | Total speedup |
|--------------|---------------------|-----------------|-----------------|---------------|
| Baseline | 38.0 ms | 100% | 38.0 ms | 0% |
| B+C only | ~13-15 ms | 100% | ~14 ms | 27% |
| B+C+E(N=2) | ~14 ms | 50% | ~7 ms | 35% |
| B+C+E(phase-out) | ~14 ms | 33%→0% | ~5 ms | 39% |

**4. Quality risk:**
- Layer 1 (B+C): mathematically identical, zero risk
- Layer 2 (E): training behavior change, moderate risk
- Combined risk = risk of E alone (B+C don't add risk)

**5. Evidence required before KEEP:**
- First validate B+C alone (zero quality risk) → measure speedup
- Then add E on top → measure quality impact
- **Two-phase gate**: Phase 1 (B+C) must pass speedup >20% with zero quality loss; Phase 2 (B+C+E) must pass quality gate

---

## Recommended C42 Experimental Order

### Ordering Rationale

The order is determined by three factors, weighted as requested:
1. **Scientific value** (primary) — does this advance understanding or just apply known tricks?
2. **Risk** (secondary) — what is the probability of quality regression?
3. **Implementation cost** (tertiary) — how much effort to test?

### Experimental Order

| Order | Candidate | Scientific value | Risk | Impl. cost | Rationale |
|-------|-----------|-----------------|------|------------|-----------|
| **1** | **B: Separable conv** | MEDIUM (proves root cause: is cuDNN the problem or is it the algorithm?) | ZERO (identical math) | LOW (modify blur function) | **Must be first**: If B fails (cuDNN 1D is not faster), the entire optimization thesis changes. If B succeeds, it establishes the engineering floor. This is the pivotal experiment that determines whether C42 is viable at all. |
| **2** | **C: torch.compile** | LOW (standard practice) | LOW (may have autograd issues) | VERY LOW (one-line change) | **Second**: Quick to test, additive to B, identifies autograd compatibility issues early. If torch.compile breaks gsplat autograd, we need to know before building more complex variants. |
| **3** | **E: Adaptive scheduling** | HIGH (novel 3DGS training strategy) | MEDIUM (training behavior change) | LOW (modify training loop) | **Third**: After B+C establish the engineering floor, E explores the research question: "Is D-SSIM equally important throughout training?" This is the candidate with the highest scientific value. The D-SSIM gradient magnitude decay curve is a key research deliverable. |
| **4** | **D: Downsampled D-SSIM** | MEDIUM (quality/speed tradeoff characterization) | MEDIUM (measurable quality drop) | LOW (add downsampling) | **Fourth**: After E, D provides an alternative training strategy if E's quality impact is too high. D and E are complementary — D changes spatial precision, E changes temporal frequency. |
| **5** | **A: Cache constants** | LOW (basic engineering) | ZERO | LOW | **Fifth (or parallel)**: In actual round-robin training, A's impact is negligible (target changes every iter). Only the kernel construction cache (~0.1 ms) is universally applicable. Test to confirm this prediction. |
| **6** | **F: Hybrid B+C+E** | MEDIUM-HIGH (synthesis) | MEDIUM (from E) | MEDIUM (integrate all) | **Last**: After individual candidates are validated, F combines the proven winners. This is the final deliverable, not a separate experiment. |

### Why This Order (Not Speed-First)

A speed-first ordering would put D (downsampled, 40% speedup) first. But this order prioritizes:

1. **B first** because it is the **pivotal experiment** — if separable conv doesn't help (cuDNN 1D is slow), then the entire "D-SSIM is algorithmically excessive work" thesis is wrong, and we need to reconsider whether the bottleneck is the algorithm or cuDNN's implementation. B tests the root cause directly.

2. **E third** (not first despite highest scientific value) because E's quality impact must be measured on a **correctly optimized baseline**. If we test E on the original slow D-SSIM and it passes the quality gate, we won't know if B+C would have changed the dynamics. Testing E after B+C isolates the training-strategy effect from the engineering effect.

3. **D after E** because D and E are both training behavior changes. Testing E first (higher scientific value) gives us the gradient-magnitude decay evidence that may also justify D (if D-SSIM matters less later, downsampling later is safer).

4. **A last** because the audit already predicts negligible impact in round-robin training. It's included for completeness, not for impact.

---

## Final Assessment: What Should C42 Be?

### Option 1: Pure Engineering Optimization (B + C only)

**If B succeeds** (separable conv is >2× faster) **and C succeeds** (torch.compile works):
- Combined speedup: ~25-30% with **zero quality risk**
- This alone passes the KEEP decision gate (>20% speedup, <0.1 dB PSNR drop)
- But research value is LOW — "we applied separable filtering and torch.compile" is engineering, not research

**Verdict**: Sufficient for practical impact, insufficient for research contribution.

### Option 2: Optimization + Training Algorithm Contribution (B + C + E)

**If B+C succeed AND E shows acceptable quality**:
- Combined speedup: ~35-40%
- Research contribution: D-SSIM scheduling in 3DGS training — when and how much D-SSIM is needed
- Key deliverable: D-SSIM gradient magnitude decay curve (evidence for scheduling)
- Key deliverable: Quality/speed Pareto frontier for D-SSIM scheduling strategies

**Verdict**: Maximizes both practical impact and research value. **RECOMMENDED.**

### Option 3: Dropped

**If B fails** (cuDNN 1D conv is not faster than 2D):
- The root cause is cuDNN's algorithm selection, not the algorithm itself
- C alone gives only ~6% speedup (insufficient)
- D and E become the only viable paths, but they change training behavior
- Without a safe engineering baseline, the quality risk of D+E is uncompensated

**Verdict**: Only if B definitively fails. Even then, E (scheduling) has independent research value.

---

## Recommendation

### C42 should be: **Optimization + Training Algorithm Contribution (Option 2)**

**Experimental plan:**
1. **Phase 1 (Engineering)**: Test B (separable) → test C (torch.compile) → measure B+C combined
2. **Phase 2 (Research)**: Measure D-SSIM gradient magnitude decay curve → test E (scheduling) → measure quality/speed Pareto
3. **Phase 3 (Synthesis)**: Combine B+C+E as F → final benchmark

**Decision gate after Phase 1:**
- B+C speedup >20% with zero quality loss → proceed to Phase 2 (research)
- B+C speedup <15% → proceed to Phase 2 anyway (E has independent value) but report engineering as DROP
- B fails (cuDNN 1D not faster) → Phase 1 becomes C-only, Phase 2 becomes primary focus

**Key research deliverable from Phase 2:**
- "D-SSIM Gradient Magnitude Decay in 3DGS Training" — a curve showing how ‖∇D-SSIM‖ changes over iterations. If it decays, this is direct evidence that D-SSIM scheduling is valid. This measurement has never been published for 3DGS (to our knowledge).

---

## Artifacts

- D-SSIM source: `scripts/epic05/phase7/loss.py`
- Implementation audit: `reports/phase-c31/c42_dssim_audit.md`
- C41 GPU utilization data: `results/phase-c31/c41_gpu_utilization.json`
- C41 trace: `results/phase-c31/c41_trace.json`
- C42 isolated D-SSIM trace: `results/phase-c31/c42_dssim_full_trace.json`
- Training loop: `scripts/epic05/phase7/train_3dgs.py` (line 272: round-robin camera selection)
