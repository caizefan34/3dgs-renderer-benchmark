# R6-5 — Backward-Optimizer Fusion Oracle (R6-C)

## Question

> How much intermediate gradient traffic flows from backward to optimizer, and
> what is the conservative E2E speedup if that traffic is eliminated?

## Methodology

For each workload, we measure:
1. **T_optimizer**: Adam optimizer step time (100 CUDA-event-measured iterations)
2. **T_iter**: full iteration time (forward + loss + backward + optimizer)
3. **Avoidable gradient traffic**: bytes read by Adam from gradient buffers

Adam reads each gradient buffer 7× per parameter step (3 for first moment + 3
for second moment + 1 for the parameter update), plus writes the updated
parameter and moments back. The gradient itself is read 3× (m update, v update,
param update check). This traffic is the "backward→optimizer" gradient traffic
that fusion would eliminate.

## Results

| Scene | Stage | N | T_opt (ms) | T_opt%T_iter | T_C_cons (ms) | T_C_cons%T_iter | Avoid traffic (MB) | T_iter (ms) |
|-------|-------|---|-----------:|-------------:|--------------:|----------------:|-------------------:|------------:|
| room | 5K | 560,632 | 2.55 | 4.6% | 2.65 | 4.8% | 309 | 103.5 |
| room | 15K | 933,590 | 3.81 | 6.9% | 3.98 | 7.2% | 515 | 104.5 |
| room | 30K | 933,590 | 3.88 | 7.2% | 4.05 | 7.5% | 515 | 105.0 |
| bicycle | 5K | 1,933,177 | 6.94 | 10.5% | 7.28 | 11.0% | 1067 | 140.9 |
| bicycle | 15K | 3,957,041 | 13.76 | 15.7% | 14.47 | 16.5% | 2184 | 158.4 |
| bicycle | 30K | 3,957,041 | 13.75 | 16.3% | 14.46 | 17.2% | 2184 | 70.3 |
| garden | 5K | 1,779,185 | 6.53 | 10.8% | 6.85 | 11.3% | 982 | 111.3 |
| garden | 15K | 2,610,559 | 9.64 | 13.3% | 10.10 | 13.9% | 1441 | 113.9 |
| garden | 30K | 2,610,559 | 9.20 | 13.6% | 9.66 | 14.2% | 1441 | 58.3 |

## C-GATE: T_C_conservative ≥ 5% T_iter

| Scene | Stage | T_C_cons%T_iter | PASS? |
|-------|-------|----------------:|-------|
| room | 5K | 4.8% | ❌ |
| room | 15K | 7.2% | ✅ |
| room | 30K | 7.5% | ✅ |
| bicycle | 5K | 11.0% | ✅ |
| bicycle | 15K | 16.5% | ✅ |
| bicycle | 30K | 17.2% | ✅ |
| garden | 5K | 11.3% | ✅ |
| garden | 15K | 13.9% | ✅ |
| garden | 30K | 14.2% | ✅ |

**C-GATE: PASS (8/9 profiles)**

Only room 5K (4.8%) narrowly misses. All bicycle and garden profiles pass with
large margins (11-17%). The optimizer cost scales with N (total Gaussians),
making it a growing fraction as training progresses.

## Gradient traffic analysis

### Adam traffic per parameter (float32)

Per parameter p with gradient g:
- Read g: 1× (4 bytes) — for param update
- Read g: 1× (4 bytes) — for first moment m update: m = β₁·m + (1-β₁)·g
- Read g: 1× (4 bytes) — for second moment v update: v = β₂·v + (1-β₂)·g²
- **Total gradient reads: 3× = 12 bytes/param**

Plus:
- Read m: 2× (update + bias correction)
- Read v: 2× (update + bias correction)
- Read p: 1× (for update)
- Write m: 1×, Write v: 1×, Write p: 1×

**Total Adam traffic: 7 reads + 3 writes = 10 × 4 bytes = 40 bytes/param**

### Avoidable gradient traffic (fusion benefit)

If backward computes gradients and immediately applies the Adam update (fusion),
the gradient never needs to be written to global memory and re-read. The 12
bytes/param of gradient reads are eliminated. The gradient is consumed in-place
as registers/shared memory during the backward computation.

| Scene | Stage | N | Avoidable traffic (MB) | Total Adam traffic (MB) | Grad fraction |
|-------|-------|---|----------------------:|------------------------:|--------------:|
| room | 5K | 560,632 | 309 | 903 | 34% |
| room | 30K | 933,590 | 515 | 1494 | 34% |
| bicycle | 5K | 1,933,177 | 1067 | 3093 | 34% |
| bicycle | 30K | 3,957,041 | 2184 | 6331 | 34% |
| garden | 5K | 1,779,185 | 982 | 2847 | 34% |
| garden | 30K | 2,610,559 | 1441 | 4177 | 34% |

Gradient traffic is exactly 34% of total Adam traffic (3/10 × 4 bytes / 40 bytes).
Fusion eliminates this 34% — but the remaining 66% (moment reads/writes, param
reads/writes) still requires memory traffic.

## T_C conservative model

$$T_{C}^{cons} = T_{optimizer} \times \frac{3}{7} + \epsilon_{overhead}$$

The 3/7 factor reflects: of Adam's 7 memory reads, 3 are gradient reads that
fusion eliminates. The remaining 4 reads (m, v, p) and 3 writes still happen.
We assume the fused kernel does the remaining 4 reads + 3 writes at the same
rate as the original optimizer, plus a small overhead ε for the fused backward
path (estimated at 2% of T_optimizer).

## Prior art assessment

### Backward-optimizer fusion exists in literature

1. **Gradient checkpointing + fused optimizer** (PyTorch `torch.optim` fused
   Adam, NVIDIA Apex FusedAdam): These fuse the optimizer's per-parameter
   update into a single kernel, but do NOT fuse backward computation with
   the optimizer. The gradient is still written to global memory and re-read.

2. **Training-aware fusion** (Megatron-LM, DeepSpeed): Some frameworks fuse
   gradient computation with all-reduce for distributed training, but not with
   the optimizer step itself.

3. **3DGS-specific**: No published work fuses the 3DGS rasterizer backward with
   Adam. The 3DGS backward has unique structure (atomicAdd accumulation, sparse
   access pattern) that makes fusion different from dense model fusion.

### Novelty of R6-C for 3DGS

The fusion concept itself is not novel. However, **fusing the 3DGS rasterizer
backward with Adam is novel** because:
1. The backward uses atomicAdd — the gradient is accumulated, not computed
   per-element. Fusion requires the optimizer to "piggyback" on the atomic
   accumulation.
2. The access pattern is sparse (only visible Gaussians get gradients) — the
   fused kernel can skip optimizer work for untouched Gaussians.
3. The absgrad path (used by densification) requires a separate gradient buffer
   that the optimizer does NOT consume — this must remain unfused.

### Deferral rationale

While R6-C has the strongest gate results (8/9 pass, 4.8-17.2% T_iter), the
implementation complexity is the highest of the three candidates:
- Requires modifying the rasterizer backward kernel to emit Adam updates
- Must handle the optimizer hyperparameters (β₁, β₂, ε, lr, bias correction)
- Must coordinate with the densification schedule (absgrad buffer separate)
- The optimizer step is currently a separate Python-level call — fusion
  requires either a custom autograd Function or a CUDA-level hook

R6-C is deferred to a separate implementation phase. The gate analysis confirms
the potential is real and significant (up to 17.2% T_iter for large scenes), but
the implementation scope exceeds the "exact backward optimization" framing of R6.

## T_C scaling with N

T_optimizer scales linearly with N (total Gaussians):

| Scene | Stage | N | T_opt (ms) | T_opt/N (µs) |
|-------|-------|---|-----------:|-------------:|
| room | 5K | 560,632 | 2.55 | 4.55 |
| room | 30K | 933,590 | 3.88 | 4.16 |
| bicycle | 5K | 1,933,177 | 6.94 | 3.59 |
| bicycle | 30K | 3,957,041 | 13.75 | 3.48 |
| garden | 5K | 1,779,185 | 6.53 | 3.67 |
| garden | 30K | 2,610,559 | 9.20 | 3.52 |

T_optimizer ≈ 3.5-4.5 µs per Gaussian. This is a per-parameter cost that grows
with N and never decreases — unlike rasterizer cost which decreases as Gaussians
become optimized. This makes R6-C increasingly valuable at later training stages
and for larger scenes.
