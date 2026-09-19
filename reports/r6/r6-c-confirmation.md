# R6-C — Confirmation: DEFER (Physically Removable Gradient Traffic Only)

## Status: DEFER (confirmed)

R6-C remains DEFER. The repaired oracle (r6-c-oracle-repair.md) already
established that the conservative E2E is 0.48–2.21% — below the 5% threshold
for all 9 workloads. This confirmation verifies that the repair correctly counts
only physically removable gradient intermediate traffic.

## What R6-C fusion eliminates (physically removable)

The backward-optimizer fusion would eliminate gradient write→read traffic:

```
Without fusion:
  Backward:  gradient → global memory write (param .grad buffers)
  Optimizer: gradient ← global memory read  (param .grad buffers)

With fusion:
  Backward computes gradient → registers (no global write)
  Fused Adam consumes immediately (no global read)
```

### Removable traffic = 2 × grad_bytes

| Component | Bytes | Removable? |
|-----------|------:|:----------:|
| Gradient write (backward → global) | grad_bytes | YES |
| Gradient read (optimizer ← global) | grad_bytes | YES |
| Parameter read | param_bytes | NO (still needed) |
| Parameter write | param_bytes | NO (still needed) |
| exp_avg read/write | 2 × param_bytes | NO (still needed) |
| exp_avg_sq read/write | 2 × param_bytes | NO (still needed) |
| Adam computation (β₁, β₂, bias correction, sqrt) | — | NO (still runs) |

**Total Adam traffic = 7 × param_bytes. Removable = 2 × param_bytes = 28.6% of Adam traffic.**

### Intermediate VJP buffers (optimistic bound)

If the ENTIRE backward chain is fused (not just the backward-optimizer
boundary), intermediate VJP buffers could also be eliminated:

- SH backward writes v_coefficients [N, K², 3] → read by autograd
- Projection backward writes v_means, v_quats, v_scales [N, 10] → read by autograd
- Rasterizer backward writes v_means2d, v_conics, v_colors, v_opacities [N, 11] → read by autograd

These intermediate buffers are written and then read in the backward chain. If
fused into registers, they never touch global memory. The optimistic bound
includes this traffic.

## Three bounds (confirmed from r6-c-oracle-repair.md)

| Bound | What it eliminates | Bandwidth | E2E range |
|-------|-------------------|-----------|-----------|
| Lower | Gradient read only (1× grad_bytes) | 1.0 TB/s | 0.48–1.36% |
| **Conservative** | Gradient write + read (2× grad_bytes) | 1.0 TB/s | **0.48–2.21%** |
| Optimistic | All gradient + intermediate VJP traffic | 1.55 TB/s | 0.68–3.09% |

## Gate evaluation

| Condition | Result |
|-----------|--------|
| C-GATE: T_saved_conservative ≥ 5% T_iter | **FAIL 0/9** (0.48–2.21%) |
| Even optimistic bound ≥ 5%? | **FAIL 0/9** (0.68–3.09%) |

## Why DEFER, not DROP

1. The mechanism is sound (fusing backward-optimizer is a valid optimization)
2. The gradient traffic is physically removable (not approximate, not skipped)
3. Future hardware with different bandwidth characteristics could change the calculus
4. A fundamentally different fusion approach (e.g., full VJP chain fusion) could
   eliminate more traffic — but that is a separate, much more complex optimization

## What changed from the recalibration

Nothing. The R6-C repair was already correct and uses only physically
removable traffic. The T_zero recalibration does not affect R6-C because
R6-C targets gradient write/read traffic at the backward-optimizer boundary,
not gradient buffer zero-init.

## Corrected evidence level

- T_optimizer: Level 1 (direct CUDA event) — AUTHORITATIVE
- Gradient traffic bytes: Level 5 (metadata-derived from tensor shapes) — but
  physically grounded (tensor sizes are exact, not estimated)
- T_saved model: Level 4 (bandwidth-limited model) — simple and physically valid
- Conservative E2E: Level 4 — AUTHORITATIVE (repaired)
