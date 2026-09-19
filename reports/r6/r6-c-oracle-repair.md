# R6-C — Oracle Repair

## The bug

The original `r6_5_fusion_oracle.py` computed:

```python
T_C_conservative = T_opt_mean + 0.5 * T_avoidable_traffic_ms
```

This **adds** the optimizer time to the avoidable traffic time, producing a
number LARGER than T_optimizer. The report then interpreted this as
"T_C conservative savings = 17.2% T_iter" — but a "savings" metric that is
larger than the entire optimizer time is physically impossible.

You cannot save more time than the operation takes.

## Correct traffic model

### Without fusion

```
Backward writes:  gradient → global memory  (param .grad buffers)
Optimizer reads:  param, gradient, exp_avg, exp_avg_sq  (4 reads)
Optimizer writes: param, exp_avg, exp_avg_sq  (3 writes)
```

### With fusion

The gradient is computed in the backward and immediately consumed by the
fused Adam update. The gradient never needs to be written to global memory
and re-read.

```
Backward computes:  gradient → registers (no global write)
Fused Adam reads:   param, exp_avg, exp_avg_sq  (3 reads, NO gradient read)
Fused Adam writes:  param, exp_avg, exp_avg_sq  (3 writes)
```

### What fusion eliminates

- **Gradient write** (backward → global memory): 1× grad_bytes
- **Gradient read** (optimizer ← global memory): 1× grad_bytes
- Total avoidable: 2 × grad_bytes

### What fusion does NOT eliminate

- Parameter read/write (still needed for Adam update)
- First moment (exp_avg) read/write (still needed)
- Second moment (exp_avg_sq) read/write (still needed)
- Adam computation (β₁, β₂, bias correction, sqrt) — still runs

### Adam traffic breakdown

| Operation | Reads | Writes | Total |
|-----------|------:|-------:|------:|
| Gradient | 1 | 0 (write is in backward) | 1 × grad_bytes |
| First moment (exp_avg) | 1 | 1 | 2 × param_bytes |
| Second moment (exp_avg_sq) | 1 | 1 | 2 × param_bytes |
| Parameter | 1 | 1 | 2 × param_bytes |
| **Total Adam** | **4** | **3** | **7 × param_bytes** |
| **Avoidable (gradient)** | **1** | **0** | **1 × param_bytes** |

Gradient traffic = 1/7 = 14.3% of total Adam traffic.

## Three bounds

### Lower bound

Only eliminate the gradient READ by optimizer (the write still happens for
autograd bookkeeping):

```
avoidable_bytes_lower = grad_bytes = total_param_bytes
T_saved_lower = grad_bytes / bandwidth_achieved
bandwidth_achieved = 1.0 TB/s (A100, ~65% of peak)
```

### Conservative bound

Eliminate both gradient write (backward) and read (optimizer), at achievable
bandwidth:

```
avoidable_bytes_cons = 2 × grad_bytes
T_saved_cons = 2 × grad_bytes / bandwidth_achieved
```

### Optimistic bound

Eliminate ALL gradient traffic (param .grad + intermediate VJP buffers) at
peak bandwidth. This requires fusing the ENTIRE backward chain, not just the
backward-optimizer boundary:

```
avoidable_bytes_opt = 2 × (grad_bytes + intermediate_grad_bytes)
T_saved_opt = avoidable_bytes_opt / bandwidth_peak
bandwidth_peak = 1.55 TB/s
```

### Sanity check

T_saved must not exceed the physically removable work:
- Gradient is 1/7 of Adam traffic → T_saved ≤ T_opt/7 (traffic-bound)
- T_saved_lower is capped at T_opt × (1/7)

## Results (all 9 workloads)

| Scene | Stage | Grad bytes (MB) | T_opt (ms) | T_saved lower (ms) | T_saved cons (ms) | T_saved opt (ms) | Cons % T_iter | Opt % T_iter | Original (broken) % |
|-------|-------|----------------:|-----------:|-------------------:|-------------------:|-------------------:|--------------:|-------------:|--------------------:|
| room | 5K | 132 | 2.54 | 0.132 | 0.265 | 0.370 | 0.48% | 0.68% | 4.8% |
| room | 15K | 220 | 3.81 | 0.220 | 0.440 | 0.616 | 0.79% | 1.10% | 7.2% |
| room | 30K | 220 | 3.88 | 0.220 | 0.440 | 0.616 | 0.82% | 1.14% | 7.5% |
| bicycle | 5K | 456 | 6.94 | 0.456 | 0.912 | 1.276 | 1.36% | 1.91% | 11.0% |
| bicycle | 15K | 934 | 13.76 | 0.934 | 1.868 | 2.612 | 2.12% | 2.97% | 16.5% |
| bicycle | 30K | 934 | 13.75 | 0.934 | 1.868 | 2.612 | 2.21% | 3.09% | 17.2% |
| garden | 5K | 420 | 6.53 | 0.420 | 0.840 | 1.176 | 1.38% | 1.94% | 11.3% |
| garden | 15K | 616 | 9.64 | 0.616 | 1.232 | 1.724 | 1.76% | 2.47% | 13.9% |
| garden | 30K | 616 | 9.20 | 0.616 | 1.232 | 1.724 | 1.81% | 2.53% | 14.2% |

## Gate verdict

### C-GATE: T_saved_conservative ≥ 5% T_iter

| Scene | Stage | Cons % T_iter | PASS? |
|-------|-------|--------------:|-------|
| room | 5K | 0.48% | ❌ |
| room | 15K | 0.79% | ❌ |
| room | 30K | 0.82% | ❌ |
| bicycle | 5K | 1.36% | ❌ |
| bicycle | 15K | 2.12% | ❌ |
| bicycle | 30K | 2.21% | ❌ |
| garden | 5K | 1.38% | ❌ |
| garden | 15K | 1.76% | ❌ |
| garden | 30K | 1.81% | ❌ |

**C-GATE: FAIL (0/9)**

Even the OPTIMISTIC bound (which assumes full backward chain fusion at peak
bandwidth) reaches only 0.68-3.09% — still below 5% for all workloads.

## R6-C status: DEFER

The repaired oracle shows the fusion opportunity is 10× smaller than the
original estimate. The conservative E2E (0.48-2.21%) is below 5% for all
workloads. Even the optimistic bound (0.68-3.09%) does not reach 5%.

R6-C is firmly DEFERRED. The fusion concept is valid but the achievable
savings are too small to justify the implementation complexity.

## Error summary

| Metric | Original (broken) | Repaired | Error factor |
|--------|-------------------:|---------:|-------------:|
| Best E2E | 17.2% | 2.21% | 7.8× overestimate |
| Median E2E | 13.9% | 1.76% | 7.9× overestimate |
| C-GATE pass | 8/9 | 0/9 | All false positives |

The original oracle added T_optimizer to traffic time. The correct model
computes T_saved as bandwidth-limited traffic time, which is a small fraction
of T_optimizer because:
1. Gradient is only 1/7 of Adam traffic (14.3%)
2. The bandwidth-limited time for gradient traffic is small compared to T_iter
3. T_optimizer includes computation (β₁, β₂, bias correction, sqrt) that
   fusion cannot eliminate
