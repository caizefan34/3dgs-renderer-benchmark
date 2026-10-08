# P11 Optimizer Dependency Audit (Gate O)

Measurement only. No cross-iteration double-buffer overlap implemented.

## Exact dependency of the next iteration

The next iteration's projection/render reads ALL of:

- `means` = `model.get_xyz` → fully-fused projection (forward)
- `quats` = `model.get_rotation` (normalized) → projection
- `scales` = `model.get_scaling` (exp) → projection (covariance)
- `opacities` = `model.get_opacity` (sigmoid) → rasterization
- SH = `model.get_features` (slice to active degree) → SH evaluation

In the frozen trainer these are read at the START of the next iteration, strictly AFTER `optimizer.step()` was enqueued on the same CUDA stream. Sequential stream semantics therefore guarantee every read observes exactly the standard updated parameter values. No event/sync trickery is required — and none is present.

## Why cross-iteration overlap is not exact

Standard Adam updates a parameter whenever its `.grad` exists — even if every gradient value is zero:

```
m ← β1·m + (1−β1)·g        (g = 0  ⇒  m ← β1·m, decays but nonzero)
v ← β2·v + (1−β2)·g²       (v ← β2·v)
update = −lr·m̂/(√v̂ + ε)    (≠ 0 while m ≠ 0)
```

The trainer's backward always materializes `.grad` for all five parameter groups (raster backward writes dense per-gaussian buffers), so momentum always applies. Any schedule that lets the next forward begin reading parameters before the update kernel completes exposes stale parameter values — a semantic change, not an optimization. Waiting on an event after the update removes the overlap entirely.

## Empirical momentum check (30K states, campaign optimizer state restored)

### room

| group | n params | lr | step ms | % of joint | moved with zero-grad | max abs delta |
|---|---:|---:|---:|---:|---|---:|
| xyz | 2,656,524 | 1.59e-05 | 0.171 | 5.2% | True | 2.253e-05 |
| shs | 42,504,384 | 2.50e-03 | 1.972 | 60.4% | True | 3.534e-03 |
| opacity | 885,508 | 2.50e-02 | 0.164 | 5.0% | True | 3.531e-02 |
| scaling | 2,656,524 | 5.00e-03 | 0.169 | 5.2% | True | 7.068e-03 |
| rotation | 3,542,032 | 1.00e-03 | 0.169 | 5.2% | True | 1.414e-03 |

Joint `optimizer.step()`: 3.266 ms; sum of single-group steps: 2.646 ms (closure gap 19.0%; foreach batching explains the difference).

### bicycle

| group | n params | lr | step ms | % of joint | moved with zero-grad | max abs delta |
|---|---:|---:|---:|---:|---|---:|
| xyz | 11,150,184 | 1.36e-05 | 0.556 | 4.9% | True | 1.931e-05 |
| shs | 178,402,944 | 2.50e-03 | 8.112 | 71.6% | True | 3.534e-03 |
| opacity | 3,716,728 | 2.50e-02 | 0.163 | 1.4% | True | 3.533e-02 |
| scaling | 11,150,184 | 5.00e-03 | 0.557 | 4.9% | True | 7.067e-03 |
| rotation | 14,866,912 | 1.00e-03 | 0.724 | 6.4% | True | 1.413e-03 |

Joint `optimizer.step()`: 11.329 ms; sum of single-group steps: 10.112 ms (closure gap 10.7%; foreach batching explains the difference).

### garden

| group | n params | lr | step ms | % of joint | moved with zero-grad | max abs delta |
|---|---:|---:|---:|---:|---|---:|
| xyz | 7,591,563 | 1.34e-05 | 0.391 | 4.9% | True | 1.907e-05 |
| shs | 121,465,008 | 2.50e-03 | 5.545 | 69.0% | True | 3.534e-03 |
| opacity | 2,530,521 | 2.50e-02 | 0.163 | 2.0% | True | 3.535e-02 |
| scaling | 7,591,563 | 5.00e-03 | 0.390 | 4.9% | True | 7.069e-03 |
| rotation | 10,122,084 | 1.00e-03 | 0.511 | 6.4% | True | 1.414e-03 |

Joint `optimizer.step()`: 8.038 ms; sum of single-group steps: 6.999 ms (closure gap 12.9%; foreach batching explains the difference).

## Same-iteration overlap note (not reopened)

The only stream-safe overlap is WITHIN an iteration: the post-backward densification-stat chain touches `xyz_gradient_accum`, `denom`, `max_radii2D` and `means2d.absgrad` — none of which the optimizer reads or writes — so overlapping the stat chain with `optimizer.step()` on a second stream is exact in principle. This is generic multi-stream pipelining (established systems technique, crowded prior art), bounded by the ~2%% densify phase, further reduced by L2/DRAM interference between the two streams, and is NOT a paper contribution; P11 does not reopen optimizer overlap.

## Decision

**exact overlap feasible = NO** (momentum moves parameters even with zero gradients; next-forward reads all parameters) → **NO_GO**.
