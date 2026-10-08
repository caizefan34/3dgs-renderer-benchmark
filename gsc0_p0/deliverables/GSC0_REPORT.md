# GSC0 P0 - Gaussian State Compression: Redundancy Oracle Report

- **Generated (UTC):** 2026-10-01T10:31:05Z
- **Verdict:** `DROP_NO_STABLE_STATE`
- **Scenes:** room, garden, bicycle
- **Phases (preregistered):** early=1000, densify_complete=15000, mature=30000
- **Scope:** P0 ORACLE ONLY. No training with reduced precision. No quality tuning. No threshold search after aggregation. All experiments executed on EPIC-05 (remote guard passed).

## Research question

Does mature 3DGS training contain Gaussian-level state redundancy that allows adaptive precision reduction without changing optimization behavior? The unit of decision is a Gaussian primitive, not a whole tensor.

## Preregistered gates (fixed before aggregation)

- **TAU** (stability / representation-error scale): `0.01`
- **Stable rule:** composite: Gaussian stable iff max-over-5-attrs rel_change <= TAU
- **Primary stability window:** densify_complete(15000) -> mature(30000), frozen topology
- **Gate A (stability):** >= 0.67 of scenes with stable_fraction >= 0.5
- **Gate B (error):** stable subset: exists p in {fp16,bf16} s.t. median bf16 attr rep-error <= 0.01 AND median bf16 optimizer-state moment rep-error <= 0.01 (all attrs/groups)
- **Gate C (economics):** stable_fraction >= 0.4 (>=20% mem reduction) AND stable_count >= 10000

## Remote guard (GSC-0)

- Host: `dsw-933622-64cc845c7d-lqswc` (EPIC-05), 8× NVIDIA A100-SXM4-80GB
- Repo HEAD: `abfa6c3e4b23caf5a927bf3722a65cba37b6fab3` (baseline/reference_v1)
- Torch 2.9.1+cu128, CUDA 12.8, Python 3.10 (gsplat env)
- Local Windows machine used as control-plane only (no training/CUDA/JIT locally).

## GSC-1 Source audit (frozen recipe)

- Gaussian params: `_xyz[N,3]`, `_shs[N,16,3]`, `_scaling[N,3]` (log), `_rotation[N,4]` (quat), `_opacity[N,1]` (logit). One persistent Adam; state migrates across topology changes.
- Densification: clone (exact copy), split (N=2 children, parent removed), prune (opacity+screen+world), global opacity reset (min 0.01). `_gaussian_ids` gives exact lineage.
- Config: 30000 iters, seed 42, densify 500-15000 every 100, grad threshold 0.0008 (absgrad), percent_dense 0.01, opacity_reset 3000, sh_progress 1000, lambda_dssim 0.2.
- **Structural fact:** densification ends at iter 15000 and opacity resets occur only at 3000/6000/9000/12000 (< 15000). The population is therefore FROZEN from iter 14901 onward, so the 15000->30000 window has an identical Gaussian ID set - a clean per-Gaussian stability measurement.

## GSC-2 Trace collection

| scene | N@dc | N@mature | frozen topology | clones | splits | prunes |
|---|---|---|---|---|---|
| room | 869212 | 869212 | True | 1131193 | 232477 | 607085 |
| garden | 2680082 | 2680082 | True | 2744981 | 874903 | 1078568 |
| bicycle | 3824885 | 3824885 | True | 4183351 | 1306657 | 1719398 |

## GSC-3 Stability analysis (mature window 15000->30000)

Relative change `||x_t - x_(t-k)|| / max(||x_t||, eps)` per Gaussian, matched by `_gaussian_ids`.
Distribution over the matched population (mature window 15000->30000):

**Whole-Gaussian (composite = max over 5 attributes) stable fraction:**

| scene | matched N | composite stable_fraction (<=TAU) | stable_count |
|---|---|---|---|
| room | 869212 | 8.513e-04 | 740 |
| garden | 2680082 | 1.118e-03 | 2997 |
| bicycle | 3824885 | 1.243e-03 | 4756 |

**Per-attribute relative-change distributions and stable fractions:**

### room

| attribute | median | p90 | p99 | stable_fraction (<=TAU) |
|---|---|---|---|---|
| xyz | med=1.529e-03 p90=4.312e-03 p99=9.052e-03 | 9.933e-01 |
| scaling | med=9.171e-02 p90=1.882e-01 p99=2.954e-01 | 1.521e-02 |
| rotation | med=1.071e-01 p90=2.393e-01 p99=4.388e-01 | 1.101e-02 |
| shs | med=2.843e-01 p90=6.060e-01 p99=9.337e-01 | 2.527e-01 |
| opacity | med=1.947e-01 p90=7.739e-01 p99=6.849e+00 | 2.662e-02 |
| **composite_max** | med=3.825e-01 p90=8.741e-01 p99=6.849e+00 | 8.513e-04 |

### garden

| attribute | median | p90 | p99 | stable_fraction (<=TAU) |
|---|---|---|---|---|
| xyz | med=7.063e-04 p90=2.133e-03 p99=5.362e-03 | 9.987e-01 |
| scaling | med=9.125e-02 p90=2.134e-01 p99=3.195e-01 | 9.983e-03 |
| rotation | med=1.051e-01 p90=2.388e-01 p99=4.342e-01 | 2.397e-03 |
| shs | med=3.565e-01 p90=6.270e-01 p99=9.221e-01 | 1.511e-01 |
| opacity | med=2.000e-01 p90=1.235e+00 p99=1.202e+01 | 3.162e-02 |
| **composite_max** | med=4.438e-01 p90=1.242e+00 p99=1.202e+01 | 1.118e-03 |

### bicycle

| attribute | median | p90 | p99 | stable_fraction (<=TAU) |
|---|---|---|---|---|
| xyz | med=8.952e-04 p90=2.839e-03 p99=6.978e-03 | 9.968e-01 |
| scaling | med=9.930e-02 p90=2.067e-01 p99=3.173e-01 | 9.149e-03 |
| rotation | med=1.182e-01 p90=2.604e-01 p99=4.542e-01 | 4.894e-03 |
| shs | med=3.568e-01 p90=7.086e-01 p99=1.045e+00 | 2.061e-01 |
| opacity | med=2.754e-01 p90=1.883e+00 p99=1.854e+01 | 2.301e-02 |
| **composite_max** | med=5.155e-01 p90=1.885e+00 p99=1.854e+01 | 1.243e-03 |

## GSC-4 Precision error oracle (mature snapshot)

FP32 reference vs FP16/BF16 candidate. Representation error = `||cast(x)-x|| / max(||x||, eps)` per Gaussian. Optimizer-state error uses a population-level magnitude floor (well-posed for near-zero moments). No training run with reduced precision.

### room  (N_mature=869212, sh_slots=16)

Representation error - ALL Gaussians:

| attribute | fp16 med | bf16 med | fp16 overflow |
|---|---|---|---|
| xyz | 1.871e-04 | 1.498e-03 | 0.00e+00 |
| scaling | 1.898e-04 | 1.520e-03 | 0.00e+00 |
| rotation | 1.813e-04 | 1.448e-03 | 0.00e+00 |
| shs | 1.972e-04 | 1.572e-03 | 0.00e+00 |
| opacity | 1.707e-04 | 1.380e-03 | 0.00e+00 |

Optimizer-state error - STABLE subset (Adam moment rep-error):

| group | fp16 med | bf16 med |
|---|---|---|
| xyz | 0.000e+00 | 0.000e+00 |
| shs | 0.000e+00 | 0.000e+00 |
| opacity | 0.000e+00 | 0.000e+00 |
| scaling | 0.000e+00 | 0.000e+00 |
| rotation | 0.000e+00 | 0.000e+00 |

Memory model: 708 B/Gaussian (FP32 state), 354 B (FP16). Stable subset = 740 Gaussians (869212 mature). State-memory reduction = 0.0% (0.3 MB saved); attr-only saving = 0.1 MB.

### garden  (N_mature=2680082, sh_slots=16)

Representation error - ALL Gaussians:

| attribute | fp16 med | bf16 med | fp16 overflow |
|---|---|---|---|
| xyz | 1.880e-04 | 1.504e-03 | 0.00e+00 |
| scaling | 1.936e-04 | 1.549e-03 | 0.00e+00 |
| rotation | 1.806e-04 | 1.444e-03 | 0.00e+00 |
| shs | 2.010e-04 | 1.610e-03 | 0.00e+00 |
| opacity | 1.688e-04 | 1.352e-03 | 0.00e+00 |

Optimizer-state error - STABLE subset (Adam moment rep-error):

| group | fp16 med | bf16 med |
|---|---|---|
| xyz | 0.000e+00 | 0.000e+00 |
| shs | 0.000e+00 | 0.000e+00 |
| opacity | 0.000e+00 | 0.000e+00 |
| scaling | 0.000e+00 | 0.000e+00 |
| rotation | 0.000e+00 | 0.000e+00 |

Memory model: 708 B/Gaussian (FP32 state), 354 B (FP16). Stable subset = 2997 Gaussians (2680082 mature). State-memory reduction = 0.1% (1.1 MB saved); attr-only saving = 0.4 MB.

### bicycle  (N_mature=3824885, sh_slots=16)

Representation error - ALL Gaussians:

| attribute | fp16 med | bf16 med | fp16 overflow |
|---|---|---|---|
| xyz | 1.857e-04 | 1.485e-03 | 0.00e+00 |
| scaling | 1.926e-04 | 1.541e-03 | 0.00e+00 |
| rotation | 1.829e-04 | 1.464e-03 | 0.00e+00 |
| shs | 2.026e-04 | 1.619e-03 | 0.00e+00 |
| opacity | 1.690e-04 | 1.359e-03 | 0.00e+00 |

Optimizer-state error - STABLE subset (Adam moment rep-error):

| group | fp16 med | bf16 med |
|---|---|---|
| xyz | 6.364e-16 | 0.000e+00 |
| shs | 0.000e+00 | 0.000e+00 |
| opacity | 8.009e-15 | 7.130e-18 |
| scaling | 0.000e+00 | 0.000e+00 |
| rotation | 0.000e+00 | 0.000e+00 |

Memory model: 708 B/Gaussian (FP32 state), 354 B (FP16). Stable subset = 4756 Gaussians (3824885 mature). State-memory reduction = 0.1% (1.7 MB saved); attr-only saving = 0.6 MB.

## GSC-5 Gate results

| gate | rule | scenes passing | pass |
|---|---|---|---|
| A stability | >= 0.67 of scenes with stable_fraction >= 0.5 | 0/3 | False |
| B error | stable subset: exists p in {fp16,bf16} s.t. median bf16 attr rep-error <= 0.01 AND median bf16 optimizer-state moment rep-error <= 0.01 (all attrs/groups) | 3/3 | True |
| C economics | stable_fraction >= 0.4 (>=20% mem reduction) AND stable_count >= 10000 | 0/3 | False |

Gate B per-scene detail (attr_ok = all attributes median rep-error <= TAU; adam_ok = all optimizer-state groups median moment rep-error <= TAU):

| scene | fp16 attr_ok | fp16 adam_ok | bf16 attr_ok | bf16 adam_ok |
|---|---|---|---|---|
| room | True | True | True | True |
| garden | True | True | True | True |
| bicycle | True | True | True | True |

**Verdict: `DROP_NO_STABLE_STATE`**

### Interpretation

Under the preregistered WHOLE-GAUSSIAN gate (a Gaussian is compressible only if ALL five attributes are stable within TAU), fewer than 2/3 of scenes show a substantial stable population. See the nuance below.

### Key nuance: whole-Gaussian vs per-attribute stability

The preregistered gate treats the Gaussian primitive as the unit of decision (composite = max over the 5 attributes). Under that reading, mature 3DGS does **not** contain a substantial whole-Gaussian stable population: only ~0.1% of Gaussians hold ALL of position, scale, rotation, SH, and opacity within TAU over the mature window.

However, stability is strongly **attribute-heterogeneous**:

- **Position (xyz)** is highly stable in every scene (~99%+ of Gaussians below TAU).
- **Scale, rotation, SH, opacity** keep drifting 5?30%+ (relative) through maturity; only ~1?25% fall below TAU.

This means whole-Gaussian redundancy is absent, but **position-only** redundancy is present. A per-attribute precision policy (e.g., aggressive position compression, conservative scale/SH) is a GSC1 question; it is outside the P0 whole-Gaussian gate, which therefore returns DROP. The position finding is the actionable signal for any follow-up.

### Note on the optimizer-state error metric

Gate B evaluates the stable subset. The composite-stable Gaussians are the most converged, so their Adam moments are near zero; the absolute FP16/BF16 representation error of these near-zero moments is therefore negligible (~1e-15), far below TAU against the population-moment floor (computed over all mature Gaussians). This is honest: for converged Gaussians the optimizer state carries almost no information, so its precision is immaterial. The attribute representation error is the meaningful compression target and is small for both FP16 and BF16.

### Binding gate

Gate A (stability) is the binding gate. The stable subset is representable within the error bound (Gate B passes in all scenes), but it is far too small (~0.1% of Gaussians) and yields negligible memory reduction (Gate C fails). Hence DROP_NO_STABLE_STATE: whole-Gaussian state redundancy is absent at maturity, though position-only redundancy is present (see nuance above).

## Provenance & reproducibility

- Frozen recipe: `baseline/reference_v1` (git `abfa6c3e`). No code modification to the frozen recipe.
- Harness: `gsc0_trace.py` reproduces the canonical 30K loop verbatim + read-only `no_grad` snapshots at the 3 preregistered phases + read-only per-Gaussian lineage accumulators (visibility count, gradient-norm sum) keyed by `_gaussian_ids`.
- gsplat: private patched tree `/root/gsc0_p0/gsplat_pkg` (true-accutile, `_backend.py` JIT flags fixed, stale `csrc.so` removed), built to `/root/gsc0_ext2/gsplat_cuda/gsplat_cuda.so` under torch 2.9.1. `accutile=False` (reference_v1 default).
- Analysis: `gsc0_analysis.py` (this report + JSONs). Thresholds fixed before aggregation.
- Raw artifacts: per-scene `traces/<scene>_<phase>_iter*.pt` (sha256 in `GSC0_TRACE_INDEX.json`).
- See `MANIFEST.sha256` for artifact hashes.

## P0 scope notes / limitations

- Gradient-magnitude and visibility statistics are cumulative lifetime proxies (grad_sum/vis_count); per-iteration gradient trace replay is forbidden in P0, so fine-grained temporal gradient stability is not measured.
- Accumulated drift is estimated statically (FP16/BF16 cast of optimizer moments + Adam update-direction perturbation), not by a reduced-precision training run (forbidden in P0).
- SH cross-phase comparison uses only the active SH slots common to both phases (4 slots early->mature; 16 slots within the mature window).

