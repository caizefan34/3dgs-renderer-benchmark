# WV-HIER-R1-MX — Stage B-min v2, corrected scientific protocol

## Scientific findings retained from Stage A
Under independent Bernoulli parent activation `M_T~Bern(q)` and independent conditional child activations `m_t~Bern(r)`, with `p=qr` and a **fixed additive per-pixel SSE gradient**, the HT estimator is unbiased. The exact variance identity, componentwise and also for squared vector L2 error, is

`V_HWCS=(1-alpha) V_G4 + alpha V_G16`, `alpha = r(1-q)/(1-p)`.

For p=.25, HWCS-1 (q=.5,r=.5) alpha=1/3; HWCS-2 (q=.75,r=1/3) alpha=1/9.

**CORRECTION:** Stage A JSON wrote `1-p*q*r` in the alpha denominator. That is `1-p^2`, not `1-p` and is inconsistent with Stage A's actual numerical alpha and derivation. Update source metadata; do not silently alter frozen reports.

## What Stage B-min does *not* measure
- Running dense full-render + masked loss does **not** execute a sparse raster CUDA kernel.
- Counts of chosen child tiles, or geometric Gaussian-child overlaps, do **not** measure the *time-efficiency* `kappa` of child refinement.
- Existing gsplat CUDA internal emitted intersection entries cannot be labeled "measured" unless they are actually extracted from a verified kernel-visible buffer. With no CUDA build/instrumentation, counts reconstructed from projected bboxes are still a geometry proxy.
- Gradient NMSE on two cameras does not establish time-to-quality or convergence.

## Fix cost interpretation
Let `rho=I_G4/I_G16` and `Gamma` be relative raster work. Stage A assumed `Gamma=1`, with

`W_N1_G4 = (1 + Gamma*p)*rho`

`W_HWCS = 1 + [p + kappa*(q-p)]*rho + Gamma*p*rho`.

At p=.25,q=.5,kappa=1, `W_HWCS=1+(.5+.25Gamma)rho`. The condition `W_HWCS < rho` (below the **N1 dense-binning-only** floor) requires `1/rho + .5 + .25Gamma < 1`. At Gamma=1 this simplifies to `rho>4`; that is *not* unconditional in unknown Gamma, and is not a result about GPU time or N2.

For selected child candidate count ratio `R=selected_or_visited_child_intersections / I_G4` under Stage A's idealization, `R=.25 + .25*kappa` if count operations were weighted uniformly. Thus `R<=.5` for **every kappa in [0,1]**. The original F2 condition "HWCS-1 entries <=0.5 G4 entries => kappa<=0.5" is invalid; it cannot be used as a GO gate. In reality kappa is an *operation-cost efficiency* parameter, and raw counts do not identify it.

Compare also **Native G4 N2 = sample child tiles, then bin/sort only selected tiles**, not just N1 = dense G4 bin/sort, then mask. Under an optimistic idealization N2 avoids dense work; actual cost must be measured rather than assumed.

## Correct falsification protocol
For each camera, use G16, G4, HWCS1, HWCS2 *measured in the same 32-seed replay*, and calculate:

`d_1(s)=NMSE_HWCS1(s) - [(2/3)NMSE_G4(s) + (1/3)NMSE_G16(s)]`

`d_2(s)=NMSE_HWCS2(s) - [(8/9)NMSE_G4(s) + (1/9)NMSE_G16(s)]`

The expectation of each `d` is zero under the exact estimator model; sample-by-sample differences need not be zero. Retain all rows, compute per-camera paired bootstrap 95% CI (resampling paired seed *vectors*), and report mean and uncertainty. Include a pool-across-cameras analysis only if it is clearly labeled and stratified by camera.

Do **not** use Stage A's 8-camera aggregate predicted point (e.g., 1.4247) as a hard 2-camera falsification interval. A deviation could simply be camera heterogeneity. Do **not** DROP when the CI is wide or the pilot has insufficient statistical power; report INCONCLUSIVE.

Measure improvement over same-camera G16 as `NMSE_G16 - NMSE_HWCS1` with paired 95% CI, and report effect size. This is the important quality gate. Strict ordering of variance G4<=HWCS<=G16 is empirically scene-dependent: signed child-gradient cross terms can be negative; only the convex blend identity is universal under assumptions.

## Sampling and loss invariants
- Pixel mask is 1/p for active 4x4 child pixels and 0 for other pixels (G16 uses active parent with the same weight p).
- Generate CPU Bernoulli uniforms with a fixed seed *once*, shared consistently between methods: independent `U_parent` of shape ceil(H/16)xceil(W/16) and `U_child` of shape ceil(H/4)xceil(W/4).
- G16 selects parents by U_parent<p; G4 selects children by U_child<p; HWCS selects parents by U_parent<q and children by U_child<r; per-pixel inclusion remains exactly p. This provides stable cross-method common-random-number coupling.
- Transfer mask to the tensor's device **before** creating the weighted loss; do not use `torch.where` with a CPU mask and CUDA branches. Check device, shape, dtype explicitly.
- Keep the same additive SSE reduction across dense and sparse losses. Divide by the fixed total pixel denominator, not a random selected-pixel denominator. No DSSIM or nonseparable image loss in this oracle.
- Masks are defined in pixel space, independent of model predictions/targets; no adaptive sampling that changes inclusion probabilities without new HT weights.
- Boundary children and parents use ceil partitioning, and each valid pixel belongs to one and only one child and parent.

## Run matrix and provenance
- On idle MX A100 other than the TEMPORAL_COVERAGE-R1 reserved GPU, original WV0 environment, do not modify original artifacts.
- bicycle__iter15000, cameras IDs 0 and 8, p=.25, 32 *explicit* seeds x 4 methods x 2 cameras. Two dense references, one per camera. Zero optimizer steps.
- Record `sha256` of checkpoint, source, CUDA binary, input camera and target, mask child bits, and full environment/command.
- Save per-camera/seed/method NMSE, cosine, rel-L2, dense grad L2, sparse grad L2, dot, selected parent/child counts, selected pixel count, Gaussian-parent and child geometry candidate counts if attainable; distinguish counts from real emitted entries.
- Stage C: if Stage B gradient oracle passes, implement a *minimal real sparse CUDA prototype*, compare N1, N2 and HWCS on matched `p`, gradient quality and achieved GPU kernel/F+B timing; then only consider training E2E/time-to-quality.

## Decision vocabulary
- `ORACLE_PASS`: exact/expected gradient estimator behavior and same-camera blend supported; **not** a performance GO.
- `ORACLE_INCONCLUSIVE`: protocol valid, sample uncertainty too large; consider added seeds only after explicit approval.
- `MODEL_OR_IMPLEMENTATION_MISMATCH`: paired estimator identity unexpectedly violated; inspect masks, loss and controls.
- `STRUCTURAL_POTENTIAL`: derived/counted cost model suggests improvement against explicit N1/N2 assumptions, not measured speed.
- `TIMING_NOT_MEASURED`: required Stage C CUDA evidence absent.
- `CODEX_CUDA_PROTOTYPE_PROPOSAL`: Stage B has not yet granted executable CUDA authorization.
