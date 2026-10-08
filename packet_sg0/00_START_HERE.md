# SplatMG — SG0 Start Here

Working name: **SplatMG**.

## Core hypothesis

The expensive 3DGS optimizer may spend many full-resolution iterations correcting low-frequency / collective parameter errors that lie in a much lower-dimensional Gaussian hierarchy.

SplatMG is NOT:
- coarse-resolution rendering;
- image-pyramid supervision;
- a hierarchical final representation;
- HiGS macro-tile rendering;
- densification;
- a new generic second-order optimizer.

The intended mechanism is classical multilevel optimization in PARAMETER SPACE while preserving the SAME full-resolution renderer and SAME original loss.

A hierarchy defines coarse variables `z_l` and prolongation `P_l` into the fine Gaussian parameterization:

    theta = theta_fine + P_l z_l

A future implementation would use fine smoothing plus coarse correction/V-cycles.

## SG0 purpose

Before designing a real coarse solver, test an ORACLE upper bound:

> Can the actual progress made by 4–8 ordinary full-resolution Adam steps be represented by a very low-dimensional, spatially coherent coarse Gaussian subspace, and if that future-informed coarse jump were available early, would it recover most of the held-out loss improvement?

If even the future-informed oracle fails, SplatMG is closed.

Read in order:
1. `SG0_PREREG.md`
2. `SG0_SCHEMA.json`
3. execute and produce `SG0_SUMMARY.json`
4. run `python sg0_analyze.py SG0_SUMMARY.json`
5. analyzer owns verdict

No CUDA. No full 30K training. No threshold tuning.
