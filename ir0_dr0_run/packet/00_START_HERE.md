# IRGS / DeltaRaster-GS — DR0 Start Here

Working name: **IRGS (IncrementalRaster-GS)**.
`DeltaRaster-GS` is a descriptive codename only; use IRGS in artifact names to avoid confusion with the project's previously closed DeltaGS/SplatCompiler branch.

## Scientific claim under test

Can 3DGS training be reformulated so that the differentiable rasterizer maintains persistent compositing state across optimization iterations and repairs only the state touched by the Gaussians that are actually updated?

This is NOT ordinary stale-state reuse.

The proposed invariant is:

- inactive Gaussian parameters are deliberately frozen for an inner/update block;
- their projection/intersection/compositing state is therefore valid by construction;
- active Gaussian changes are repaired exactly in the affected projection/intersection/compositing regions;
- alpha compositing summaries use the associative operator

    `(C1,T1) ⊗ (C2,T2) = (C1 + T1*C2, T1*T2)`.

DR0 does NOT implement the dynamic tree. It asks whether enough sparse-update / touched-work headroom exists to justify that engineering.

## Execution order

1. `IR0_PREREG.md`
2. `IR0_METRICS_SCHEMA.json`
3. produce `IR0_SUMMARY.json`
4. run `python ir0_analyze.py IR0_SUMMARY.json`
5. analyzer verdict owns promotion/drop
6. only `IR0_STRONG_PROMOTE` or `IR0_PROMOTE` may proceed to IR1

No production CUDA in DR0.
