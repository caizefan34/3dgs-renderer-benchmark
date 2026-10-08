# DSH task: execute PGR0 Projection Gradient Redundancy Oracle (P0)

Execute this P0 oracle according to `README.md` and `FAIL_CLOSED_GATES.md`.
Generate a fail-closed verdict from raw measurements. CUDA is forbidden in P0.

Your job:
1. Audit and hash-record the gsplat projection source
   (`public_baselines/gsplat/gsplat/cuda/csrc/ProjectionEWA3DGSFused.cu` and
   `public_baselines/gsplat/gsplat/cuda/include/Utils.cuh`).
2. Reimplement the fused projection forward and backward faithfully in pure
   PyTorch on CPU (`scripts/pgr0_projection_ref.py`), matching `Utils.cuh`
   exactly (see `FAIL_CLOSED_GATES.md` G1 for the non-obvious conventions).
3. Validate the analytic backward against `torch.autograd` (both `recompute`
   and `reuse` modes must match autograd to ≤1e-4 relative L2).
4. Run `scripts/pgr0_oracle.py` over room/bicycle/garden, 2 cameras, both
   `calc_compensations`, `float64` and `float32`.
5. Run `scripts/pgr0_analyze.py` to apply the fail-closed gates and emit the
   verdict, stats, and gradient-similarity artifacts.
6. Refresh prior art and emit `PGR0_PRIOR_ART_REFRESH.md`.
7. Emit `MANIFEST_PGR0.json` with SHA256 of every evidence file.

Hard rules:
- no CUDA, no `.cu`/C++ build, no kernel run in P0;
- no renderer/C0/AABS0/FGS-H0/H8 source or result modification;
- no new optimization branch; this pack only screens redundancy;
- the oracle never skips baseline work and never claims a measured speedup;
- the opportunity screen is an explicit analytical model, not a timing.

Expected final verdict files (under `artifacts/`):
- `PGR0_VERDICT.json`
- `PGR0_REPORT.md`
- `PGR0_PRIOR_ART_REFRESH.md`
- `PGR0_STATS.json`
- `PGR0_GRADIENT_SIMILARITY.json`
- `MANIFEST_PGR0.json`
