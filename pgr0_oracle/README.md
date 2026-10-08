# PGR0 — Projection Gradient Redundancy Oracle / DSH Execution Pack

Status: FROZEN DESIGN, ORACLE-ONLY. CUDA implementation is forbidden in P0.

## One-sentence hypothesis
The gsplat fused 3DGS projection **backward** recomputes, from the raw inputs,
the same continuous intermediates (rotation `R`, world covariance, camera-space
mean `mean_c`, camera-space covariance `covar_c`, perspective Jacobian `J`, and
`covar2d`) that the **forward** already computed and then discarded; if that
recomputation is bit-similar to the forward and a cached-reuse backward is
gradient-equivalent to autograd, the recomputation is in principle redundant.

## Scope frozen for PGR0
- Baseline harness: repository `public_baselines/gsplat`, gsplat 1.5.3 lineage.
- Kernel under test: `projection_ewa_3dgs_fused` (forward + backward), `H8_MR == False`.
- Config: single image, `packed=False`, `tile_size=16`, `eps2d=0.3`,
  `near_plane=0.01`, `far_plane=1e10`, `camera_model=pinhole`,
  `calc_compensations ∈ {False, True}`.
- Oracle-only: no `.cu`/C++/CUDA extension is built or run; everything is
  pure-Python/NumPy/PyTorch on CPU. `float64` is the reference precision;
  `float32` emulates CUDA precision.
- EPIC-05 only: this pack does not reopen C0, AABS0, FGS-H0, H8, or any
  rasterizer semantics.

## Reuse target (what the backward recomputes)
- `R` (quaternion→rotation), world `covar` (quat×scale),
- `mean_c` (world→camera), `covar_c` (covariance world→camera),
- perspective Jacobian `J` and `covar2d`.

NOT reuse targets: `means2d`, `depths`, `conics`, `radii`, `compensations`
outputs, and the VJP chain itself (`persp_proj_vjp`, `posW2C_VJP`,
`covarW2C_VJP`, `quat_scale_to_covar_vjp`, `inverse_vjp`, `add_blur_vjp`),
which are the genuine gradient work.

## Run order
1. Source/provenance audit (record gsplat kernel + Utils.cuh hashes).
2. Prior-art audit and novelty wording freeze.
3. Faithful CPU reimplementation of forward + backward (`pgr0_projection_ref.py`),
   validated against `torch.autograd` (the analytic backward must match autograd).
4. Smoke test the recompute vs reuse vs autograd equivalence.
5. Run the oracle over `room/bicycle/garden`, 2 cameras, both compensations,
   `float64` and `float32`.
6. Apply `FAIL_CLOSED_GATES.md` mechanically and emit one verdict.
7. Emit `PGR0_VERDICT.json`, `PGR0_REPORT.md`, `PGR0_PRIOR_ART_REFRESH.md`,
   `PGR0_STATS.json`, `PGR0_GRADIENT_SIMILARITY.json`, `MANIFEST_PGR0.json`.

See `DSH_TASK.md` for executable instructions and `FAIL_CLOSED_GATES.md` for the
frozen verdict policy.
