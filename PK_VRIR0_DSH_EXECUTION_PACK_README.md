# VRIR0 — Certified Structural Reuse Oracle / DSH Execution Pack

Status: FROZEN DESIGN, ORACLE-ONLY. CUDA implementation is forbidden in VRIR0.

## One-sentence hypothesis
When the same training camera is revisited after optimizer updates, the continuous Gaussian parameters may change while the discrete raster structure (Gaussian→tile membership and per-tile depth order) remains unchanged; if that structure can be certified exactly and fail-closed, gsplat's fresh intersection/sort/offset work can in principle be skipped while recomputing all current continuous attributes.

## Scope frozen for VRIR0
- Baseline harness: repository `baseline/reference_v1`, gsplat 1.5.3 lineage.
- Single-camera training path, `packed=False`, tile size 16.
- Oracle-only: baseline still performs the normal fresh structure construction every observed visit.
- Primary window: iterations 15001–30000 (post-densification static topology).
- Diagnostic-only window: 0–15000.
- Candidate isolation: do not modify or compose with AABS0 or FGS-H0 while VRIR0 is being screened.
- No `.cu`, `.cuh`, C++/CUDA extension, or renderer kernel changes.

## Reuse target
Only the discrete traversal structure:
- `flatten_ids`
- `isect_offsets`
(or an equivalent per-tile ordered Gaussian-ID list)

Do **not** reuse old `means2d`, `conics`, `opacities`, colors/SH, depths, radii, rendered pixels, alpha, or gradients. Those must be current-state values.

## Run order
1. Source/provenance audit.
2. Prior-art audit and novelty wording freeze.
3. Add oracle instrumentation only.
4. Smoke test on one tracked camera.
5. Run structural coherence oracle on room/bicycle/garden, primarily 15k–30k.
6. Validate both the exact O(M) reference certificate and the O(N) cached-slack certificate against the fresh baseline structure; false accepts must be exactly zero.
7. Compute coherence opportunity, cache footprint, certification cost proxy, and Amdahl upper bound.
8. Emit KEEP / REDESIGN / DROP. Do not implement CUDA in this package.

See `DSH_TASK.md` for executable instructions and `FAIL_CLOSED_GATES.md` for the frozen verdict policy.
