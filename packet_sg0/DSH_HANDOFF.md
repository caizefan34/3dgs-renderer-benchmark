# DSH Handoff — SplatMG SG0

Execute `SG0_PREREG.md` exactly.

Important implementation guidance:

1. This is an ORACLE study. Do not invent a deployable multigrid solver yet.
2. Preserve the original full-resolution loss. P17's coarse-loss route is a separate failed direction and must not be reused.
3. Reuse the existing frozen C0 checkpoints, optimizer state, camera lists, evaluation helpers, and hash manifests.
4. Implement hierarchy/projection analysis outside the frozen renderer whenever possible.
5. The only training required is short no-densification native Adam rollouts of K=4 and K=8.
6. Quaternion displacement/projection must be done in SO(3) tangent coordinates; do not subtract raw quaternion components.
7. Validate the identity-basis control before accepting any coarse projection numbers.
8. Random-cluster control is diagnostic only.
9. Do not add hierarchy levels, cluster heuristics, basis families, or thresholds after seeing results.
10. Run `sg0_analyze.py` exactly once on the finalized `SG0_SUMMARY.json`. The analyzer owns the verdict.

Return all required JSONs, report, raw logs, exact script hashes, and immutable manifest.

Do not write CUDA.
Do not run 30K.
Do not manually reinterpret a DROP.
