# DSH Handoff — HomotopyGS H0

Execute `H0_PREREG.md` exactly.

## Implementation discipline

1. Search existing frozen C0 manifests first. Do not reconstruct checkpoint identities manually.
2. Create an isolated experiment worktree/branch; never edit the frozen C0 tree in place.
3. Before scientific runs, prove native endpoint parity:
   - E2D multiplier=1
   - OPA gamma=1
4. Reuse the exact optimizer state stored in each checkpoint.
5. Topology is frozen only for the short H0 diagnostic.
6. The training image/loss remains full-resolution and unchanged.
7. Native evaluation after each step is measurement only and must not enter training gradients.
8. Exclude evaluation overhead from step timing.
9. If 5K, 15K, or 25K exact checkpoint is absent, emit `H0_CHECKPOINT_BLOCKER`; do not substitute.
10. Do not create joint E2D+OPA or tune schedules after observing results.
11. Run `h0_analyze.py` on finalized data; analyzer owns the verdict.
12. No production CUDA beyond the smallest isolated change needed to expose the continuation parameter. Prefer existing runtime `eps2d` paths for E2D and a Python/autograd opacity transformation for OPA.

Return all required JSONs, report, raw logs, exact executed script hashes, git diff, and manifest.
