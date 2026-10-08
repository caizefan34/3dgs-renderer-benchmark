# DSH Handoff — AtomBirth-GS AB0

Execute `AB0_PREREG.md` exactly.

## Guidance

1. Start from exact frozen checkpoint/camera/source manifests.
2. If 5K or 15K exact checkpoint is absent, emit `AB0_CHECKPOINT_BLOCKER`; do not substitute.
3. Keep the base scene completely frozen during candidate fitting.
4. All fitting/evaluation uses the native full-resolution renderer/loss.
5. Do not use external depth, MVS, optical flow, learned features, semantic masks, or edge detectors.
6. The cheap screening proxy is only for candidate ranking.
7. The final candidate is selected ONLY by design loss; held-out cameras are revealed after selection.
8. Candidate Adam moments start at zero; use frozen per-family LRs.
9. Implement the parent-derived baseline exactly as specified; do not replace it after seeing results.
10. Run random/existing-center controls only after primary measurements.
11. No production CUDA.
12. Run `ab0_analyze.py` exactly once on finalized `AB0_SUMMARY.json`.
13. Analyzer owns verdict.

Return every required JSON, raw log, exact executed script hash, git diff, and immutable manifest.
