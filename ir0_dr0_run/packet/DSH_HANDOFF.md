# DSH Handoff — IRGS / DR0

You are an execution agent, not the scientific lead.

Run `IR0_PREREG.md` exactly.

Important shortcuts:
- Before writing any new instrumentation, search the completed formal A0 artifacts for existing per-hit Gaussian IDs, masks, ordered bucket/hit traces, camera IDs, and checkpoint identities.
- Reuse raw A0 traces only when their semantics exactly satisfy an IR0 metric.
- Never use the failed A0-P10 OOM path as the data source when the formal A0 path already produced valid 4-cell/8-camera artifacts.
- Stage IR0-A should be mostly offline analysis over existing traces plus native gradients/optimizer updates.
- If the hard fixed-support locality stop fires, do not run IR0-B/C.
- If IR0-A passes, implement the smallest possible measurement-only patch for sparse-update support/order census. Do not implement persistent segment trees.

Return the complete required artifact bundle and the stdout of `ir0_analyze.py`.

No manual verdict override.
