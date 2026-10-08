# P11 Opportunity Audit

This package intentionally contains no new CUDA optimization.

Run `DSH_P11_PROFILER_ONLY_PROMPT.md` first. The goal is to select at most one exact
candidate with enough **post-strong-baseline** E2E headroom to justify a prototype.

Current primary candidate: exact loss-adjoint/raster-backward boundary fusion.
Current backup: densification-stat sink.

Historical sparse-tail, zero-init, block-atomic, and backward-optimizer ideas are not
authorized for reimplementation without passing the new gates.
