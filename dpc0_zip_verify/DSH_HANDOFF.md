# DSH-DPC — Execute DPC0

This is the highest-priority systems/reproducibility track.

EPIC-05 A100 ONLY. No local GPU.

Order:
1. Verify all four canonical checkpoint hashes and frozen binary pins.
2. Build DPC as a separate extension. Never overwrite the authoritative C0 .so.
3. Preserve the exact C0 forward path.
4. Implement tile-primitive fixed-order partial accumulation.
5. Implement deterministic Gaussian-major segmented reduction with NO float atomicAdd to final
   Gaussian gradients.
6. Run 32-repeat same-GPU hash test FIRST.
   - If any gradient hash differs: STOP -> DPC0_NONDETERMINISTIC.
7. Run cross-A100 hash test on GPU0/1/2/3.
8. Run C0 ensemble correctness, directional sanity, and one-step equivalence.
9. Only after correctness, benchmark full DPC prep+backward timing.
10. Run frozen dpc0_analyze.py. Analyzer owns verdict.

Do not integrate VkSplat-style fusion in DPC0. That is DPC1 only if DPC0 passes.

Important originality discipline:
- per-Gaussian backward itself is prior art;
- fused Adam itself is prior art;
- the candidate paper contribution is fast deterministic rasterized 3DGS training as a complete
  state-transition property.

Return all required JSON/report/manifest artifacts and raw logs needed to audit hashes/timing.
