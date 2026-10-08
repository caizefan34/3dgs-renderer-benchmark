# DSH Handoff — CVT-GS CVT0

Execute `CVT0_PREREG.md` exactly.

Critical:
1. Use corrected SB0 train/true-holdout semantics.
2. True holdout must NEVER enter source selection.
3. One source = one exact native Adam step from an identical cloned state.
4. Build all 8x8 transfer matrices before interpretation.
5. Highest-current-loss is a mandatory baseline because TurboGS already uses error-aware view selection.
6. No learned scheduler, multi-view batch, or full training in CVT0.
7. Run `cvt0_analyze.py` on finalized summary; analyzer owns verdict.
8. If DROP, close this mechanism. Do not change panels/gates.

Return exact hashes, camera IDs, raw matrices, report, logs and manifest.
