# DSH TASK — N1D Execution Only

You only execute EPIC-05/A100 actions. Do not redesign the method or decide the verdict.

1. Verify the exact same P12/C0 and checkpoint identities used by N1.
2. Reuse the original N1 anchors and held-out sets exactly.
3. Freeze three new deterministic random partners per anchor before running any new action.
4. From identical restored states execute:
   - A0
   - A1 LR x1.00
   - A1 LR x0.75
   - A1 LR x0.50
   - A1 LR x0.25
   - A2-r1
   - A2-r2
   - A2-r3
5. One-step LR scaling must not modify Adam moments, beta, eps, global step, or future LR schedule. Restore the original schedule after each counterfactual.
6. Evaluate the same held-out 8-camera loss after every action.
7. Measure representative action timings with 10 warmups + 30 repeats using CUDA events + host wall.
8. Export raw rows only. ChatGPT will compute the frontier/verdict.

Required outputs:
- N1D_SOURCE_IDENTITY.json
- N1D_PARTNER_LIST.json
- N1D_ACTION_RESULTS.parquet (or JSON if parquet tooling unavailable)
- N1D_ACTION_TIMING.json
- N1D_CONTAMINATION.json
- MANIFEST_N1D.json

Forbidden:
- changing anchors/held-out cameras;
- partner selection from outcomes or gradients;
- A2-best search;
- threshold tuning;
- controller training;
- sequential/full training;
- garden;
- scientific verdict.

STOP after raw artifact return.
