# R17 Source Audit — patched package (2026-10-08)

This is **a corrected copy** of the user-supplied R17 package, not an MX-executed result. The uploaded original zip remains unchanged.

## Corrected engineering defects

1. `scripts/capture_intersections.py::_subset_rerun`: sample `n_images` must be computed from **sampled** `[C, k, 2]` data, not the original `[C, N, 2]` tensor divided by sample size. The original error reported `C*N/k` cameras and caused P5's estimator to greatly overstate the number of full-scene intersections. The corrected expression uses `small["means2d"].numel()`.
2. `scripts/r17_p3_controlled.py::run_cell`: initialize `result = None` before the forward try block, preventing a secondary UnboundLocalError if OOM/abort happens prior to assignment.
3. Missing real camera metadata: P3 still runs synthetic control cells and emits explicit missing-camera rows for real cells; P4/P5 emit explicit unavailable results rather than crashing before reporting.
4. Reruns of P3 and P4 archive the previous CSV outputs into `logs/` before writing new rows, rather than silently appending mixed generations.

## Research limitations (not fixed by code)

- **P3's forward CUDA-event timing includes diagnostic interception and extra GPU subset replay. It is not a clean renderer speed benchmark.** Use P4 timing for performance comparison, and only after correctness/workload comparability checks.
- P5's projected memory request is a **heuristic based on a sampled intersection count** and extra factors. It is not a measured peak and can underestimate as well as overestimate; `SKIP_FULL` is not equivalent to OOM, and `RUN_FULL` is not a guarantee of success.
- The "A vs D" contrast changes both input activation and camera. Use single-variable contrasts (A↔B/B2/B3, A↔C2, C↔D) before making causal attribution. Check camera semantics against training code.
- `RUN_STATUS.summary.executed == 10` only means ten deliverable files parsed; it is **not** evidence that all eight P3 cells and four P4 configurations succeeded or even ran. Inspect row statuses and per-step exit codes.
- Frozen binary hashes and checkpoint identity must match for cross-campaign claims. Partial runs are still diagnostically useful but must be marked as such.
- No GPU runs or MX end-to-end benchmark have been performed on this corrected package.

## CPU-only smoke checks

```bash
python3 tests/test_r17_patch.py
python3 -m compileall -q scripts reference tests
```

## MX invocation

From an empty or uniquely named R17 output directory, run:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_on_mx.sh
```

Afterwards report `logs/step_exit_codes.txt`, `RUN_STATUS.json`, the entire P3/P4 CSVs, `R17_4P4M_SAFETY_GATE.json`, and `R17_FINAL_VERDICT.json`. Avoid treating the file count as the success criterion.
