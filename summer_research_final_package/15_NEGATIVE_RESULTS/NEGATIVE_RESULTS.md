# NEGATIVE_RESULTS.md — Failed / Falsified / Dismissed Candidates

> This file lists every hypothesis that was tested and rejected. Per the project's
> scientific-integrity rules, negative results are mandatory content for the final report.
> Each entry: hypothesis -> experiment -> result -> evidence file.

## 1. C1 depth compression 32->16 bit (RTX 5070)
- Hypothesis: 16-bit depth keys halve sort memory and cut sort latency ~2x.
- Experiment: 30K iteration training, 11 scenes, seed 42; kernel-level timing harness.
- Result: sort latency reduced ~1.2% but total forward time improved only ~0.8%
  (32.05ms -> 31.78ms). Advisory threshold (<1%) not met; noisy on repeated runs.
- Verdict: FAILED_TO_MEET_BAR (drop).
- Evidence: reports/phase-c1/c1_final_report.md; results/a100/phase-c1/*.json

## 2. C1 12->8 pass reduction (unverified claim)
- Hypothesis: depth-sort pass count drops from 12 to 8 (from comment in code).
- Result: audit found the 12/8 numbers come from a code comment, not from measurement;
  no profiler data supports it. Marked UNVERIFIED, not a valid basis for speedup claim.
- Evidence: reports/phase-c1/c1_audit_unverified_claim.md

## 3. C17-0 segmented sort
- Hypothesis: splitting sort into segments improves cache locality.
- Result: benchmark shows seg-sort slower than baseline in 9/11 scenes (avg +4.1%
  kernel time). Root cause: smaller segments reduce occupancy.
- Verdict: DROP.
- Evidence: reports/phase-c17/c17_0_segmented_sort.md

## 4. C17-1 tile-local queues
- Hypothesis: per-tile queues avoid global atomic contention.
- Result: correct but queue allocation overhead dominates; not a speedup in any scene.
- Verdict: DROP (code kept for reference).
- Evidence: reports/phase-c17/c17_1_tile_queues.md

## 5. C17-3 metadata cache
- Hypothesis: caching per-tile metadata removes repeated loads.
- Result: CUDA cache already handles the working set; no measurable gain; adds complexity.
- Verdict: DROP.
- Evidence: reports/phase-c17/c17_3_metadata_cache.md

## 6. C43 adaptive tile-size
- Hypothesis: runtime tile-size autotuning beats fixed tile16.
- Result: adaptive search overhead > gain in all scenes; no deterministic winner.
- Verdict: FALSIFIED.
- Evidence: reports/phase-c43/c43_adaptive_tile_*.md

## 7. C49 as independent candidate
- Hypothesis: attribute-decoupled backward is a standalone optimization.
- Result: after G_dens/G_opt separation correction, max theoretical E2E gain ~1.13%;
  below threshold; absorbed into C51 methodology instead.
- Verdict: DROP_AS_INDEPENDENT (merged).
- Evidence: reports/phase-r2-attribute-decoupled-backward-gate.md

## 8. C51-B2 partial-update backward
- Hypothesis: updating only 80% of Gaussians keeps quality with backward savings.
- Result: B2 produced inconsistent updates (-0.49 dB PSNR / -0.0054 SSIM at B1
  settings); violates consistency requirement.
- Verdict: FAILED (B1/B3 adopted; B2 not).
- Evidence: reports/phase-c51/c51_b2_gate.md

## 9. Flat-Lohmann instruction selection (REPEALED)
- Hypothesis: DPP-based instruction selection improves merge throughput.
- Result: after data-preservation review, the claim was retracted; flagged REPEALED.
- Evidence: reports/phase-r0.1-evidence-correction.md

## 10. BloomGPU lift (REPEALED)
- Hypothesis: Bloom GPU memory reduces footprint.
- Result: repeated benchmark could not reproduce; decision rolled back. REPEALED.

---

## Writing rules for the final report
- Never present a rejected candidate as "supported".
- When quoting a negative result, cite both the negative finding and the control
  (i.e. the baseline it was compared against).
- Use the same thresholds everywhere: <1% speedup -> DROP; quality delta outside
  PSNR/SSIM/LPIPS bounds -> FAILED; methodological issue -> REPEALED/FALSIFIED.