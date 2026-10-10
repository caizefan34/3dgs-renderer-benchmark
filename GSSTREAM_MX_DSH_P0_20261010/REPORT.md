# GSSTREAM-P0 — Falsification Report (2026-10-10)

CPU-only preregistered falsification of the GSSTREAM-P0 hypothesis. No CUDA, no GPU,
no training. This report follows the preregistered OBSERVED / EVIDENCE / HYPOTHESIS /
FALSIFICATION / DECISION structure.

## 1. Question tested
Could a GPU-cheap **compressed row-span support index** eliminate per-pixel Gaussian
presence checks in a differentiable rasterizer while preserving forward + all backward
families, with a build cheap enough to help end-to-end? (See DSH_TASK.md.)

## 2. Hypothesis
Per-scanline exact ellipse support intervals, materialized as a rank-stable compressed
stream during binning and consumed in BOTH forward and backward, would reduce candidate
work / memory enough to matter, without an already-published equivalent.

## 3. What would falsify it
- G0: an existing paper/code performs the same compressed build-and-consume mechanism.
- G1: a CPU correctness failure (false-negative culling, gradient mismatch, unsound branch).
- G2: real-data representation density does not show the preregistered >=5x exact-hit and
  >=~2.86x conservative-candidate reduction.
- G4/G5: theoretical headroom below the frozen thresholds (f>=2/3 for 3x; >=1.30x full-step
  with nonnegative memory).

## 4. Experiment performed (ordered, CPU-only)
1. **Literature gate** — PRIOR_ART_MATRIX.md (5 sources + collision analysis).
2. **Correctness audit** — independent review + finite-difference gradient check
   (scratch_audit/CORRECTNESS_AUDIT.md, fd_gradcheck.py).
3. **Preflight reproduction** — re-ran test_properties.py + cpu_oracle.py on seeds 42/123;
   byte-compared to packaged evidence (see preflight_reproduction.md).
4. **Reality gate** — searched repo for authentic projected Gaussian data
   (see reality_gate_report.md). FOUND real room-scene FastGS projections.
5. **Cost model** — synthetic + REAL room cohort; footprint vs row-span (COST_MODEL.json,
   real_cohort_cost_honest.json).

## 5. Controls
Seeds 42/123 reproduced; H,W=1600x1066 and tile=16 taken from the authoritative
camera_tile_manifest.json sidecar (not guessed). Candidate baselines use the rasterizer's
own 3-sigma screen-radius footprint math (gsplat/FastGS) and the kernel's real emitted
(intersecting) pair counts. Packaged manifest SHA256 verified self-consistent; reproduced
artifacts logically byte-identical after LF/CRLF normalization.

## 6. What was measured
- **Correctness:** 400/400 property tests; parity 0.0; analytic gradients match FD to ~1e-9.
- **Prior art:** 5 sources; gsplat packed/sparse fwd/bwd + 3DGART identified as collisions.
- **Real-data density (8 room cameras, N=572546):** real_intersect / conservative_footprint
  ≈ 0.020 (median); footprint_tiles/real_intersect ≈ 50.4; rowspan_candidates / footprint_pixels
  ≈ 6.79 (median); footprint tiles per visible Gaussian ≈ 59 (median).

## 7. Mechanism-level evidence
- The strongest existing renderer (FastGS/gsplat-class) already performs exact opacity +
  exact ellipse culling, removing ~98% of its own conservative footprint candidates. The
  opportunity the hypothesis targets is therefore already captured by the baseline.
- The proposed row-span structure is *denser* than the per-tile footprint it would replace
  (~6.8x more candidate pixels, ~183–1120 candidates/gaussian vs ~59 footprint tiles/gaussian).
  Materializing it adds bytes and a build pass on top of gsplat's already-shipped packed
  per-tile index that is already reused in both passes.

## 8. E2E result
Not measured (CPU-only P0, no GPU authorized). Not required: G0/G2/G5 fail on structure and
cost grounds before any E2E run would be meaningful.

## 9. Correctness / quality
CPU toy-objective parity and gradients are correct. Native fp32/clamp/sort/SH/absgrad
parity is NOT TESTED (by design). Real data is authentic but the shipped oracle cannot
ingest the 99.76% of gsplat conics that are zero-padded for culled Gaussians (hard SPD
assert, no dense fallback) — a G3 robustness gap, not a silent bug.

## 10. What remains uncertain
- web_search unavailable; prior-art sweep not exhaustive for "scanline/interval compressed
  list reused in backward" beyond the five named sources. No contrary evidence found.
- G4 headroom unmeasured (no GPU / no same-hardware fused f). Structural argument indicates
  the ceiling is far below the 3x requirement regardless.

## 11. Verdict
**STOP_NOVELTY.** G0 FAIL (mechanism already published: gsplat packed/sparse active-tile
fwd/bwd + 3DGART backward buffer), independently corroborated by G2 FAIL and G5 FAIL on
real room-scene data (proposed stream ~6.8x denser than the structure it replaces, while the
existing renderer already culls to ~2% of its conservative footprint). G1 PASS_WITH_CAVEAT,
G3 HOLD, G4 BLOCKED.

Per the preregistered outcome rule: G0 fail ⇒ STOP. Not P0_CANDIDATE_ONLY (G0 not distinct).
Not HOLD-on-data (real data IS available; the STOP is on novelty/cost). **P0 authorizes
nothing — no P1 experiment, no GPU execution.**

## 12. Highest-value next action
None authorized by this STOP. If the program continues at all, it must first re-scope to a
narrow, defensible claim — per-ROW vs per-TILE rank-stable segment compression — and beat
gsplat's packed per-tile baseline on measured build bytes / sort cost / backward global
traffic on the same hardware, citing AccuTile/Speedy-Splat, gsplat, and 3DGART as prior art
it must exceed. Also correct the FlashGS citation (arXiv:2408.07967, Aug 2024; no
FlashSplat/multi-resolution) before registering anything.

## Provenance
- Local git HEAD: ddb2354f299aa10ae1567309111df66a586061ae
- Kit source SHA256: cpu_oracle.py d639c522…, test_properties.py 07c82377…
- Real data: native_tiles_*.npz (room, FastGS iter-30000) + camera_tile_manifest.json (see COST_MODEL.json for all hashes).
- Reproduction: logically byte-identical to packaged out/s42, out/s123 after LF/CRLF normalize; packaged manifest self-consistent.
