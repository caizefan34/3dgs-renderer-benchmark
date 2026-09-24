# 00 · README FIRST — Instructions for the Final-Report Writer

You are an AI tasked with independently writing the **final Summer Research Report**
for a 2026 summer research project titled:

> **"Benchmarking and Optimizing 3D Gaussian Splatting (3DGS) Renderers: A Candidate-Driven Study of Tile Geometry, Sparse Computation, and Training Performance"**

This package contains a **curated, cross-indexed evidence base**. It is NOT
itself the report: it is the raw material, organized so you can write the report
with correct numbers, correct provenance, and correct honesty about what was
and was not found.

---

## 1. How to use this package

1. Start with `01_PROJECT_OVERVIEW/` to understand the project structure, research tracks, and key terminology.
2. Read `02_RESEARCH_TIMELINE/RESEARCH_TIMELINE.md` for the chronological narrative (dates, phases, candidates, commits).
3. Consult the per-topic sections `03`–`14` for detailed evidence, following the file pointers to raw data.
4. Cross-check every number against `03_FINAL_RESULT_MATRIX/` and the source JSON/CSV files referenced there.
5. If you encounter conflicting numbers, do NOT silently pick one. Report the conflict; consult `DATA_CONFLICTS.md` if present in this package or in `reports/`.
6. Respect the **cohort model**: hardware-cohort-specific results must never be presented as global truths.

---

## 2. The one-sentence summary of the research

Using a reproducible benchmark suite with official real-scene datasets (Mip-NeRF 360, Tanks & Temples, Deep Blending) and strict hardware-cohort rules, this project (a) established a validated reference baseline ("Reference V1") of the canonical 3DGS renderer, (b) exhaustively evaluated candidates that optimize the forward pass (tile geometry / tile sizes / sorting), the backward pass (sparse backward / selective attribute computation), and the training pipeline (progressive resolution, LOD, staged training), and (c) produced one formally pre-registered, confirmatory, quality-preserving training-speedup result (`gsplat_30k_fused_prune10_rclip05`) — while discovering and honestly documenting many candidates that failed or were falsified.

---

## 3. Non-negotiables (do not violate these)

| # | Rule | Why |
|---|------|-----|
| 1 | Never mix results across hardware cohorts in a single speedup claim. | The project's core methodology; see `05_PROTOCOL_AND_DATASETS/hardware_cohorts.md`. |
| 2 | `speedup` denominators must always be stated (kernel-only vs forward-only vs end-to-end vs wall). | Different sections of the report use different denominators; a bare number is meaningless. |
| 3 | Quality-gating is mandatory: any "speedup" claim must state the quality gate (PSNR/SSIM/LPIPS delta bounds) that was used, and whether it passed. | Throughout the project, speedups without quality-preservation were treated as *not claimed*. |
| 4 | Negative results are results. Do not omit them. Every falsified hypothesis is listed in `NEGATIVE_RESULTS` and must be discussed. | The project's scientific value is largely in falsification. |
| 5 | Numbers must match the cited JSON/CSV files. When you quote a number, cite the exact source file. | Auditable evidence is the whole point of this package. |

---

## 4. Where things live (map)

| Section | Contents |
|---------|----------|
| `01_PROJECT_OVERVIEW/` | One-page project summary, research questions, track map |
| `02_RESEARCH_TIMELINE/` | Chronological narrative with dates, phases, candidates, commit anchors |
| `03_FINAL_RESULT_MATRIX/` | Machine-readable final comparison, per-candidate verdicts |
| `04_CRITICAL_FINDINGS/` | The 5–8 most important findings, one page each |
| `05_PROTOCOL_AND_DATASETS/` | Reference V1 protocol, datasets, hardware cohorts, seed policy |
| `06_BASELINE_REFERENCE_V1/` | Reference V1 baseline evidence (12 unit tests, 30K run results, instrumentation) |
| `07_TILE_GEOMETRY/` | Tile-size & occupancy/hardware-dependence research (tile 8/16/20/32, A100 vs RTX5070) |
| `08_C42/` | Candidate C42: definition, experiments, baselines (Speedy/FastGS/FasterGS), final interpretation |
| `09_FORWARD_SORT/` | C1 depth compression, segmented sort, C17-1/C17-2, radix sort analysis |
| `10_BACKWARD_C25/` | Backward profiling, C25 sparse/selective backward, the ~1.7ms floor |
| `11_C51/` | C51 selective backward variants: B1/B2/B3, gates A/B/C/D, 5K and 30K results |
| `12_C49_ATTRIBUTE_DECOUPLED/` | C49 attribute decoupling, utility concentration & signed utility correction, 1.13% ceiling |
| `13_CANDIDATE_C/` | Candidate C: C++/CUDA implementation, X-ray (conical) formulation, R3/R3.1/R4 certificates |
| `14_TRAINABLE_HIGS/` | Trainable HiGS: architecture, correctness suite, dynamic topology, T_HIGS_REGRESSION, 13-scene results |
| `15_NEGATIVE_RESULTS/` | NEGATIVE_RESULTS.md — every falsified/dropped hypothesis with evidence |
| `16_REPRODUCIBILITY/` | REPRODUCIBILITY.md — environment, commands, seeds, dataset hashes, checkpoints |
| `17_RAW_DATA_INDEX/` | Pointer file to the raw data archives (large files excluded) |

## 5. Suggested report structure (not mandatory — but all facts must trace here)

1. Introduction & motivation
2. Benchmark methodology (protocol, datasets, cohorts, gates)
3. Reference baseline establishment and validation
4. Forward-pass optimization studies (tile geometry, sort)
5. Backward-pass optimization studies (profiling, C25/C51)
6. Attribute-decoupled computation (C49)
7. Candidate C: certificate-based computation
8. Trainable HiGS: implementation and end-to-end results
9. Negative results & falsified hypotheses
10. Conclusions, limitations, and future work

---

*Generated by the evidence-collection agent. Verify important claims against the
cited source files before putting them in the report.*
