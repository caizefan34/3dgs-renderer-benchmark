# NOVELTY_DELTA_MATRIX — Compute-Densification Co-Design (C51-CD0) Stage 0

Scope: CPU/literature/read-only audit. No new mechanism found that is both (a) non-trivial
and (b) non-isomorphic to published prior art or repo CLOSED laws. Verdict: **NO_NOVELTY**.

## Component map

| proposed mechanism | closest prior art / CLOSED law | published/closed? | substantial exact distinction? | evidence | falsification | status |
|---|---|---|---|---|---|---|
| Gradient-information spend: skip CUDA backward for low-grad Gaussians | C51 Stage4A/B K50-B1 (this repo); Faster-GS gradient approximation | YES (repo) | None — intrinsic speed is only +0.5% (Room), negative on bicycle/garden | Stage5 Table 2/6: C vs A +0.5%/−6.6%/−1.7%; Stage4A microbench 10.2% at K50 but E2E 6.7% | C-vs-A matched-population already <5% gate on all 3 scenes | FALSIFIED_AS_LEVER |
| Gaussian-capacity growth control via densification | FastGS (Ren et al., CVPR2026) multiview-consistent densify/prune | YES (external) | None — "joint gradient+budget densification" reduces to quality/consistency-driven density control | Stage4B: B speedup tracks final-GS reduction 64–70% | FastGS already covers multiview density control; adding a raster-cost signal has no measured headroom | NOVELTY_THREAT |
| Coupled "birth budget" reducing future raster work | ReBudgetGS (CLOSED row 9, N1_SMALL_BUT_ACTIONABLE); LAW-D0 dead-work; VTS0 (row 24, VTS0_DROP) | YES (repo CLOSED) | None — explicit compute-aware birth budget ≈ reallocation/dead-work-removal | CLOSED_DIRECTIONS rows 9,14,24 | ReBudget actionable-but-small; VTS0 proved topology-skip headroom ≈ 0 | CLOSED_OVERLAP |
| Optimizer-side gradient-spend to save time | LAW-OPT-FRAC (optimizer 5.94–14% of wall); VOS/IGF (DROP) | YES (repo CLOSED) | None — Amdahl-bound <1.05x | amps_gs verdict f_opt 0.059–0.14; even all-BF16 caps E2E at 1.017–1.042x | Measured strong-baseline optimizer fraction <0.30 | FALSIFIED_AS_LEVER |
| Future-raster-workload cost model driving capacity | TFC (footprint tightening) is the ONLY certified forward lever; raster cost ∝ intersections | PARTIAL | Forward pipeline is the dominant lever, not backward/optimizer; TFC already 1.38x intersect / PBR0 1.1988x native | PAPER0 matrix TFC/K50 rows | Backward-side (this line) is the minority of step cost; no backward cost model certified | NOT_THE_LEVER |

## Independent-mechanism finding
No non-trivial independent mechanism survives. The co-design hypothesis = (skip-backward) + (density
control), both already covered; the coupling itself maps to VTS0 / ReBudgetGS CLOSED rows. Per protocol
§0C: "未发现一个非平凡的独立机制，就结束并判 P0_NO_NOVELTY."

## Prior-art references
- FastGS (Ren et al., CVPR 2026): multiview consistency densification/pruning — covers capacity control.
- Faster-GS (Hahlbohm et al., CVPR 2026): gradient approximation + efficient fused training — covers gradient spend.
- Repo CLOSED_DIRECTIONS_20260930.md: VTS0, ReBudgetGS, VOS, IGF, LAW-OPT-FRAC, LAW-D0.
