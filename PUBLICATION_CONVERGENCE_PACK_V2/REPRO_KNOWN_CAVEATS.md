# REPRODUCIBILITY — Known Caveats

**Generated:** 2026-10-02  |  Warnings that MUST accompany any reproduction.

1. **PBR0 M4 is NOT production training.** PBR0 is a cross-pack reconciliation + gradient noise-floor audit (measurement only, fail-closed). Its native full-step 1.1988x is a reconciled renderer-step figure, not a 30K training result.

2. **Combined residual-derived optimizer timing is forbidden.** Never infer pure optimizer cost from residual subtraction of independent loops (PUB0 Amdahl buckets are directly measured, not residual-derived). Adam optimizer ~30% of step, O(N), untouched by ALL current policies.

3. **DFC padding-thread illegal-read fix.** A padding-thread illegal-read existed in an earlier DFC variant; the SAFE production implementation must be used. The illegal-read variant is a defect and must not be reported as a result.

4. **TFC1 vs TFC2 implementation equivalence is ESTABLISHED by PBR0.** TFC2 reuses the identical geometry code path and adds only the analytic circular superset; it does NOT import or modify TFC1 (byte-identical, independent kernel sha). Equivalence is a PBR0-established fact, not an assumption.

5. **Historical cohort mixing is PROHIBITED.**
   - K50-B1 stage4a (room-only, A100-PCIE-40GB, config-buggy 5K) != stage4b (3-scene, A100-80GB, clean 30K).
   - Renderer-frozen-PLY cohorts (TFC/DFC/CGC/KFC/AGS) != full-training cohorts (K50-B1/VOS/IGF).
   - EPIC-05 A100-SXM4-80GB != A100-PCIE-40GB (stage4a).
   - K50-B1 is HISTORICAL multi-GPU (baseline gpu_id vs policy gpu_id differ) -> per-run speedups hardware-confounded; only the same-run controlled microbenchmark is clean.

6. **Legacy 1.3564x (TFC) is footprint-stage fwd+bwd**, NOT full-E2E/fwd+bwd of a training step and NOT total training. Reconciled native full-step (PBR0) = 1.1988x; current-cohort full-step (PUB0) = 1.0317x.

7. **No total 30K production training wall-clock exists** for any method. K50-B1 per-iter only (implied +8.2/+24.4/+28.9%); TFC/DFC renderer-local. Do not fabricate a total-time field.

8. **Single-hw cohort.** EPIC-05 A100 remote-only; no cross-GPU/cross-vendor validation. K50-B1 single-seed, no multi-seed CI, LPIPS unmeasured.

9. **DFC1-R1 saved-T is a no-op at t>=1e-5** and delivers no standalone speedup; its value is confined to the full-render/low-T regime (backward correctness only). SUPPORTING, not a headline.

10. **DFC2 DOES_NOT_GENERALIZE_UNIVERSALLY** — no single fixed threshold meets quality<=0.05 dB AND workload>=10% on all 6 held-out scenes. DFC_FC0 train_dfc.py is an unverified scaffold (200-iter smoke REQUIRED before 30K).
