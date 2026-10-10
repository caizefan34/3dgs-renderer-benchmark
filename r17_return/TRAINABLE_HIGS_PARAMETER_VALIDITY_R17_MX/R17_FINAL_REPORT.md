# R17 Final Report — parameter validity & camera protocol adjudication

Status: **EXECUTED**

## Verdicts

- **OPACITY_INPUT_CONTRACT: VIOLATED**
- **SCALE_INPUT_CONTRACT: VIOLATED**
- **ROTATION_INPUT_CONTRACT: VIOLATED**
- **CAMERA_PROTOCOL: SYNTHETIC**
- **BENCHMARK_VALIDITY: INVALIDATED**
- **HIGS_MEMORY_SCALABILITY: PASS**

## P0 — R16 arithmetic audit

```json
{
  "claimed_reduction_r16_prose": 360251,
  "recomputed_raw_minus_sigmoid": 351319,
  "recomputed_elementwise_abs_sum": 360251,
  "sigmoid_vs_threshold_identical": true,
  "claimed_reduction_reproducible": false,
  "r15_raw_exact_match": true,
  "branch_coverage": {
    "alpha_threshold": 0.00392156862745098,
    "n": 1000,
    "raw_logit_le_0__logf_of_negative__t_forced_by_fminf_nan": 916,
    "raw_logit_in_(0,alpha]__logf_negative__t_negative": 0,
    "raw_logit_capped_t_equals_1": 84,
    "raw_logit_in_(alpha, alpha*e^0.5)__t_in_(0,1)": 0,
    "activated_opacity_le_alpha__true_skip_branch_coverage": 0,
    "activated_opacity_gt_alpha": 1000,
    "activation_fixes_skip_coverage": false
  }
}
```

## P3 — 580K attribution cells

| cell | fwd_status | bwd_status | sum_counts | n_isects_meta | coverage_frac | count_max | peak_vram_gb | oracle_mismatch | oracle_verdict |
|---|---|---|---|---|---|---|---|---|---|
| A | PASS | PASS | 490387290 | 490387290 | 1.0 | 2040.0 | 12.065439232 | 915 | GEOMETRIC_DISCREPANCY_REQUIRES_CUDA_SEMANTIC_AUDIT |
| B | PASS | PASS | 399549407 | 399549407 | 1.0 | 2040.0 | 9.900014592 | 0 | ALL_COMPARABLE_COUNTS_EQUAL |
| B2 | PASS | PASS | 298577 | 298577 | 1.0 | 2032.0 | 0.334925312 | 905 | GEOMETRIC_DISCREPANCY_REQUIRES_CUDA_SEMANTIC_AUDIT |
| B3 | PASS | PASS | 490387290 | 490387290 | 1.0 | 2040.0 | 12.079341056 | 915 | GEOMETRIC_DISCREPANCY_REQUIRES_CUDA_SEMANTIC_AUDIT |
| C | PASS | PASS | 160081 | 160081 | 1.0 | 389.0 | 0.334101504 | 0 | ALL_COMPARABLE_COUNTS_EQUAL |
| C2 | PASS | PASS | 705992313 | 705992313 | 1.0 | 2040.0 | 17.284425728 | 910 | GEOMETRIC_DISCREPANCY_REQUIRES_CUDA_SEMANTIC_AUDIT |
| D | PASS | PASS | 545285 | 545285 | 1.0 | 417.0 | 0.378532864 | 0 | ALL_COMPARABLE_COUNTS_EQUAL |
| D2 | PASS | PASS | 586492927 | 586492927 | 1.0 | 2040.0 | 14.419236864 | 0 | ALL_COMPARABLE_COUNTS_EQUAL |

Attribution deltas vs cell A:

```json
{
  "B": {
    "sum_counts": 399549407.0,
    "delta_vs_A_pct": -18.523702561703832,
    "fwd_status": "PASS",
    "oracle_mismatch_pairs": "0"
  },
  "B2": {
    "sum_counts": 298577.0,
    "delta_vs_A_pct": -99.9391140418831,
    "fwd_status": "PASS",
    "oracle_mismatch_pairs": "905"
  },
  "B3": {
    "sum_counts": 490387290.0,
    "delta_vs_A_pct": 0.0,
    "fwd_status": "PASS",
    "oracle_mismatch_pairs": "915"
  },
  "C": {
    "sum_counts": 160081.0,
    "delta_vs_A_pct": -99.96735620941563,
    "fwd_status": "PASS",
    "oracle_mismatch_pairs": "0"
  },
  "C2": {
    "sum_counts": 705992313.0,
    "delta_vs_A_pct": 43.96627469688295,
    "fwd_status": "PASS",
    "oracle_mismatch_pairs": "910"
  },
  "D": {
    "sum_counts": 545285.0,
    "delta_vs_A_pct": -99.88880523392032,
    "fwd_status": "PASS",
    "oracle_mismatch_pairs": "0"
  },
  "D2": {
    "sum_counts": 586492927.0,
    "delta_vs_A_pct": 19.597905361698913,
    "fwd_status": "PASS",
    "oracle_mismatch_pairs": "0"
  }
}
```

## P4 — matched benchmark (activated state, real cameras)

| protocol | backend | fwd_med_ms | bwd_med_ms | fb_med_ms | peak_gb | n_isects | status |
|---|---|---|---|---|---|---|---|
| B1 | std_ll | 2.499071955680847 | 3.057663917541504 | 5.556735873222351 | 0.495011328 |  | PASS |
| B1 | higs_native | 2.6536959409713745 | 2.8262399435043335 | 5.479935884475708 | 0.532861952 | 545285 | PASS |
| B4 | std_ll | 4.0273919105529785 | 8.80947208404541 | 12.836863994598389 | 0.808638464 |  | PASS |
| B4 | higs_native | 4.362239837646484 | 6.578175783157349 | 10.940415620803833 | 0.88796416 | 2092255 | PASS |

## P5 — 4.41M safety gate

```json
{
  "available": true,
  "raw_identity_b1": {
    "status": "ESTIMATED_GATE_SKIPPED_FULL",
    "fwd_status": "SAFE_ABORT_estimate_only",
    "bwd_status": "NOT_RUN",
    "peak_vram_gb": 2.090044416,
    "estimate": 3732527759.1795,
    "vs_r8": {
      "r8_requested_bytes": 119485990174.72,
      "r17_projected_bytes_conservative": 134370999330.462,
      "ratio_r17_over_r8": 1.124575350917511
    }
  },
  "raw_identity_b4": {
    "status": "ESTIMATED_GATE_SKIPPED_FULL",
    "fwd_status": "SAFE_ABORT_estimate_only",
    "bwd_status": "NOT_RUN",
    "peak_vram_gb": 3.723114496,
    "estimate": 14930111036.718,
    "vs_r8": {
      "r8_requested_bytes": 119485990174.72,
      "r17_projected_bytes_conservative": 537483997321.848,
      "ratio_r17_over_r8": 4.498301403670044
    }
  },
  "activated_real_b1": {
    "status": "FULL_EXECUTED",
    "fwd_status": "PASS",
    "bwd_status": "PASS",
    "peak_vram_gb": 3.314783744,
    "estimate": 3913049.6999999997,
    "vs_r8": {
      "r8_requested_bytes": 119485990174.72,
      "r17_projected_bytes_conservative": 140869789.2,
      "ratio_r17_over_r8": 0.001178964906212111
    }
  },
  "activated_real_b4": {
    "status": "FULL_EXECUTED",
    "fwd_status": "PASS",
    "bwd_status": "PASS",
    "peak_vram_gb": 5.084178432,
    "estimate": 14963241.930000002,
    "vs_r8": {
      "r8_requested_bytes": 119485990174.72,
      "r17_projected_bytes_conservative": 538676709.48,
      "ratio_r17_over_r8": 0.004508283428812974
    }
  }
}
```

## R3–R16 conclusion disposition

```json
{
  "R3_std_vs_higs_timing_claims": "REMEASURE",
  "R3_R8_HiGS_OOM_root_cause": "REINTERPRET: raw logit opacity + log-scale + unnormalized quats + synthetic camera violate the documented contract; see p3_attribution_deltas",
  "R12_intersection_counts_@580K_raw": {
    "r17_cell_A_sum_counts": "490387290",
    "note": "direct comparison basis for the ~490M R12 claim at the same 580K/identity protocol"
  },
  "R16_OPACITY_SEMANTICS_CONFIRMED": {
    "r17_cell_A_vs_B": {
      "A_sum": "490387290",
      "B_sum": "399549407"
    },
    "disposition": "CONFIRMED_AS_HARNESS_EFFECT_ONLY: real, but it quantifies the gap between a violated and an activated opacity input, not a property of the training protocol"
  },
  "R15_oracle_discrepancy": "REINTERPRET: oracle mismatch pairs measured under raw inputs; see per-cell oracle_mismatch_pairs (cell A vs cell D)"
}
```

## Harness source evidence

```json
{
  "r3": {
    "file": "/mnt/storage_pool/liaoyuanjun/TRAINABLE_HIGS_PARAMETER_VALIDITY_R17_MX/reference/run_r3_benchmark.py",
    "sha256": "c78b029c68378486dbe61a8bd5187a00df378ed96f6b0e47b39ebd0a64a528ad",
    "applies_sigmoid_to_opacity": false,
    "applies_exp_to_scaling": false,
    "normalizes_quats": true,
    "passes_opacity_directly": true
  },
  "r8": {
    "file": "/mnt/storage_pool/liaoyuanjun/TRAINABLE_HIGS_PARAMETER_VALIDITY_R17_MX/reference/R8_WORKING_ENTRYPOINT.py",
    "sha256": "9f2bc47de41c2842ba92c73747f8b6a75884fd3d16a6d75ec944671aaf13f5e6",
    "applies_sigmoid_to_opacity": false,
    "applies_exp_to_scaling": false,
    "normalizes_quats": false,
    "passes_opacity_directly": true
  }
}
```

## Rules

```json
{
  "benchmark_validity": "INVALIDATED = the raw/identity harness protocol (used by R3-R16) changes the intersection structure versus the contract protocol by >=20% counts or >=10pp coverage; PARTIAL unchanged-structure but non-protocol cameras; VALIDATED only when both coincide.",
  "memory_scalability": "PASS = contract protocol 4.41M fwd+bwd executed and passed; FAIL = executed full run OOMed; INCONCLUSIVE = the byte projection did not fit free VRAM so the full run was not attempted (raw scenarios are estimate-only by policy)."
}
```
