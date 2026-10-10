# R17 Final Report — parameter validity & camera protocol adjudication

Status: **PARTIAL_MISSING_p3_p4_p5**

## Verdicts

- **OPACITY_INPUT_CONTRACT: VIOLATED**
- **SCALE_INPUT_CONTRACT: VIOLATED**
- **ROTATION_INPUT_CONTRACT: VIOLATED**
- **CAMERA_PROTOCOL: SYNTHETIC**
- **BENCHMARK_VALIDITY: UNKNOWN**
- **HIGS_MEMORY_SCALABILITY: UNKNOWN**

## P0 — R16 arithmetic audit

```json
{
  "claimed_reduction_r16_prose": 360251,
  "recomputed_raw_minus_sigmoid": 80,
  "recomputed_elementwise_abs_sum": 80,
  "sigmoid_vs_threshold_identical": true,
  "claimed_reduction_reproducible": false,
  "r15_raw_exact_match": null,
  "branch_coverage": {
    "status": "NO_ALPHA_OR_ROWS"
  }
}
```

## NOT_EXECUTED

Missing: p3, p4, p5

## R3–R16 conclusion disposition

```json
{
  "status": "NOT_EXECUTED_P3_MISSING"
}
```

## Harness source evidence

```json
{
  "r3": {
    "file": "C:\\Users\\36570\\3dgs-renderer-benchmark\\R17_EXEC_PACK\\TRAINABLE_HIGS_PARAMETER_VALIDITY_R17_MX\\reference\\run_r3_benchmark.py",
    "sha256": "c78b029c68378486dbe61a8bd5187a00df378ed96f6b0e47b39ebd0a64a528ad",
    "applies_sigmoid_to_opacity": false,
    "applies_exp_to_scaling": false,
    "normalizes_quats": true,
    "passes_opacity_directly": true
  },
  "r8": {
    "file": "C:\\Users\\36570\\3dgs-renderer-benchmark\\R17_EXEC_PACK\\TRAINABLE_HIGS_PARAMETER_VALIDITY_R17_MX\\reference\\R8_WORKING_ENTRYPOINT.py",
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
  "benchmark_validity": "Cells A and D must both pass before validity can be judged."
}
```
