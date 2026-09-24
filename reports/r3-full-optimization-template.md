# R3 Full Measurement Report — Certificate Tightness After Vectorization

## 1. Execution status
- Window 5000: 30/30 iterations completed
- Window 15000: 30/30 iterations completed
- Window 30000: 30/30 iterations completed
- Commit: e494458
- Python env: anysplat (GPU: see remote log)

## 2. Certificate validity (correctness gate)
- All candidate bounds: finite, non-negative, and satisfy B >= ||g|| for every Gaussian (0 violations)
- SPD-disabled count: 0

## 3. Tightness statistics (distribution of ||g|| / B)
For each family (color, opacity, mean2d, conic), per-window per-iteration:
- min / median / p75 / p90 / p99 / max of ||g||/B

## 4. Budget 消费分析 (0.95 / 0.9 / 0.8 / 0.5 / 0.3)
- Naive per-family budget coverage vs joint budget coverage
- Exact-zero (sigma_min) removal fraction
- Joint-skip pairs (4-family) vs 3-family and 2-family subtype coverage

## 5. Complexity accounting
- Visible Gaussians per image
- Per-Gaussian workload (128 pixel queries per Gaussian per image at tile 32x32)

## 6. Verification & artifacts
- Raw JSONs: results/reference_v1/r3/{5000,15000,30000}/
- Combined: results/reference_v1/r3/aggregated_summary.json
- Decision gate log: results/reference_v1/r3/decision_gate.json
