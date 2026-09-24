# AccuTile A0.5 forensic correction

Status: `COMPLETE — GENUINE UNSAFE CULLING`.

The 26 original candidate cases were re-evaluated with a dedicated CUDA float32
kernel using the Reference V1 fragment equations exactly:

```text
px = j + 0.5; py = i + 0.5
sigma = 0.5 * (A*dx^2 + 2*B*dx*dy + C*dy^2)
alpha = min(0.999, opacity * __expf(-sigma))
discard iff sigma < 0 or alpha < 1/255
```

No antialias compensation is active in this baseline (`calc_compensations=False`
and the rasterizer consumes the same packed raw opacity tensor). Conic order is
`(A,B,C)` in both the projection output and raster kernel.

| Scene | Original false negatives | CUDA-confirmed | Material | Max ratio in scene |
| --- | ---: | ---: | ---: | ---: |
| room | 6 | 6 | 6 | 16.7180 |
| bicycle | 9 | 9 | 9 | >1.01 |
| garden | 11 | 11 | 11 | >1.01 |

All 26 are `MATERIAL` (`alpha/ALPHA_THRESHOLD > 1.01`), not numerical-borderline
or small-margin cases. The maximum ratio is `16.7179952`. They are therefore
not caused by the prior Python/float64 evaluator. The upstream AccuTile selected
range omits actual Reference V1 fragment-support pixels under frozen raw-opacity
semantics. The full per-case dump is
`experiments/final_sprint/accutile/false_negative_forensics.csv`.

No repair is allowed: the protocol permits one conservative repair only when all
violations are numerical/boundary cases. No renderer integration, performance
benchmark, render parity, gradient check, 13-scene validation, or training run
was performed.
