# AccuTile / SNUGBOX final decision

```text
ACCUTILE_FINAL_STATUS: COMPLETE

OPPORTUNITY_GATE: PASS

3SCENE_N_ISECTS_REDUCTION:
room = 51.58%
bicycle = 50.56%
garden = 44.49%

IMPLEMENTATION:
Measurement-only upstream evaluator. No Reference V1 backport was made.

CORRECTNESS: FAIL (26 CUDA-confirmed MATERIAL false negatives)
RENDER_PARITY: NOT_RUN
GRADIENT_PARITY: NOT_RUN
3SCENE_EMIT_SPEEDUP: NOT_RUN
3SCENE_SORT_SPEEDUP: NOT_RUN
3SCENE_FORWARD_SPEEDUP: NOT_RUN
MEMORY_REDUCTION: NOT_MEASURED
13SCENE_STATUS: NOT_RUN
13SCENE_GEOMEAN_N_ISECTS_REDUCTION: NOT_APPLICABLE
13SCENE_GEOMEAN_FORWARD_SPEEDUP: NOT_APPLICABLE
TRAINING_SANITY: NOT_RUN

PRIMARY_MECHANISM:
Upstream ellipse-aware localization removes 44.49%–51.58% of AABB pairs, so
emit/sort/memory opportunity is real.

PRIMARY_LIMITING_FACTOR:
Pixel-support false negatives: room=6, bicycle=9, garden=11. CUDA float32
forensics found all 26 MATERIAL; max alpha/threshold ratio=16.7179952.

FINAL_VERDICT: DROP_BEFORE_BACKPORT
```

Existing-method attribution: upstream gsplat / Speedy-Splat. Raw evidence is in
`experiments/final_sprint/accutile/raw`.
