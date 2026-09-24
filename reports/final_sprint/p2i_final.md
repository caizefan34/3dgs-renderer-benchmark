# P2I final decision

```text
P2I_FINAL_STATUS: COMPLETE

OPPORTUNITY_GATE: DROP

COUNT_PASS_SHARE:
room = 0.201%
bicycle = 0.245%
garden = 0.432%

IMPLEMENTATION:
Not implemented. P0 stopped the candidate before any gsplat CUDA, wrapper, sort,
or rasterizer change.

CORRECTNESS:
P0_TILE_COUNT_CONTROL_PASS; fused-pipeline checks NOT_RUN (no fused path).

TILE_COUNT_MISMATCHES:
room = 0 (max diff 0)
bicycle = 0 (max diff 0)
garden = 0 (max diff 0)

3SCENE_PRE_RASTER_SPEEDUP: NOT_MEASURED (no implementation)
3SCENE_FORWARD_SPEEDUP: NOT_MEASURED (no implementation)
13SCENE_STATUS: NOT_RUN (P0 hard early-stop)
13SCENE_GEOMEAN_FORWARD_SPEEDUP: NOT_APPLICABLE
TRAINING_SANITY: NOT_RUN (P0 hard early-stop)

PRIMARY_MECHANISM:
Count has one thread per visible Gaussian and writes one int32. Emit's
per-intersection loop plus unchanged sort/raster work dominate instead.

PRIMARY_LIMITING_FACTOR:
Removal is at most 0.0297/0.0266/0.0225 ms before non-negative fusion overhead.

FINAL_VERDICT: DROP
```

Evidence: `experiments/final_sprint/p2i/raw/p2i_opportunity_gate.json`, CSV,
environment fingerprint, commands, and log.
