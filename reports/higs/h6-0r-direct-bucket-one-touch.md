# H6-0R — Direct-Bucketed One-Touch Exact Macro Construction

## Decision

**FORWARD_REOPEN_STRONG.** The direct-bucketed, one-touch CUDA validation prototype is exact on room/cam0, bicycle/cam0, and garden/cam0 and is faster than fresh interleaved B2 F4 on all three. It removes H6-0's generic compact stream and global macro+depth radix sort; no Macro Raster, F5, backward, or autograd code was added.

The pipeline is:

```text
candidate bbox slots → exact AccuTile mask + atomic macro count once
→ macro scan → valid-slot rebucket/scatter → segmented depth sort
→ gather already-written {id, uint32 mask} → batch metadata
```

The exact `accutile_fine_mask_8x4` support, B2 FP32 operations, and R2 integer boundaries are unchanged. A zero mask is the validity sentinel, so no `nonzero`-style compact stream is materialized.

## Structure and tie ordering

| scene | macro entries | fine pairs | missing / extra / duplicate / wrong ID / mask mismatch / non-tie inversion |
| --- | ---: | ---: | --- |
| room | 124,017 | 953,144 | 0 / 0 / 0 / 0 / 0 / 0 |
| bicycle | 311,610 | 1,412,189 | 0 / 0 / 0 / 0 / 0 / 0 |
| garden | 66,682 | 533,928 | 0 / 0 / 0 / 0 / 0 / 0 |

Equal-depth records are separately identified: 82/206/20 compared pairs and 36/32/3 equal-depth ordering inversions for room/bicycle/garden. H3-R2/B2 specifies depth ordering, not a secondary equal-depth key; all non-tie ordering is exact. A secondary ID key was therefore not added merely to manufacture an ordering contract absent from the reference.

## Direct-bucketed stage timing

Protocol: mx A100-PCIE-40GB; 20 warmups, 100 CUDA-event samples, 5 repetitions, interleaved total-F4 comparison. Stage profiling is outside the total timer.

| scene | candidate bbox | exact coverage + count | macro scan | rebucket/scatter | payload segmented sort | batch metadata |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| room | 0.010240 | 0.344064 | 0.027520 | 0.049216 | 0.048128 | 0.026624 |
| bicycle | 0.011264 | 0.312320 | 0.026016 | 0.101408 | 0.096256 | 0.026624 |
| garden | 0.010240 | 0.163840 | 0.027232 | 0.035200 | 0.046080 | 0.025600 |

All values are median milliseconds. The rebucket stage is `0.049216 / 0.101408 / 0.035200 ms`; payload segmented sort, including payload gather but no mask recomputation, is `0.048128 / 0.096256 / 0.046080 ms`.

## End-to-end F4

| scene | B2 F4 | R2 Macro-F4 | H6-0 global | H6-0R direct bucket |
| --- | ---: | ---: | ---: | ---: |
| room | 0.550912 | 0.809984 | 0.748544 | **0.543744** |
| bicycle | 0.665600 | 0.817152 | 0.818176 | **0.619520** |
| garden | 0.423936 | 0.463872 | 0.552960 | **0.344064** |

The global H6-0 path is retained solely as a comparison and was not optimized. H6-0R is below B2 by `0.007168`, `0.046080`, and `0.079872 ms` respectively.

## Forward headroom

| scene | F4 penalty H6-0R − B2 | B2 F5 | penalty / B2 F5 | future F5 required for break-even | speedup required vs B2 F5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| room | -0.007168 ms | 0.6717 ms | -1.07% | ≤0.678868 ms | 0.989× |
| bicycle | -0.046080 ms | 0.9103 ms | -5.06% | ≤0.956380 ms | 0.952× |
| garden | -0.079872 ms | 0.4127 ms | -19.35% | ≤0.492572 ms | 0.838× |

Negative penalty means no future F5 speedup is required for F4+F5 break-even; the future F5 may be correspondingly slower than B2 F5 and still break even. This satisfies the strong gate directly because H6-0R beats B2 F4 in every scene.

## Memory

H6-0R avoids H6-0's compacted-global-record plus global-sort peak, though its candidate record stream remains a bounded temporary.

| scene | R2 Macro peak | H6-0 global peak | H6-0R peak | H6-0R candidate bytes |
| --- | ---: | ---: | ---: | ---: |
| room | 2,488,320 | 8,215,264 | 5,591,556 | 2,248,832 |
| bicycle | 6,241,280 | 20,755,827 | 14,192,404 | 5,254,512 |
| garden | 1,341,440 | 4,454,055 | 3,040,388 | 1,237,424 |

The detailed candidate, count, offset, scatter, segmented-sort, persistent, temporary, and peak breakdown is in [memory.json](/C:/Users/36570/3dgs-renderer-benchmark/artifacts/higs-h6-0r/memory.json).

## Artifacts

All required deliverables are in [artifacts/higs-h6-0r](/C:/Users/36570/3dgs-renderer-benchmark/artifacts/higs-h6-0r), including [stage timings](/C:/Users/36570/3dgs-renderer-benchmark/artifacts/higs-h6-0r/stage_timing.csv), [full timings](/C:/Users/36570/3dgs-renderer-benchmark/artifacts/higs-h6-0r/timing.csv), [structural oracle](/C:/Users/36570/3dgs-renderer-benchmark/artifacts/higs-h6-0r/structural_room.json), [headroom accounting](/C:/Users/36570/3dgs-renderer-benchmark/artifacts/higs-h6-0r/forward_break_even.json), and [provenance](/C:/Users/36570/3dgs-renderer-benchmark/artifacts/higs-h6-0r/provenance.json).
