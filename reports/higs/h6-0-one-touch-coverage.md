# H6-0 — One-Touch Exact Coverage

## Result

**Gate: ONE_TOUCH_WEAK.** The isolated CUDA prototype is structurally exact on `room/cam0`, `bicycle/cam0`, and `garden/cam0`, and it evaluates the immutable R2 AccuTile macro predicate once rather than once in COUNT and again in FILL. It does not reach B2 F4: compaction and global record ordering consume the recovered geometric work.

The prototype is deliberately validation-only: it produces `macro_offsets`, `macro_sorted_ids`, sorted `uint32` masks, and `macro_batch_offsets`. It changes neither B2/R2 coverage semantics nor any raster, F5, backward, or autograd code.

## Exact work removed

The R2 COUNT and FILL each evaluate the same exact `accutile_fine_mask_8x4` predicate over the coarse macro-bbox candidates. The one-touch stream invokes it once and writes `{macro_id, depth_key, gaussian_id, fine_mask, valid}` immediately.

| scene | candidates | R2 COUNT+FILL predicate calls | one-touch calls | duplicate calls removed | R2 calls including post-sort mask recompute |
| --- | ---: | ---: | ---: | ---: | ---: |
| room | 140,552 | 281,104 | 140,552 | 140,552 | 405,121 |
| bicycle | 328,407 | 656,814 | 328,407 | 328,407 | 968,424 |
| garden | 77,339 | 154,678 | 77,339 | 77,339 | 221,360 |

R2 CUDA-event stage medians for COUNT/FILL are room `0.346112/0.356160 ms`, bicycle `0.319488/0.336128 ms`, and garden `0.166912/0.192288 ms`. One-touch exact emission is `0.316416/0.259072/0.151552 ms`. Thus the duplicated geometric rule itself is removed, but the overall construction is not faster enough.

## Capacity and allocation

Strategy A (upper-bound global stream) was implemented because its capacity is empirical and tight: candidate/exact is `1.1333×`, `1.0539×`, and `1.1598×` for room, bicycle, and garden. Its raw streams (records, validity, and per-G count/prefix) are `2,748,648`, `7,035,119`, and `1,510,627` bytes.

| scene | visible G | exact macro entries | candidates/G mean / p50 / p90 / p95 / p99 / max |
| --- | ---: | ---: | --- |
| room | 44,908 | 124,017 | 3.130 / 2 / 6 / 8 / 16 / 352 |
| bicycle | 181,525 | 311,610 | 1.809 / 1 / 3 / 4 / 8 / 330 |
| garden | 24,483 | 66,682 | 3.159 / 2 / 6 / 9 / 16 / 132 |

The four requested allocation designs, including atomic/synchronization/sort trade-offs and their measurement status, are in [allocation_strategy_analysis.json](/C:/Users/36570/3dgs-renderer-benchmark/artifacts/higs-h6-0/allocation_strategy_analysis.json). B/C/D were not selected: they have no measured benefit here, while A's upper bound is already practical.

## Sort comparison

The comparison starts from the *same already compacted exact record stream* and excludes coverage generation.

| scene | global radix macro+depth | macro offset + segmented depth | selected sort-only result |
| --- | ---: | ---: | --- |
| room | 0.299008 ms | 0.157696 ms | segmented |
| bicycle | 0.427008 ms | 0.285696 ms | segmented |
| garden | 0.247808 ms | 0.142336 ms | segmented |

The global permutation was used for this structural prototype because it carries masks directly with the sorted record. The segmented comparator is clearly faster, but a payload-preserving segmented integration was not promoted as an end-to-end claim in this task; it must be integrated and re-timed before claiming its sort-only advantage. Workspaces are recorded in [sort_comparison.csv](/C:/Users/36570/3dgs-renderer-benchmark/artifacts/higs-h6-0/sort_comparison.csv).

## Exactness

Every required structural field is zero for all three scenes: missing, extra, duplicate, wrong ID, mask mismatch, and semantic non-tie inversion. Equal-depth permutation inversions are also zero in this implementation.

| scene | macro entries | reconstructed fine pairs | missing / extra / duplicate / wrong ID / mask mismatch / non-tie inversion |
| --- | ---: | ---: | --- |
| room | 124,017 | 953,144 | 0 / 0 / 0 / 0 / 0 / 0 |
| bicycle | 311,610 | 1,412,189 | 0 / 0 / 0 / 0 / 0 / 0 |
| garden | 66,682 | 533,928 | 0 / 0 / 0 / 0 / 0 / 0 |

## Timing gate

Protocol: 20 warmups, 100 CUDA-event measurements, five repetitions, interleaved B2/R2/one-touch. All values are milliseconds.

| scene | method | median | mean | p10 | p90 | std |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| room | B2 F4 | 0.544768 | 0.545362 | 0.543744 | 0.546816 | 0.002777 |
| room | R2 Macro-F4 | 0.814080 | 0.815446 | 0.813056 | 0.817152 | 0.013579 |
| room | One-Touch | 0.762880 | 0.763095 | 0.760832 | 0.765952 | 0.005738 |
| bicycle | B2 F4 | 0.681984 | 0.720740 | 0.664576 | 0.714854 | 0.131136 |
| bicycle | R2 Macro-F4 | 0.824320 | 0.876173 | 0.817152 | 0.830464 | 0.176855 |
| bicycle | One-Touch | 0.851968 | 0.900661 | 0.848896 | 0.861184 | 0.161974 |
| garden | B2 F4 | 0.422912 | 0.477362 | 0.420864 | 0.703488 | 0.110912 |
| garden | R2 Macro-F4 | 0.462848 | 0.531347 | 0.460800 | 0.816128 | 0.139446 |
| garden | One-Touch | 0.544768 | 0.607640 | 0.542720 | 0.870400 | 0.128780 |

One-touch is 6.3% faster than R2 on room, but 3.4% and 17.7% slower on bicycle and garden, respectively. It beats B2 on zero of three scenes. The core reason is visible in its stages: compact is approximately `0.09–0.10 ms`, and global sort plus offsets is `0.24–0.44 ms`.

## Memory

| scene | B2 peak | R2 Macro peak | One-Touch peak |
| --- | ---: | ---: | ---: |
| room | 23,683,072 | 2,488,320 | 8,215,264 |
| bicycle | 36,462,080 | 6,241,280 | 20,755,827 |
| garden | 13,265,920 | 1,341,440 | 4,454,055 |

One-touch is bounded and remains below B2's peak, but uses 3.3× R2 Macro's peak because it retains the raw bound stream, compacted records, and global-sort temporary state at once. Full persistent/temporary/peak values are in [memory.json](/C:/Users/36570/3dgs-renderer-benchmark/artifacts/higs-h6-0/memory.json).

## Artifacts

All H6-0 deliverables are in [artifacts/higs-h6-0](/C:/Users/36570/3dgs-renderer-benchmark/artifacts/higs-h6-0), with provenance in [provenance.json](/C:/Users/36570/3dgs-renderer-benchmark/artifacts/higs-h6-0/provenance.json).
