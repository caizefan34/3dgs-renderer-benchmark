# H3-FWD-1A-R CUDA Macro-F4 validation

The isolated SM80 build passed. The binary SHA256 is `78ace12b294d274209b4711b95da578ade4ac2ed213cbe25c372cb5e86d7c890`; the patch SHA256 is `e5f085eb076d46ebd6c62d546d852d871c58adc602bf9d432b783b9062bc998d`.

`HIGS_TRAIN_F4` is validated as `baseline|macro_fp32`, defaults to `baseline`, and rejects any other value. Its scope is validation-only: Macro-F4 output is captured for comparison while F5 remains the B2 path.

| Scene | B2 pairs | macro entries | reconstructed pairs | missing / extra / duplicate / wrong ID |
| --- | ---: | ---: | ---: | --- |
| room/cam0 | 953,144 | 124,150 | 953,144 | 0 / 0 / 0 / 0 |
| bicycle/cam0 | 1,412,189 | 311,675 | 1,412,189 | 1 / 1 / 0 / 2 |
| garden/cam0 | 533,928 | 66,724 | 533,928 | 0 / 0 / 0 / 0 |

All observed order inversions are between equal FP32 depth keys: room 73/73, bicycle 82/101, garden 3/3 (inversions/ties). The non-tie inversion count is zero. Bicycle does not meet the exact membership gate; it also differs from the CPU oracle by one macro entry and one reconstructed pair. The discrepancy is retained, not repaired or normalized.

Room CUDA-event timing (20 warmup, five interleaved 100-sample repetitions): B2 F4 median 0.557056 ms; Macro-F4 median 0.365568 ms. Macro stages: count 0.114688 ms, scan/prefix 0.024464 ms, fill 0.145376 ms, segmented sort 0.043008 ms, batch metadata 0.026624 ms. By the requested 5% band the timing classification is `MACRO_F4_FASTER`.

Room Macro-F4 persistent production state is 499,424 B; estimated temporary state is 1,990,624 B; validation masks add 496,600 B and are excluded. The equivalent B2 persistent result state is 11,481,760 B and its fresh-process allocator workspace estimate is 12,201,312 B.

The later `last_ids` mapping is feasible by retaining macro offsets, sorted IDs, macro batch offsets, and the entry masks (996,024 B on room including masks). It maps a fine tile through its macro segment, 1024-entry batch, 32-entry mini-batch, mask membership, then rank among preceding entries for that tile.

Classification: `STRUCTURE_REPAIR`. `PROMOTE_TO_MACRO_RASTER` is not met because the bicycle exact-pair gate fails. No macro rasterizer or backward code was changed; B2 F0–F3, baseline F4/F5, backward, and the authoritative B2 patch remain unchanged.
