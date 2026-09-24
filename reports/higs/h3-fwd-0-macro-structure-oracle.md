# H3-FWD-0 Macro-Structure Oracle

Fixture: room/cam0, tile size 16, 2048 long-side. Inputs were freshly produced by B2 F0–F2: FP32 means2d, conics and depths, plus B2 opacity. F3 colors were intentionally not used.

The oracle reproduces official HiGS hierarchy dimensions (8×4 macro tiles, 1024-entry batches, 32-bit fine-tile masks) in a validation-only FP32 geometry implementation. It emits macro entries from the opacity-threshold ellipse and reconstructs fine-tile lists from the corresponding ellipse masks; B2 `isect_tiles` plus sort is the independent reference.

| Metric | Result |
| --- | ---: |
| B2 fine tile–Gaussian pairs | 953,144 |
| HiGS macro entries | 124,150 |
| Reconstructed fine pairs | 953,144 |
| Missing / extra / duplicate pairs | 0 / 0 / 0 |
| Exact tile-order match fraction | 1.0 |
| Pairwise inversions / equal-depth ties | 0 / 0 |
| Mean mask popcount | 7.6774 |
| p50 / p90 / p95 / p99 / max mask popcount | 5 / 18 / 26 / 32 / 32 |

The 7.6774 ratio is representation compression only: reconstructed fine-tile work equals B2 work exactly. It is not a rendering or pixel-compute speedup claim.

Source verification: B2 writes `last_ids[pixel] = cur_idx[pixel]` in `RasterizeToPixels3DGSSerialBatchFwd.cu:293`. It is the tile-local position in the sorted fine-tile list, not a Gaussian ID or global flatten index. A macro path needs an equivalent per-pixel position in the reconstructed fine list and the mapping through macro batch/mask state.

Gate: **EXACT_STRUCTURE_PASS** for the measured room/cam0 structural fixture.

No final macro rasterizer or macro backward was implemented. CUDA-event F4 timing is explicitly not reported: this first oracle is a CPU/Python validation reconstruction and cannot supply a comparable macro-F4 CUDA timing result.

