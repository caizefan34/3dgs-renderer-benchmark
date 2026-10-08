# FastGS residual–native-hit overlap screen

**2026-09-28 data-provenance correction:** The garden@4100/30K checkpoint rows were produced on the EPIC-05 garden sparse model later identified as a corrupt flat 1,839,236-point cloud. They cannot establish an official-garden opportunity gate; quarantine garden-specific conclusions until rerun on verified official sparse data. The room rows remain a separate diagnostic. See `../P16_GARDEN_POINT_ORDER_AUDIT_20260928.md`.

2026-09-27. The preregistered same-region opportunity gate **failed in all four states**. The highest-training-L1-residual 25% of pixels consumed only 24.94%–26.07% of native accepted Gaussian–pixel hits, below the fixed 40% requirement. Stop the high-residual-concentrated-work direction before an intervention or 30K run. This result does not exclude a method that improves quality in one region while saving work elsewhere.

| State | Gaussians | Top-25% L1 residual share | All accepted hits, four views | Top-25% accepted hits | Pooled hit share `f25` | Gate |
|---|---:|---:|---:|---:|---:|---|
| room@4100 | 651,547 | 59.34% | 338,644,872 | 88,268,660 | 26.0653% | fail |
| room@30K | 566,856 | 59.89% | 293,996,253 | 75,435,769 | 25.6588% | fail |
| garden@4100 | 4,035,481 | 54.64% | 337,526,978 | 84,188,783 | 24.9428% | fail |
| garden@30K | 6,840,333 | 63.91% | 289,659,344 | 74,795,119 | 25.8218% | fail |

The 16 per-view `f25` values range from 23.3311% to 26.8267%; per-view top-10% hit shares range from 9.5703% to 11.1333%. High L1 error is concentrated, but native accepted-hit work is not concentrated in the same pixels. Room top-25% hits from Gaussian IDs present in at least two selected views were 87.94%/87.80% at 4100/30K; garden was 31.58%/26.86%. These are only cross-view shared-ID proxies, not evidence of common surface error or a repair mechanism.

All eight distinct images were the protocol-selected `--eval` **training** images, reused at 4100 and 30K; no formal held-out image was opened. The effective width was 1600. The unchanged author renderer produced exactly the same RGB for original and all-ones/top-25/top-10 counter renders in every view (maximum absolute error 0). Count-vector length, subset monotonicity, and top-10 nesting checks passed; all 16 views and four states completed without failure. The probe used the pinned FastGS commit `44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f`, `train.py` SHA-256 `1fd700c341d109f13f95ff3a3cdf1328445f27909119e7eacadb31913e34c963`, and previously idle physical A100 GPU 2. No author code or model parameter was changed.

The measurement is spatial overlap between per-pixel training-image L1 residual and native accepted-hit counts. L1 is not a decomposition of the fused-SSIM training loss; accepted hits are not backward milliseconds or complete training wall time. The four states and selected cameras are development diagnostics, not an unseen-view evaluation. No algorithm, training speed, or quality claim follows from this screen.

## Audit artifacts

- Frozen protocol: `../FASTGS_RESIDUAL_COST_OVERLAP_PROTOCOL.md`, SHA-256 `af034979f495ee94622e0ca8c1fbfc4edc06bad303c6575447c88f86bf89eabe`.
- Probe: `../fastgs_residual_cost_overlap_probe.py`, SHA-256 `8f43bc60880c1efdb7d798dca4dc7c963cbb912d16070c3b5278f78f07fef86f`.
- Raw [`result.json`](result.json): SHA-256 `94bb87fda43b482983168a462f23f35b2407a33a0a45c524824e4b1df9c16b19`. It retains all 16 per-view rows, model hashes, counts, residual sums, camera splits and the fixed gate decision.
- Raw [`run.log`](run.log): SHA-256 `64a4dbcb6a729fd25d8aa6ea6ab3b977a85620368e45f057ac6cc3ccf9a8f846`. The author camera loader reported its standard 1600-width downscale.
- Remote originals: `/root/fastgs_residual_cost_overlap_result.json` and `/root/fastgs_residual_cost_overlap_probe_20260927.log` on EPIC-05. Remote and local JSON hashes matched after copy. The probe process had exited and GPU 2 had no compute process at the final check.
