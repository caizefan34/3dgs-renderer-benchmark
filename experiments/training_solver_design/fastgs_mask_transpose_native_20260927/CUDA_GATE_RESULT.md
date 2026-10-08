# Isolated native CUDA gate result

Date: 2026-09-27. Candidate stayed in `fastgs_mask_transpose_native_trial/` and remote clone `/root/codex_fastgs_mask_transpose_trial_20260927`; `public_baselines/fastgs` was not modified. The frozen protocol is in `CUDA_GATE.md`.

## Verdict

**NO-GO for this candidate on the fixed-state performance gate; correctness parity passes.** The four-view native-vs-candidate forward/loss/all-gradient comparison passes every registered threshold. On an idle GPU3, however, all three paired fixed-state forward+loss+backward block medians were slower (candidate/native ratios 1.244, 1.291, 1.238); this fails the required `<=0.95` ratio. A separate forward-only render probe also measured 4.7%, 5.1%, and 7.1% overhead. Do not infer end-to-end training impact: these probes omit optimizer updates, dataloading, topology changes, and long-run convergence/quality.

## Frozen parity inputs and gates

- FastGS base commit: `44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f`.
- Room checkpoint `room_big_capture4100`, iteration 4100; PLY SHA256 `64160970e5ea580aea4f79278a8fbe334d8d81ef6e304a903626843cc48f85bc`.
- Training cameras, in order and repeated in every timing block: `DSCF4668`, `DSCF4770`, `DSCF4873`, `DSCF4976`; each effective width 1600. No test views.
- Native renderer and loss were frozen: author forward and `0.8 * L1 + 0.2 * (1 - fused_ssim)`.
- Parity thresholds: per-view RGB max abs `<=1e-4`; scalar loss abs `<=1e-6`; all six parameter gradient groups and full screen-space `[N,4]` group require relative L2 `<=1e-3` or max abs `<=1e-7`. Signed screen xy and per-contribution mean absgrad xy are also checked separately.
- `parity.json` reports `all_pass=true`. Every view has RGB and loss errors exactly zero. Per-view screen full-four rel-L2 is `6.85e-8` to `8.22e-8`; absgrad xy rel-L2 is `6.56e-8` to `7.98e-8`. Every parameter group passes. This confirms the isolated candidate against the frozen native extension for these four views only.

## Fixed-state timing

The probe clears parameter gradients and runs render + original loss + backward; it has no optimizer update. Each process did 16 warmup blocks followed by five measured blocks of 128 iterations. Camera order starts at `DSCF4668` and cycles in the frozen order. Candidate timings include allocation/initialization and all accepted-mask writes. Each pair was native then candidate, three pairs total. Host load was around 6.3–6.7 on 96 logical CPUs. GPU3 was confirmed at 0 MiB / 0% before and after this clean rerun; GPU0/1 had external allocations but showed 0% utilization around the GPU3 run. GPU2 had an unrelated publication-trainer process, so it was not used for the accepted timing result.

| Pair | Native median seconds / 128 | Candidate median seconds / 128 | Candidate/native |
|---:|---:|---:|---:|
| 1 | 0.6936 | 0.8628 | 1.244 |
| 2 | 0.6954 | 0.8978 | 1.291 |
| 3 | 0.6943 | 0.8596 | 1.238 |

All three pair medians favor native, and candidate overhead is consistent. The earlier GPU2 timing files (`timing_1..3_*.json`) are retained as invalid diagnostics only: an unrelated GPU2 process was active and GPU0/1 also became busy, producing large drift; they are excluded from the verdict.

A forward-only probe, also on GPU3 with the same state, camera cycle, 16 warmups, and five 128-render blocks per process, measures only `render_fastgs`. Candidate accepted-mask clearing and writes are included. Its paired median ratios were 1.047, 1.051, and 1.071. These are an informational component measurement; the fixed-state F+loss+B result is the registered performance gate.

## Resource/source record

- Accepted mask storage is `bucket_sum * 256 * sizeof(uint32_t)` bytes: four extra bytes per pixel per bucket, zeroed each forward. The candidate also performs a global `accepted_mask |= bit` store for each accepted contribution. Exact total accepted stores and memset duration were not separately instrumented; both costs are included in the forward-only and F+loss+B measurements.
- Candidate transpose CTA shared storage is two `32*64` float arrays plus 64 uint masks = 16,640 bytes per CTA (16.25 KiB). Register count/occupancy was not captured. The backward path launches the candidate transpose kernel; static logs do not include profiler counters.
- Build succeeded with CUDA 12.8 / PyTorch 2.9.1+cu128 in the isolated venv. Full build log is `build_mask.log`.
- Source SHA256 (public baseline -> isolated candidate):
  - `forward.cu`: `F32394FB3F1A201575C2F5FDBE89FD2960395F676433904AB421AB7BEAE6F1DC` -> `458E3C9FF5FF4077CFBB4981BDDADD3BD6ADED6C987EF2A8B17CDB88AEA14153`
  - `backward.cu`: `851041054CC0DEDD7A2B049FAE6E10F2A486D1DF5AC2978B4DF85C1F552BC90E` -> `5EB407934DF2B33BA77B8EDC01169F252081F667F3FA161F886B44C84CD76B78`
  - `rasterizer_impl.cu`: `9990134F570ECBA9F11692C8CCDCAA2D20D98E679030F5F0508AFEB160E60234` -> `495021D2301FD8F479E41D454EE7FF64D9509392172EC0DA395D609BE26F69F4`
  - Candidate header hashes and probe hashes are recorded in `source_hashes.json`.

Raw inputs/results are retained alongside this report: `parity.json`, `timing_gpu3_*_{native,candidate}.json`, `forward_gpu3_*_{native,candidate}.json`, and `build_mask.log`. No GPU test process remains running. No 30K run or production change was made.
