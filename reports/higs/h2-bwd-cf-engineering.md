# H2-BWD-CF engineering record

## Scope and source identity

- Base: `77ab983ffe43420b2131669cb35776b883ca4c3c`.
- Authoritative B2 patch SHA256: `74e5d8b3b6273b9446ec0551ce91409783e2aa935c8d8e354b4099341390c84c`.
- H2 patch: [`patches/higs-h2-bwd-cf.patch`](../../patches/higs-h2-bwd-cf.patch), SHA256 `d001eac2d6908126103ead7c060f896cfc569989ac90d264236a97edc68137ff`.
- The original `higs_blend_bwd_px_kernel` body was not modified. The default selector value is `baseline`.
- Selector: `HIGS_BWD_CF_VARIANT=baseline|sigma_gate|scalar_adjoint|uv_reuse|combined`; unsupported values fail closed with `TORCH_CHECK`. Experimental variants require the frozen primary `HIGS_PX_RUNTIME=2`; baseline continues to support its existing PX values.

Modified extension files are `HigsNativeBackward.cu`, `HigsNativeBackward.h`, and `ext.cpp`. The authoritative patch was not changed. The reproducible source diff is the H2 patch above.

## Build and binary

The fresh source root was `/tmp/higs_h2_bwd_cf/source`, the normal cache was `/tmp/higs_h2_bwd_cf/cache`, and the optional counter build used `/tmp/higs_h2_bwd_cf/instrumented-cache`. Each harness was run as a fresh Python process and loaded the H2 extension, not the existing B2 experiment extension.

- nvcc: CUDA 12.8 V12.8.93; Torch: `2.9.1+cu128`; CUDA: `12.8`; target: `compute_80,code=sm_80`.
- GPU: NVIDIA A100-PCIE-40GB (SM80, 108 SMs).
- Binary SHA256: `da53009841c5f9a6145dfb01e5ab84de5286f52710c9a7011bcb4b85eb18842c`.

## Compile resources: CDIM=3, PX=2, tile size 16

| variant | registers/thread | spills (store/load) | static / dynamic shared | active blocks/SM | theoretical occupancy |
|---|---:|---:|---:|---:|---:|
| baseline | 67 | 0 / 0 B | 0 / 5120 B | 7 | 43.75% |
| sigma_gate | 67 | 0 / 0 B | 0 / 5120 B | 7 | 43.75% |
| scalar_adjoint | 56 | 0 / 0 B | 0 / 5120 B | 9 | 56.25% |
| uv_reuse | 64 | 0 / 0 B | 0 / 5120 B | 8 | 50.00% |
| combined | 56 | 0 / 0 B | 0 / 5120 B | 9 | 56.25% |

`ptxas` resource output is retained in `artifacts/higs-h2-bwd-cf-build/ptxas_build.log`; the machine-readable table is `resource_usage.csv`.

## Raw measurements

Correctness used room/cam0 at 2048×1365, a fixed seeded nonzero VJP, immutable identical master inputs, and raw Stage-1 plus final-master gradient measurements. It records max/mean absolute difference, relative L2, cosine, NaN/Inf count, and zero/nonzero support disagreement for `v_means2d`, `v_conics`, `v_colors`, `v_opacities`, `means`, `quats`, `scales`, `opacities`, and `SH`. All raw values are in [`correctness_raw.json`](../../artifacts/higs-h2-bwd-cf-build/correctness_raw.json).

CUDA Event backward-only timing used 20 warmups and 100 measured iterations on the same room/cam0 fixture:

| variant | median ms | mean ms | p10 | p90 | std |
|---|---:|---:|---:|---:|---:|
| baseline | 2.166272 | 2.234061 | 2.134016 | 2.294170 | 0.180574 |
| sigma_gate | 2.458624 | 2.462843 | 2.446336 | 2.481152 | 0.017032 |
| scalar_adjoint | 1.927680 | 1.936384 | 1.914880 | 1.942528 | 0.062659 |
| uv_reuse | 2.114048 | 2.113833 | 2.098176 | 2.128998 | 0.012675 |
| combined | 2.191360 | 2.191237 | 2.172928 | 2.207744 | 0.013906 |

The raw per-iteration CSV is [`timing_raw.csv`](../../artifacts/higs-h2-bwd-cf-build/timing_raw.csv).

The separate counter build recorded 178,086,883 candidate pixel-Gaussian evaluations: 77,984,366 pre-exp drops, 31,905 pre-exp clamps, and 100,070,612 samples requiring `exp`; fractions are 0.4379006735, 0.0001791541, and 0.5619201724, respectively. Counter execution was not included in timing. See [`sigma_stats.json`](../../artifacts/higs-h2-bwd-cf-build/sigma_stats.json).

## Artifacts

- [`build_manifest.json`](../../artifacts/higs-h2-bwd-cf-build/build_manifest.json)
- [`binary_manifest.json`](../../artifacts/higs-h2-bwd-cf-build/binary_manifest.json)
- [`resource_usage.csv`](../../artifacts/higs-h2-bwd-cf-build/resource_usage.csv)
- [`correctness_raw.json`](../../artifacts/higs-h2-bwd-cf-build/correctness_raw.json)
- [`timing_raw.csv`](../../artifacts/higs-h2-bwd-cf-build/timing_raw.csv)
- [`sigma_stats.json`](../../artifacts/higs-h2-bwd-cf-build/sigma_stats.json)
