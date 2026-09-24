# H1-CLEAN-REPRO — terminal status

## BUILD/ABI BLOCKER

The existing build at `/tmp/h1_clean_authoritative` completed compilation and linking. The cache-built binary is `/tmp/h1_clean_authoritative/gsplat_cuda/gsplat_cuda.so`, SHA256 `e5088f91236013a2ef8a820ab4824aa2863b29e7b4dc0525e8296fa1455c8286` (90,316,496 bytes; mtime 2026-09-19 22:55:14 +0800). It was loaded from that exact path in a fresh Python process. The first build process itself aborted during import because it loaded the old `/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx/gsplat/csrc.so` (SHA256 `40a009837e75620454059cfdbb5088cc08423f281e4c734da3053b727d39df50`) and then attempted a second `TORCH_LIBRARY("gsplat")` registration. Directly loading the cache binary in isolation succeeded; this was loader contamination, not a compile or link failure.

The ABI gate **passed** for `projection_ewa_3dgs_fused`, `intersect_tile`, and `rasterize_to_pixels_3dgs`. Their source wrappers explicitly reorder arguments to the runtime schemas, including `opacities` into projection operator position 4. No wrapper/binary positional mismatch remains.

The B1A room/cam0 sanity test **passed** at 2048×1365: 115,278 total Gaussians, 44,908 visible, 953,144 intersections, 11,008 active tiles, RGB finite, alpha finite. This is the expected ~10^6 scale, not the pathological ~10^9 scale.

The cohort cannot proceed because the pinned tree at `77ab983ffe43420b2131669cb35776b883ca4c3c` exposes only inference rendering. It contains no `rasterize_gaussian_higs_frozen`, `create_higs_renderer`, or `higs_native` backward API; the required `HigsNativeBackward.cu` is also absent from that source tree and from the built extension. The attempted import failed exactly with `ImportError: cannot import name 'rasterize_gaussian_higs_frozen' from 'gsplat.experimental'`. The full-trainable wrapper and native backward exist only in the dirty H1 checkout and require a separate native-backward extension build. Because another build was explicitly prohibited, no old extension was reused and profiling stopped before correctness/timing.

| Gate | Result |
|---|---|
| Core source/build match | PASS — 77ab983 tracked sources; exact GLM gitlink payload |
| Compile/link | PASS |
| Runtime ABI | PASS |
| B1A room/cam0 scale + finite output | PASS — 953,144 intersections |
| B1A/B2 correctness | BLOCKED — B2 implementation unavailable in pinned source/build |
| 3-scene × 3-camera timing cohort | NOT RUN |
| Classification / geomean | NOT RUN |

GLM provenance: commit `77ab983` records gitlink `2d4c4b4dd31fde06cfffad7915c2b3006402322f`. The dependency payload was absent from the source archive because the submodule was uninitialized. `/mnt/storage_pool/liaoyuanjun/higs-glm-2d4c4b4` was compared against an exact checkout; all 1,665 file hashes matched.

Toolchain verification: `CUDA_HOME=/mnt/storage_pool/liaoyuanjun/higs-13scene-env`; Ninja invokes `/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/nvcc` (CUDA 12.8 V12.8.93), includes and libraries come from that environment, and the binary targets `compute_80/sm_80`. Login-shell CUDA 11.5 was not used.

Source commits: HiGS base `77ab983ffe43420b2131669cb35776b883ca4c3c` (tree `0a8487ad41b2ccfb4dc066fd7ffcb82f2a21eb86`); research repo `02375033388d4348376b6b607ab85f551e498a77`; AccuTile reference `28e794ca44a4c25ffc39175370c5ee7b38bfcc36`; GLM gitlink `2d4c4b4dd31fde06cfffad7915c2b3006402322f`.

Artifacts: [environment.json](../../artifacts/h1-clean-repro/environment.json), [source_manifest.json](../../artifacts/h1-clean-repro/source_manifest.json), [build_manifest.json](../../artifacts/h1-clean-repro/build_manifest.json), [abi_audit.json](../../artifacts/h1-clean-repro/abi_audit.json), [room_cam0_smoke.json](../../artifacts/h1-clean-repro/room_cam0_smoke.json), [analysis.json](../../artifacts/h1-clean-repro/analysis.json). Timing and correctness CSVs intentionally contain no cohort measurements.
