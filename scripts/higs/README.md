# HiGS development helpers (Windows)

One-command wrappers for the **patched gsplat** dev loop on Windows. They set
`PYTHONPATH` so `import gsplat` resolves to the locally-built patched source
tree and load the MSVC environment (`vcvars64.bat`) that torch's JIT fallback
for `gsplat_scene_cuda` requires. Without `cl` on PATH the HiGS CUDA tests fail
with a misleading `Failed to load gsplat_scene_cuda via JIT build/load` error
(15 failures); with this env the full suite passes (276 passed / 1 skipped on
the local Windows + RTX dev box, 2026-08-04).

## Setup (one-time)

1. Get the patched gsplat source and build it in place:

   ```
   git clone https://github.com/nerfstudio-project/gsplat artifacts/renderer-sources/gsplat
   git -C artifacts/renderer-sources/gsplat checkout 77ab983
   git -C artifacts/renderer-sources/gsplat apply ../../patches/higs-differentiable.patch
   scripts\higs\env.cmd
   python setup.py build_ext --inplace   # from artifacts/renderer-sources/gsplat
   ```

   (`artifacts/renderer-sources/` is git-ignored; the patch is the source of
   truth and applies cleanly to pristine `77ab983`.)

2. Optionally add a tiny `sitecustomize.py` dir to `PYTHONPATH` (the local
   `.build_tmp\pyfix` shim) to avoid a Windows subprocess-decode crash inside
   torch's `cpp_extension`. The helpers add it automatically when present.

## Commands

```
scripts\higs\run_tests.cmd                          # full repo pytest suite
scripts\higs\run_tests.cmd tests/test_higs_native_backward.py::TestTileSampledBackward -q
scripts\higs\run_benchmark.cmd --scene tanks_and_temples/train --backends std higs_native higs_native_ts --tile-sampling-ratio 0.5
```

Overridable variables: `GSPLAT_SRC` (patched tree), `HIGS_PYFIX` (sitecustomize
dir), `HIGS_VCVARS` (vcvars64.bat path), `PYTHON` (python executable).

For the EPIC-05 (Linux A100) flow use `scripts/linux/rebuild_higs_csrc.py` and
the commands in `reports/higs-trainability-implementation.md` instead.
