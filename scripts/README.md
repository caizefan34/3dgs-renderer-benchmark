# Command Layout

The user-facing command is `benchmark` (or `python benchmark.py` from a checkout).
Implementation scripts remain under `src/scripts/` so one CLI owns dataset preparation, execution, result collection, and report generation.

Native Ubuntu setup and the strict five-renderer Tier A runner are documented in
[`linux/README.md`](linux/README.md).

## HiGS development (patched gsplat)

The HiGS differentiable-training work builds gsplat from source with
[`patches/higs-differentiable.patch`](../patches/higs-differentiable.patch). On
Windows, [`higs/README.md`](higs/README.md) provides one-command wrappers that
set up the patched tree, load the MSVC environment, and run the test suite or
the training-path benchmark:

```
scripts\higs\run_tests.cmd                              # full repo pytest suite
scripts\higs\run_tests.cmd tests/test_higs_native_backward.py -q
scripts\higs\run_benchmark.cmd --scene tanks_and_temples/train --backends std higs_native higs_native_ts --tile-sampling-ratio 0.5
```

On Linux (EPIC-05), use `linux/rebuild_higs_csrc.py` and the commands in
`reports/higs-trainability-implementation.md` instead.
