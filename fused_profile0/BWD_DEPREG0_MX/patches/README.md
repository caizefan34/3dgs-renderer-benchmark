# patches/

Probe instrumentation for the BWD-DEPREG0 diagnostic lane.

- 0001_blend_backward_probe_instrumentation.diff : unified diff of
  rasterization/include/kernels_backward.cuh (pristine .orig -> instrumented).
  Adds the depreg_probe.h include and gates the recurrence / color / opacity /
  conic gradient blocks behind BWD_DEPREG_* macros.

Macros are NOT baked into the source. build_fast.sh rewrites
rasterization/include/depreg_probe.h per variant (for example
"define BWD_DEPREG_NORECUR 1") before compiling, so a single source tree
produces all _C_depreg0_<tag>.so variants. R0 (production) writes only
"pragma once" into the probe header (no macro defined) -> production path.

All gated variants are ORACLE_ONLY / SEMANTICALLY_INVALID and are never used
for production or correctness claims. See BWD_DEPREG0_SOURCE_DIFF.json.
