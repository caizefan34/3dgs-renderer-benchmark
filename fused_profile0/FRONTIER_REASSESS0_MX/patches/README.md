# patches/

Probe instrumentation for the FRONTIER-REASSESS0 lane.

- 0001_f2_expf_variants.diff : unified diff of kernels_backward.cuh adding the
  F2 expf formulation macros (FRONTIER_EXPF_FAST / FRONTIER_EXPF_EXP2F) gating the
  gaussian computation. Production r0 path unchanged (no macro = expf).

Macros are injected by rewriting rasterization/include/depreg_probe.h per build
variant (build_f2.sh). All F2 variants are ORACLE/diagnostic; none promoted.
