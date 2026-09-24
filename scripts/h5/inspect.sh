#!/bin/bash
echo "== build dir full =="
find /tmp/higs_scalar_adjoint_build_20260920_r2 -type f 2>/dev/null | head -20
echo "== frozen build =="
ls /tmp/higs_scalar_adj_freeze_build 2>/dev/null
find /tmp/higs_scalar_adj_freeze_build -name '*.so' 2>/dev/null | head
echo "== scenario capture results =="
ls -la /tmp/higs_scalar_adjoint_build_20260920_r2/results 2>/dev/null
ls -la /tmp/h3_fwd_1a_r2 2>/dev/null | head
echo "== what npy/tensors in r2 results =="
find /tmp/higs_scalar_adjoint_build_20260920_r2 -name '*.npy' -o -name '*.pt' -o -name '*.bin' 2>/dev/null | head
echo "== grep run log for so path/env =="
head -30 /tmp/higs_scalar_adjoint_build_20260920_r2/../higs_scalar_adjoint_build_20260920_r2/*.log 2>/dev/null
ls /tmp/*.log 2>/dev/null
echo "== repo experimental path =="
ls ~/3dgs-renderer-benchmark/gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference/ 2>/dev/null