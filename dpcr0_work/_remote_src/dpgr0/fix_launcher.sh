#!/usr/bin/env bash
set -euo pipefail
D=/root/epic05_dpgr0/src_dpgr0/gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference
B=/root/epic05_dpc0/src_full/gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference
# Derive DPGR0 launcher from the DETERMINISTIC DPC0 launcher (no atomicAdd in proj/SH VJP).
cp $B/HigsNativeBackwardDpc.cu $D/HigsNativeBackwardDpgr0.cu
sed -i 's/#include "HigsNativeBackwardDpc.h"/#include "HigsNativeBackwardDpgr0.h"/' $D/HigsNativeBackwardDpgr0.cu
sed -i 's/#include "dpc_stage1.h"/#include "dpgr0_stage1.h"/' $D/HigsNativeBackwardDpgr0.cu
sed -i 's/higs_rasterize_backward_dpc/higs_rasterize_backward_dpgr0/g' $D/HigsNativeBackwardDpgr0.cu
sed -i 's/dpc_stage1(/dpgr0_stage1(/g' $D/HigsNativeBackwardDpgr0.cu
echo "=== verify swaps ==="
grep -n 'HigsNativeBackwardDpgr0.h\|dpgr0_stage1.h\|higs_rasterize_backward_dpgr0\|dpgr0_stage1(\|atomicAdd\|gpuAtomicAdd' $D/HigsNativeBackwardDpgr0.cu | head -30
echo "=== atomicAdd count (should be 0 in proj/SH; reduce kernels use fixed +=) ==="
grep -c 'atomicAdd' $D/HigsNativeBackwardDpgr0.cu || echo 0
