#!/bin/bash
# DPC0 build setup: mirror the C0 extension build, swap in the DPC backward.
set -e
SRC=/root/epic05_frozen/higs_c0_final30k_worktree/gsplat/experimental/render/kernels/cuda
GSI=$SRC/csrc/gaussian_inference
DPC=/root/epic05_dpc0/dpc_src
BLD=/root/epic05_dpc0/build_dpc
EXT=$BLD/experimental_gaussian_render_inference_scene_cuda
rm -rf $BLD
mkdir -p $EXT
# copy all C0 sources except HigsNativeBackward.cu
cp $SRC/ext.cpp $EXT/
for f in GaussianRenderInferenceScene.cu GatherVisible.cu GatherlessProjectedProducer.cu \
         IntersectCommon.cu IntersectMTFused.cu MacroTileIntersect.cu MacroTileRasterize.cu \
         Projection.cu SegmentedSort.cu SHCompression.cu SphericalHarmonics.cu; do
  cp $GSI/$f $EXT/
done
# headers (so includes resolve from the build dir too)
cp $GSI/*.h $EXT/ 2>/dev/null || true
# DPC sources
cp $DPC/HigsNativeBackwardDpc.cu $EXT/
cp $DPC/dpc_stage1.cu $EXT/
cp $DPC/dpc_stage1.h $EXT/
cp $DPC/HigsNativeBackwardDpc.h $EXT/
# also need HigsNativeBackward.h (the DPC .cu includes it)
cp $GSI/HigsNativeBackward.h $EXT/ 2>/dev/null || true
echo "=== files in build dir ==="
ls -la $EXT/
