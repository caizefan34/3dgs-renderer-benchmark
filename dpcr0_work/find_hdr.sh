D=/root/epic05_frozen/higs_c0_final30k_worktree/gsplat/experimental/render/kernels/cuda
echo "=== cuda dir ==="
ls "$D"
echo
echo "=== find headers ==="
find "$D" -name "RasterizeToPixels3DGSDevice.cuh" 2>/dev/null
find "$D" -name "Utils.cuh" 2>/dev/null
find "$D" -name "Common.h" 2>/dev/null
find "$D" -name "Config.h" 2>/dev/null