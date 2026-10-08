#!/bin/bash
set -u
T=/root/epic05_p8/gsplat-main-512d366b/gsplat/cuda/csrc
echo "=== 3DGS Bwd kernel traversal loop ==="
grep -n 'for (\|last_id\|tile_start\|contribut\|min(T\|contributes\|break\|alpha' $T/RasterizeToPixels3DGSSerialBatchBwd.cu | head -40
echo "=== Device.cuh fwd termination + last_ids ==="
grep -n 'last_ids\|contributes\|min(T\|next_T\|alpha.*test\|break' $T/RasterizeToPixels3DGSDevice.cuh | head -30
echo "=== num_contributing kernel ==="
grep -rn 'num_contributing\|NumContributing' $T/*.cu $T/*.cuh 2>/dev/null | head -10
