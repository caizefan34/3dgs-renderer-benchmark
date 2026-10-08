#!/bin/bash
cd /root/msb0_p0/scripts
PY=/root/miniforge3/envs/p16_fastergs/bin/python
CK[room]=/root/epic05_p16rb/src/nerficg/output/FasterGSFused/fgs_fused_iso_room_s42_30000_*/checkpoints/final.pt
CK[garden]=/root/epic05_p16rb/src/nerficg/output/FasterGSFused/fgs_fused_iso_garden_s42_30000_2026-10-01-09-45-48/checkpoints/final.pt
CK[bicycle]=/root/epic05_p16rb/src/nerficg/output/FasterGSFused/fgs_fused_iso_bicycle_s42_30000_2026-10-01-09-55-47/checkpoints/final.pt
for s in room garden bicycle; do
  sp=/root/epic05_p16rb/common_data/mipnerf360/$s/sparse/0
  echo "=== $s ==="
  CUDA_VISIBLE_DEVICES=0 $PY msb0_economics.py --scene "$s" --ckpt "${CK[$s]}" --sparse "$sp" --out /root/msb0_p0/out 2>&1 | tail -4
done
