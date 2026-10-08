cd /mnt/storage_pool/liaoyuanjun/fused_profile0/BWD_DEPREG0_MX
source run_env.sh
CFG=/mnt/storage_pool/liaoyuanjun/fused_profile0/profile/mat_room.yaml
CK=/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_room_2026-10-06-01-01-33/checkpoints/final.pt
for V in dep nocolor noconic occ64; do
  $PY scripts/depreg_paired.py --config $CFG --checkpoint $CK --scene room \
    --base _C_depreg0_r0 --variant _C_depreg0_${V} --warmup 30 --iters 120 \
    --out raw/paired_room_${V}.json > logs/paired_room_${V}.log 2>&1
  echo "DONE $V exit=$?"
done
echo "PAIRED_ROOM_ALL_DONE"
