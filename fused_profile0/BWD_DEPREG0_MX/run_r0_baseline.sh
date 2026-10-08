cd /mnt/storage_pool/liaoyuanjun/fused_profile0/BWD_DEPREG0_MX
source run_env.sh
declare -A CK=( [room]=fp1_room_2026-10-06-01-01-33 [bicycle]=fp1_bicycle_2026-10-06-01-01-33 [garden]=fp1_garden_2026-10-06-01-32-48 )
for s in room bicycle garden; do
  $PY scripts/depreg_multitim.py \
    --config /mnt/storage_pool/liaoyuanjun/fused_profile0/profile/mat_${s}.yaml \
    --checkpoint /mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/${CK[$s]}/checkpoints/final.pt \
    --scene $s --modules _C_depreg0 --warmup 50 --iters 100 \
    --out raw/timing_R0_${s}.json > logs/r0_${s}.log 2>&1
  echo "DONE $s exit=$?"
done
echo "ALL_R0_DONE"
