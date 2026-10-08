cd /mnt/storage_pool/liaoyuanjun/fused_profile0/BWD_DEPREG0_MX
source run_env.sh
declare -A CK=( [bicycle]=fp1_bicycle_2026-10-06-01-01-33 [garden]=fp1_garden_2026-10-06-01-32-48 )
for s in bicycle garden; do
  CFG=/mnt/storage_pool/liaoyuanjun/fused_profile0/profile/mat_${s}.yaml
  KCP=/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/${CK[$s]}/checkpoints/final.pt
  for V in dep occ64; do
    $PY scripts/depreg_paired.py --config $CFG --checkpoint $KCP --scene $s \
      --base _C_depreg0_r0 --variant _C_depreg0_${V} --warmup 30 --iters 100 \
      --out raw/paired_${s}_${V}.json > logs/paired_${s}_${V}.log 2>&1
    echo "DONE $s $V exit=$?"
  done
done
echo "PAIRED_OUTDOOR_ALL_DONE"
