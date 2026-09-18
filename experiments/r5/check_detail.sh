#!/bin/bash
echo "=== Baseline metrics ==="
for s in 1 2; do
  for scene in train truck; do
    dir="/mnt/storage_pool/liaoyuanjun/r5_a/${scene}_seed${s}/baseline"
    echo -n "$scene seed=$s: "
    if [ -f "$dir/training_metrics.json" ]; then
      python3 -c "import json; d=json.load(open('$dir/training_metrics.json')); ck=d['checkpoints']; k=max(ck.keys(), key=int); print(f'iter {k}: PSNR={ck[k][\"psnr\"]:.2f} SSIM={ck[k][\"ssim\"]:.4f} N={ck[k][\"N_gaussians\"]}')" 2>/dev/null
    else
      echo "metrics NOT SAVED"
    fi
  done
done

echo ""
echo "=== All python processes ==="
ps aux | grep python | grep -v grep | grep r4_train | head -10

echo ""
echo "=== Candidate progress ==="
for s in 1 2; do
  for scene in train truck; do
    log="/mnt/storage_pool/liaoyuanjun/r5_a_logs/${scene}_seed${s}_candidate.log"
    echo -n "$scene seed=$s candidate: "
    grep "ITER" "$log" 2>/dev/null | tail -1 | head -c 150
    echo
  done
done
