#!/bin/bash
cd /root/epic05_p11
echo "=== pipeline.log tail ==="
tail -n 8 logs/pipeline.log
echo "=== stategen summaries ==="
for s in room bicycle garden; do
  echo "--- $s ---"
  tail -n 3 logs/stategen_$s.log 2>/dev/null
  cat stategen/$s/stategen_summary.json 2>/dev/null
  echo
done
echo "=== camera sequence cross-check (stategen vs campaign) ==="
/root/miniforge3/envs/gsplat/bin/python - <<'EOF'
import numpy as np, os
for s in ["room", "bicycle", "garden"]:
    a = np.load(f"/root/epic05_p11/stategen/{s}/camera_sequence.npy")
    b = np.load(f"/root/3dgs-publication-epic05/full30k/gmain03_s42_{s}/camera_sequence.npy")
    n = min(len(a), 14000)
    same = bool((a[:n] == b[:n]).all())
    print(f"{s}: stategen len={len(a)} campaign len={len(b)} first-{n}-match={same}")
EOF
