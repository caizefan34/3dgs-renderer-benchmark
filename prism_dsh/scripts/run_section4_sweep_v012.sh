#!/bin/bash
# Section 4 fixed matrix: {bicycle,garden} x {5k,14k} x {20260927,20260928,20260929}
# Sequential; per-unit log; skips units already complete. GPU 2 only.
set -u
PY=/root/miniforge3/envs/gsplat/bin/python
TREE=/root/epic05_prism/prism_gs_v0.1.2
OUTDIR=/root/epic05_prism/outputs
cd "$TREE" || exit 1
for scene in bicycle garden; do
  for snap in 005000 014000; do
    case $snap in
      005000) itag=5k ;;
      014000) itag=14k ;;
    esac
    for seed in 20260927 20260928 20260929; do
      out=results/probe_${scene}_${itag}_s${seed}.json
      log=$OUTDIR/probe_${scene}_${itag}_s${seed}.log
      if grep -q PROBE_COMPLETE "$out" 2>/dev/null; then
        echo "SKIP $out"
        continue
      fi
      echo "=== RUN $scene $itag seed $seed $(date -Is) ==="
      PRISM_GPU=2 "$PY" -u -m prism.run_gpu \
        --gx-path /root/epic05_grc3d/scripts --gpu 2 \
        --scene "$scene" \
        --snapshot "/root/epic05_grc3d/inputs/snapshots/snap_${scene}_${snap}.pt" \
        --camera-seq "/root/epic05_p12/runs/p12a/p12a_gmainf_s42_${scene}/camera_sequence.npy" \
        --seed "$seed" --out "$out" > "$log" 2>&1
      echo "exit=$? $(date -Is) $out"
    done
  done
done
echo SWEEP_DONE
