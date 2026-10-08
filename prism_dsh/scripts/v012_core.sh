#!/bin/bash
# v0.1.2 verification: CPU tests -> cpu_probe -> M0 -> 4 parity states.
# Thresholds, camera rules, selection and arms are identical to v0.1.1.
set -u
PY=/root/miniforge3/envs/gsplat/bin/python
TREE=/root/epic05_prism/prism_gs_v0.1.2
SCRIPTS=/root/epic05_prism/scripts
OUTDIR=/root/epic05_prism/outputs
cd "$TREE" || exit 1

echo "=== unit tests $(date -Is) ==="
"$PY" -m unittest discover -s tests -v > results/unit_tests.log 2>&1
echo "exit=$? $(tail -3 results/unit_tests.log | tr '\n' ' ')"

echo "=== cpu_probe $(date -Is) ==="
"$PY" -m prism.cpu_probe --out results/cpu_probe.json > "$OUTDIR/cpu_probe_v012.log" 2>&1
echo "exit=$? $(date -Is)"

echo "=== M0 bicycle 5k $(date -Is) ==="
PRISM_GPU=2 "$PY" -u -m prism.run_gpu \
  --gx-path /root/epic05_grc3d/scripts --gpu 2 --scene bicycle \
  --snapshot /root/epic05_grc3d/inputs/snapshots/snap_bicycle_005000.pt \
  --camera-seq /root/epic05_p12/runs/p12a/p12a_gmainf_s42_bicycle/camera_sequence.npy \
  --m0-only --out results/bicycle_5k_m0.json > "$OUTDIR/m0_v012_run.log" 2>&1
echo "exit=$? $(date -Is)"

for sp in bicycle:5000 bicycle:14000 garden:5000 garden:14000; do
  scene=${sp%%:*}; state=${sp##*:}
  echo "=== parity $scene $state $(date -Is) ==="
  "$PY" -u "$SCRIPTS/parity_v012.py" --gpu 2 --scene "$scene" --state "$state" \
    --out "$OUTDIR/parity_${scene}_${state}.json" \
    --arrays-dir "$OUTDIR/parity_arrays_v012/${scene}_${state}" \
    > "$OUTDIR/parity_${scene}_${state}_v012.log" 2>&1
  echo "exit=$? $(date -Is)"
done
echo CORE_DONE
