#!/bin/bash
# Dedicated instr789 tree (DO NOT share with DSH-3, which rsync-clobbers it).
set -u
W=/root/dsh_20261001
PY=/root/miniforge3/envs/p16_fastergs/bin/python
FROZEN=/root/epic05_p16rb/src/nerficg
TREE=$W/instr789/nerficg
ITERS=500,2000,5000,10000,14500,20000,25000
mkdir -p $W/artifacts/DSH789_PHASEB

echo "=== re-sync pristine -> instr789 ==="
mkdir -p $(dirname $TREE)
rsync -a --delete --exclude __pycache__ $FROZEN/ $TREE/ && echo "rsync OK"

echo "=== apply dsh789 trace patch ==="
$PY $W/scripts/dsh789_patch.py --tree $TREE \
  --trace-module $W/scripts/dsh_trace.py \
  --report $W/artifacts/DSH789_PHASEB/instr789_patch_report.json 2>&1 | tail -4

echo "=== verify hooks present ==="
echo -n "rasterization HOLDER stash: "; grep -c "dsh_trace.HOLDER" $TREE/src/Methods/FasterGSFused/FasterGSFusedCudaBackend/FasterGSFusedCudaBackend/torch_bindings/rasterization.py
echo -n "trainer capture call: "; grep -c "dsh_trace.capture" $TREE/src/Methods/FasterGSFused/Trainer.py
echo -n "dsh_trace.py in src: "; test -f $TREE/src/dsh_trace.py && echo yes || echo no
echo -n "py_compile: "; $PY -m py_compile $TREE/src/dsh_trace.py $TREE/src/Methods/FasterGSFused/Trainer.py && echo OK

echo "=== launch garden (GPU6) + bicycle (GPU7) ==="
for scene in garden bicycle; do
  case $scene in
    garden) gpu=6;;
    bicycle) gpu=7;;
  esac
  trace=$W/runs/DSH_PHASEA/$scene/trace
  log=$W/runs/DSH_PHASEA/$scene/train_rerun2.log
  CUDA_VISIBLE_DEVICES=$gpu DSH_TRACE=1 DSH_TRACE_OUT=$trace \
    DSH_TRACE_TAG=PHASEA_$scene DSH_TRACE_SEED=42 DSH_TRACE_ITERS=$ITERS \
    setsid nohup $PY $TREE/scripts/train.py \
    -c $W/configs/dsh789_full_$scene.yaml > $log 2>&1 < /dev/null &
  echo "launched $scene on GPU $gpu pid $! log=$log"
done
