#!/bin/bash
# P16-RB curve-schedule smoke: FasterGS room, N=6100, P16_BACKUP_ITERS=5970,6000.
# Expected evidence: checkpoint+render at 5970 (bright, pre-reset), 6000 (dark, just-reset),
# final 6100 post-training. Validates the env-list patch through the production wrapper path.
set -u
D=/root/epic05_p16rb
PYFAST=/root/miniforge3/envs/p16_fastergs/bin/python
WRAP=$D/bundle/p16rb_wrap_run.py

sed -e 's|^  MODEL_NAME: .*|  MODEL_NAME: p16_smoke_dip|' \
    -e 's|^  NUM_ITERATIONS: .*|  NUM_ITERATIONS: 6100|' \
    $D/configs/p16_fastergs_room_1920.yaml > $D/configs/p16_smoke_dip_fastergs_room.yaml
grep -n '^  MODEL_NAME:\|^  NUM_ITERATIONS' $D/configs/p16_smoke_dip_fastergs_room.yaml

if [ -d $D/curves/FasterGS/room_materials ]; then
  mv $D/curves/FasterGS/room_materials $D/curves/FasterGS/room_materials_dip_20260928
  echo MATERIALS_ARCHIVED
fi

mkdir -p $D/smoke
setsid nohup $PYFAST $WRAP --method FasterGS --scene smoke_dip \
  --cwd $D/curve_src/nerficg \
  --command "P16_BACKUP_ITERS=5970,6000 PATH=/root/miniforge3/envs/p16_fastergs/bin:\$PATH python scripts/train.py -c $D/configs/p16_smoke_dip_fastergs_room.yaml" \
  --out $D/smoke/wrapper_train.json --log $D/smoke/train.stdout.log \
  > $D/smoke/wrapper.nohup.log 2>&1 < /dev/null &
echo LAUNCHED_PID=$!

sleep 30
ps aux | grep 'train[.]py' | grep -v grep | head -3
echo ---
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed -n '3p'
ls $D/curve_src/nerficg/output/FasterGS/ 2>/dev/null
echo SMOKE_LAUNCH_END
