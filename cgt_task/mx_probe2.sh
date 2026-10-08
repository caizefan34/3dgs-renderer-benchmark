#!/bin/bash
R=/mnt/storage_pool/liaoyuanjun/sota_rebase_v2_runs
D=/tmp/3dgs-renderer-benchmark/data
PY=/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python
echo "=== cameras.bin/images.bin per scene (official) ==="
for s in room garden bicycle; do
  echo "-- $s --"
  ls -la $D/official/mipnerf360/$s/sparse/0/cameras.bin $D/official/mipnerf360/$s/sparse/0/images.bin 2>/dev/null
done
echo "=== datasets sparse cameras ==="
for s in room garden bicycle; do ls $D/datasets/mipnerf360/$s/sparse/0/cameras.bin 2>/dev/null; done
echo "=== garden/bicycle multiphase? ==="
find $R -maxdepth 2 -type d \( -iname '*garden*multi*' -o -iname '*bicycle*multi*' -o -iname '*garden*stage*' -o -iname '*bicycle*stage*' \) 2>/dev/null
echo "=== timing.json (room 30k) ==="
$PY -c "import json;d=json.load(open('$R/B1_room_30k/timing.json'));print(json.dumps(d,indent=1)[:1500])"
echo "=== make isolated dir ==="
mkdir -p /home/liaoyuanjun/cgt_realscene/{src,results,logs,profiles} && echo "made /home/liaoyuanjun/cgt_realscene"
