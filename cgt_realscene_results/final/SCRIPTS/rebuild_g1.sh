#!/bin/bash
set -u
PY=/mnt/storage_pool/liaoyuanjun_mx0qual/bin/python
FIN=/home/liaoyuanjun/cgt_realscene/final
REC=$FIN/G1_RAW_RECORDS
cd "$FIN" || exit 1

echo "=== STEP 0: quarantine stale buggy g1_room_r*.json (single room) ==="
mkdir -p "$REC/stale_superseded_buggy_teacher"
for f in g1_room_r5k.json g1_room_r10k.json g1_room_r15k.json g1_room_r30k.json; do
  if [ -f "$REC/$f" ]; then mv "$REC/$f" "$REC/stale_superseded_buggy_teacher/"; echo "moved $f"; fi
done

echo "=== STEP 1: rename fresh g1_room_room_r*.json -> g1_room_r*.json ==="
for r in r5k r10k r15k r30k; do
  if [ -f "$REC/g1_room_room_$r.json" ]; then mv "$REC/g1_room_room_$r.json" "$REC/g1_room_$r.json"; echo "renamed room_$r"; fi
done

echo "=== remaining g1_*.json (should be 6 fresh) ==="
ls -la --time-style=+%H:%M "$REC"/g1_*.json | awk "{print \$6,\$7,\$8}"

echo "=== STEP 2: analyze_g1.py -> G1_REAL_SCENE_COMPRESSIBILITY.json ==="
cd "$FIN/SCRIPTS"
$PY analyze_g1.py --inputs "$REC/g1_*.json" --out "$FIN/G1_REAL_SCENE_COMPRESSIBILITY.json" 2>&1
