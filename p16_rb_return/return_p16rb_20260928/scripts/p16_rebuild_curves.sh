#!/bin/bash
# P16-RB: uniform rebuild of all 12 curve JSONs with the callback-corrected builder v2.
# Run ONLY after CURVES_ALL_DONE. Archives v1 curve JSONs by rename (never deletion),
# then rebuilds every pair from persisted materials (deterministic; re-runnable).
set -u
D=/root/epic05_p16rb
PY=/root/miniforge3/envs/gsplat/bin/python
ARCH=$D/curves_v1_archive_20260928
mkdir -p $ARCH

for M in FasterGS FasterGSFused FastGS SADGS; do
  for S in room bicycle garden; do
    OLD=$D/curves/$M/$S.json
    if [ -f "$OLD" ]; then
      mv "$OLD" "$ARCH/${M}__${S}.json"
      echo "ARCHIVED_V1 $M $S"
    fi
  done
done

for PAIR in FasterGS:room FasterGS:bicycle FasterGS:garden \
            FasterGSFused:room FasterGSFused:bicycle FasterGSFused:garden \
            FastGS:room FastGS:bicycle FastGS:garden \
            SADGS:room SADGS:bicycle SADGS:garden; do
  M=${PAIR%%:*}; SC=${PAIR##*:}
  MAT=$D/curves/$M/${SC}_materials
  if [ ! -f $MAT/points.json ]; then
    echo "REBUILD_SKIP_${M}_${SC} (no materials)"
    continue
  fi
  if [ "$M" = "FasterGS" ] || [ "$M" = "FasterGSFused" ]; then
    LOWER=$(echo $M | tr 'A-Z' 'a-z')
    OUTD=$(ls -d $D/curve_src/nerficg/output/$M/p16_curve_${LOWER}_${SC}_* 2>/dev/null | tail -1)
    $PY $D/scripts/p16_build_curve2.py --method $M --scene $SC --points-json $MAT/points.json \
      --train-log $MAT/train.stdout.log --materials-dir $MAT --nerficg-outdir $OUTD \
      --out $D/curves/$M/$SC.json >> $MAT/curve_build_v2.log 2>&1
  else
    $PY $D/scripts/p16_build_curve2.py --method $M --scene $SC --points-json $MAT/points.json \
      --train-log $MAT/train.stdout.log --materials-dir $MAT \
      --out $D/curves/$M/$SC.json >> $MAT/curve_build_v2.log 2>&1
  fi
  echo "REBUILD_${M}_${SC}_rc=$?"
done

$PY $D/scripts/p16_verify_curves.py
VRC=$?
echo "CURVES_REBUILD_DONE rc=$VRC"
exit $VRC
