#!/bin/bash
# P16-RB: stage the return package (renames only; no deletions of originals) and tar it.
set -eu
D=/root/epic05_p16rb
TS=20260928
STAGE=$D/return_p16rb_${TS}
if [ -e "$STAGE" ]; then mv "$STAGE" "$D/return_p16rb_${TS}_prev_$(date +%H%M%S)"; fi
mkdir -p $STAGE/artifacts $STAGE/runs $STAGE/curves $STAGE/curves_v1_archive $STAGE/profiles \
         $STAGE/envs $STAGE/diffs $STAGE/bundle $STAGE/scripts $STAGE/logs

find $D/artifacts -maxdepth 1 -type f -exec cp -a {} $STAGE/artifacts/ \;

cd $D
find runs -maxdepth 4 -type f \( -name "P16_RUN.json" -o -name "metrics_common.json" \
     -o -name "name_map.json" -o -name "finalize_meta.json" \) | while read f; do
  mkdir -p "$STAGE/$(dirname "$f")"; cp -a "$f" "$STAGE/$f"
done

for M in FasterGS FasterGSFused FastGS SADGS; do
  mkdir -p $STAGE/curves/$M $STAGE/profiles/$M
  for S in room bicycle garden; do
    cp -a curves/$M/$S.json $STAGE/curves/$M/$S.json
    cp -a curves_v1_archive_20260928/${M}__${S}.json $STAGE/curves_v1_archive/ 2>/dev/null || true
    cp -a profiles/$M/${S}_mid.json profiles/$M/${S}_final.json $STAGE/profiles/$M/
  done
done

cp -a envs/. $STAGE/envs/
cp -a diffs/. $STAGE/diffs/
cp -a profiles/_raw/instrumentation_*.diff $STAGE/diffs/
cp -a curve_src/curve_schedule.patch $STAGE/diffs/
cp -a bundle/. $STAGE/bundle/
cp -a scripts/*.py scripts/*.sh $STAGE/scripts/ 2>/dev/null || true
cp -a logs/. $STAGE/logs/

if [ -f $D/bundle/README_RETURN.md ]; then cp -a $D/bundle/README_RETURN.md $STAGE/README_RETURN.md; fi

tar czf $D/p16rb_return_${TS}.tar.gz -C $D return_p16rb_${TS}
sha256sum $D/p16rb_return_${TS}.tar.gz | tee $D/p16rb_return_${TS}.tar.gz.sha256
du -sh $D/return_p16rb_${TS} $D/p16rb_return_${TS}.tar.gz
echo "RETURN_TAR_DONE"
