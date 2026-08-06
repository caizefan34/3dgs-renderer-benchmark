#!/usr/bin/env bash
set -euo pipefail
URL='https://storage.googleapis.com/gresearch/refraw360/stump.zip?generation=1704838352475642'
TOTAL=1528290830
DEST=/mnt/workspace/codex-3dgs-epic05/datasets/downloads
PART=$DEST/mipnerf360-stump.zip.partial
SEG=95518177
# segments 2,3,7 need re-download; sub-split each into 8 pieces
for seg in 2 3 7; do
  rm -f /tmp/stump_seg_$seg.part
  base=$(( seg * SEG ))
  for k in 0 1 2 3 4 5 6 7; do
    start=$(( base + k * 11939772 ))
    end=$(( start + 11939771 ))
    if (( end >= base + SEG )); then end=$(( base + SEG - 1 )); fi
    ( curl -sL --connect-timeout 20 --max-time 1800 -r $start-$end -o /tmp/stump_retry_${seg}_$k.part "$URL" ) &
  done
  wait || { echo "retry seg $seg failed"; exit 1; }
  for k in 0 1 2 3 4 5 6 7; do
    cat /tmp/stump_retry_${seg}_$k.part >> /tmp/stump_seg_$seg.part
  done
  rm -f /tmp/stump_retry_${seg}_*.part
  echo "seg $seg size: $(stat -c %s /tmp/stump_seg_$seg.part)"
done
# verify all 16 segments present at expected size
ok=1
for seg in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
  s=$(stat -c %s /tmp/stump_seg_$seg.part 2>/dev/null || echo 0)
  if (( s != SEG )); then
    # last segment may be shorter (TOTAL-15*SEG+1 bytes? check)
    if (( seg == 15 )); then exp=$(( TOTAL - 15 * SEG )); else exp=$SEG; fi
    if (( s != exp )); then echo "seg $seg size $s != $exp"; ok=0; fi
  fi
done
if (( ok != 1 )); then echo 'SEGMENT VERIFY FAILED'; exit 1; fi
rm -f $PART
for seg in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
  cat /tmp/stump_seg_$seg.part >> $PART
done
rm -f /tmp/stump_seg_*.part
actual=$(stat -c %s $PART)
echo "assembled size: $actual (expected $TOTAL)"
if (( actual != TOTAL )); then echo 'SIZE MISMATCH'; exit 1; fi
echo 'md5 check:'
md5sum $PART
python3 - <<'PY'
import binascii
p='/mnt/workspace/codex-3dgs-epic05/datasets/downloads/mipnerf360-stump.zip.partial'
data=open(p,'rb').read()
print('zlib crc32:', hex(binascii.crc32(data)&0xffffffff))
try:
    import google_crc32c
    print('crc32c:', google_crc32c.value(data))
except Exception as e:
    print('google_crc32c unavailable:', e)
PY
mv $PART $DEST/mipnerf360-stump.zip
echo DONE
