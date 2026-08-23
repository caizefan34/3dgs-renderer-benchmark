#!/usr/bin/env bash
set -euo pipefail
URL='https://storage.googleapis.com/gresearch/refraw360/stump.zip?generation=1704838352475642'
TOTAL=1528290830
DEST=/mnt/workspace/codex-3dgs-epic05/datasets/downloads
PART=$DEST/mipnerf360-stump.zip.partial
NSEG=16
rm -f $PART /tmp/stump_seg_*.part
seg_size=$(( (TOTAL + NSEG - 1) / NSEG ))
for ((i=0; i<NSEG; i++)); do
  start=$(( i * seg_size ))
  end=$(( start + seg_size - 1 ))
  if (( end >= TOTAL )); then end=$(( TOTAL - 1 )); fi
  ( curl -sL --connect-timeout 20 --max-time 3600 -r $start-$end -o /tmp/stump_seg_$i.part "$URL" ) &
done
fail=0
if ! wait; then fail=1; echo 'some segment jobs failed' >&2; fi
if (( fail )); then echo 'SOME SEGMENTS FAILED' >&2; exit 1; fi
for ((i=0; i<NSEG; i++)); do
  cat /tmp/stump_seg_$i.part >> $PART
done
rm -f /tmp/stump_seg_*.part
actual=$(stat -c %s $PART)
echo "assembled size: $actual (expected $TOTAL)"
if (( actual != TOTAL )); then echo 'SIZE MISMATCH' >&2; exit 1; fi
echo 'md5 check:'
md5sum $PART
echo 'crc32c check (via python):'
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
