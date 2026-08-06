#!/usr/bin/env bash
set -euo pipefail
URL='https://storage.googleapis.com/gresearch/refraw360/stump.zip?generation=1704838352475642'
TOTAL=1528290830
DEST=/mnt/workspace/codex-3dgs-epic05/datasets/downloads
PART=$DEST/mipnerf360-stump.zip.partial
SEG=95518177
# append the single missing terminal byte for rebuilt segments 3 and 7
curl -sL --connect-timeout 20 --max-time 120 -r 382072707-382072707 -o /tmp/fix_byte_3.bin "$URL"
cat /tmp/fix_byte_3.bin >> /tmp/stump_seg_3.part
curl -sL --connect-timeout 20 --max-time 120 -r 764145415-764145415 -o /tmp/fix_byte_7.bin "$URL"
cat /tmp/fix_byte_7.bin >> /tmp/stump_seg_7.part
# verify every segment size
ok=1
for seg in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
  s=$(stat -c %s /tmp/stump_seg_$seg.part 2>/dev/null || echo 0)
  if (( seg == 15 )); then exp=$(( TOTAL - 15 * SEG )); else exp=$SEG; fi
  echo "seg $seg size $s (expect $exp)"
  if (( s != exp )); then ok=0; fi
done
if (( ok != 1 )); then echo 'SEGMENT VERIFY FAILED'; exit 1; fi
rm -f $PART
for seg in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
  cat /tmp/stump_seg_$seg.part >> $PART
done
rm -f /tmp/stump_seg_*.part /tmp/fix_byte_*.bin
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
echo STUMP_DOWNLOAD_DONE
