#!/bin/bash
find /root -name 'fused_ssim*.so' 2>/dev/null | while read f; do
  h=$(sha256sum "$f" | cut -c1-16)
  echo "$f $h"
done
echo "--- how trainer imports it ---"
grep -n 'fused_ssim' /root/epic05_p12/publication_trainer_p12.py | head -4
grep -rn 'sys.path' /root/epic05_p12/publication_trainer_p12.py | head -6
