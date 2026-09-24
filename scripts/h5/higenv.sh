#!/bin/bash
HP=/mnt/storage_pool/liaoyuanjun/higs-13scene-env
echo "== higs-13scene-env =="
ls -d $HP/python* $HP/*/bin/*python* $HP/targets/x86_64-linux/bin/python* 2>/dev/null
ls $HP 2>/dev/null | head
find $HP -maxdepth 3 -name 'python*' -type f 2>/dev/null | head
echo "== any gsplat .so/python under it =="
find $HP -maxdepth 6 -iname 'gsplat_cuda*' -o -iname 'gsplat*csrc*' 2>/dev/null | head
echo "== gsplat_cuda .so search whole disk-lite =="
find /home/liaoyuanjun -maxdepth 6 -name 'gsplat*.so' 2>/dev/null | head
find /mnt/storage_pool/liaoyuanjun -maxdepth 8 -name 'gsplat_cuda*.so' 2>/dev/null | head