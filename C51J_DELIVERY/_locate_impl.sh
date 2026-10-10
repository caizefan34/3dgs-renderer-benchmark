#!/bin/bash
set +e
ROOT=/mnt/storage_pool/liaoyuanjun
echo "=== A. trainer source with radius_clip + opacity-prune logic (fused_prune10_rclip05 impl) ==="
grep -rIl --include=*.py -e "radius_clip" -e "opacity_prune" -e "prune_points" -e "fused" $ROOT 2>/dev/null | grep -iE "train|higs|gsplat|accel|confirmatory" | head -25
echo
echo "=== B. confirmatory_accel15 per-run raw results (json/csv) anywhere ==="
find $ROOT -maxdepth 8 -type d -iname "*accel15*" 2>/dev/null | head -20
find $ROOT -maxdepth 8 -type d -iname "*confirmatory*" 2>/dev/null | head -20
echo
echo "=== C. per-run jsons keyed by scene+seed for the 11 scenes (psnr/wall) ==="
find $ROOT -maxdepth 8 -type f -name "*.json" 2>/dev/null | grep -iE "accel15|confirmatory|fused_prune" | head -25
echo
echo "=== D. dataset split + eval scripts ==="
find $ROOT -maxdepth 6 -type f -iname "*split*.py" -o -maxdepth 6 -type f -iname "*eval*.py" 2>/dev/null | grep -iE "mipnerf|360|higs|gsplat|eval" | head -20
echo
echo "=== E. EPIC-05 presence ==="
ls -d $ROOT/epic05 $ROOT/EPIC05 $ROOT/*epic05* 2>/dev/null | head
find $ROOT -maxdepth 3 -type d -iname "*epic05*" 2>/dev/null | head -10
echo
echo "=== F. higs-paper-protocol + trainer commit record ==="
find $ROOT -maxdepth 6 -iname "higs-paper-protocol.json" -o -maxdepth 6 -iname "*trainer*sha*" 2>/dev/null | head -10
echo "DONE"
