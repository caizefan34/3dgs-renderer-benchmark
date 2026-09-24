#!/bin/bash
# TASK 0: locate any experiment .so and hash; flag matches to 7ca1c6bf.. or ffbb91cf..
TARGET="7ca1c6bf6c8e4307ecb8fcdbcaf2953bf95305d2c9814f84f3fb5130859301f6"
BASE2="ffbb91cf85e16aabb690ed17cce8c5fabd8f26d67f84136047fc6864b0d3ad9a"
echo "=== all matching .so under pov dirs (hash + path) ==="
find /mnt/storage_pool/liaoyuanjun -type f -name 'experimental_gaussian_render_inference_scene_cuda.so' 2>/dev/null | while read f; do
  h=$(sha256sum "$f" | cut -d' ' -f1)
  printf '%-68s %s\n' "$h" "$f"
done
echo "=== also search names containing experimental_gaussian .so (all forms) ==="
find /mnt/storage_pool/liaoyuanjun -type f -name '*.so' 2>/dev/null | grep -i 'experimental_gaussian' | while read f; do
  h=$(sha256sum "$f" | cut -d' ' -f1)
  printf '%-68s %s\n' "$h" "$f"
done
echo "=== any file whose sha starts 7ca1c6bf (deeper, all .so under likely roots) ==="
for root in higs_h8_mr_cache higs_p3h_cache reconstruction reconstructions artifacts; do
  d="/mnt/storage_pool/liaoyuanjun/$root"
  [ -d "$d" ] && find "$d" -type f -name '*.so' 2>/dev/null | while read f; do
    h=$(sha256sum "$f" | cut -d' ' -f1)
    case "$h" in
      7ca1c6bf*|ffbb91cf*) echo "MATCH $h  $f" ;;
    esac
  done
done
echo "DONE_TASK0"