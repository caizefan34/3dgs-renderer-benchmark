#!/bin/bash
# Diagnose build flags + fatbin archs for prod composed (7ca1c6bf) and V0 (a9265f1b)
CUDA_BIN=$(ls -d /mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin 2>/dev/null)
cuobjdump=$CUDA_BIN/cuobjdump
[ -x "$cuobjdump" ] || cuobjdump=$(which cuobjdump 2>/dev/null)
echo "cuobjdump=$cuobjdump"

for C in c0_cache_composed p3h_cache/V0; do
  echo "===== $C ====="
  SO=/mnt/storage_pool/liaoyuanjun/higs_$C/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so
  # build_params.json location hunt
  bp=$(find /mnt/storage_pool/liaoyuanjun/higs_$C -maxdepth 4 -name build_params.json 2>/dev/null | head -1)
  echo "--- build_params.json: $bp ---"
  [ -n "$bp" ] && python - "$bp" <<'PY'
import json,sys
try:
    d=json.load(open(sys.argv[1]))
except Exception as e:
    print("read err",e); sys.exit()
for k in ("name","extra_cflags","extra_cuda_cflags","extra_ldflags","extra_include_paths"):
    print(k,"=",d.get(k))
PY
  echo "--- fatbin archs ---"
  "$cuobjdump" --list-elf "$SO" 2>/dev/null | head
done
echo DONE_DIAG