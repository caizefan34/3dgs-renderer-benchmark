#!/bin/bash
# Build the two renamed-namespace probes into separate caches.
set -e
ENV=/mnt/storage_pool/liaoyuanjun/higs-13scene-env
BASE=/mnt/storage_pool/liaoyuanjun/p3h_probe
PY="$ENV/bin/python"
export CUDA_HOME="$ENV"
export PATH="$ENV/bin:$PATH"

SNIPPET='import os,sys,time,importlib.util
bf=os.environ["P3H_BF"]
spec=importlib.util.spec_from_file_location("p3h_probe_build",bf)
m=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=m
spec.loader.exec_module(m)
t0=time.time()
mod=m.build_and_load_experimental_gaussian_render_inference_scene()
print("SO_PATH",mod.__file__)
print("BUILD_SECONDS %.3f"%(time.time()-t0))'

function build_probe() {
  local tag=$1 bf=$2 flags=$3 cache=/mnt/storage_pool/liaoyuanjun/p3h_probe_cache/$1
  rm -rf "$cache"; mkdir -p "$cache"
  echo "=== building $tag (flags='$flags') ==="
  P3H_BF="$bf" NVCC_FLAGS="$flags" TORCH_EXTENSIONS_DIR="$cache" "$PY" -c "$SNIPPET"
}

build_probe probe_prod "$BASE/probe_prod/experimental/render/kernels/cuda/build.py" ""
build_probe probe_v0   "$BASE/probe_v0/experimental/render/kernels/cuda/build.py"   "-DP3_SINK_MODE=0"
echo "=== done ==="
ls -la /mnt/storage_pool/liaoyuanjun/p3h_probe_cache/*/