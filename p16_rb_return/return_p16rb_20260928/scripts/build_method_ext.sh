#!/bin/bash
# P16-RB: build one external method's CUDA extensions (native submodules).
# usage: build_method_ext.sh <repo_dir> <conda_env> <submodule_paths...>
# Records a pre-build source-state snapshot so any later edit shows up as a diff
# (BUILD_COMPAT_PATCH discipline: minimal, stored, no algorithm change).
set -u
REPO=$1; ENV=$2; shift 2
D=/root/epic05_p16rb
NAME=$(basename "$REPO")
LOG=$D/logs/build_${NAME}.log
SNAP=$D/diffs/${NAME}_prebuild_snapshot.txt

mkdir -p "$D/diffs"
export PATH=/root/miniforge3/envs/$ENV/bin:/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}
export CUDA_HOME=/usr/local/cuda-12.8
export CUDA_PATH=/usr/local/cuda-12.8
export TORCH_CUDA_ARCH_LIST=8.0
export MAX_JOBS=8

cd "$REPO" || exit 3
{
  echo "=== prebuild source state $(date -Is)"
  echo "--- git status --porcelain"
  git status --porcelain
  echo "--- git rev-parse HEAD"
  git rev-parse HEAD
  echo "--- tracked file hashes (submodules + top-level python)"
  find submodules -type f \( -name '*.py' -o -name '*.cu' -o -name '*.cpp' -o -name '*.h' -o -name '*.hpp' \) -print0 2>/dev/null \
    | sort -z | xargs -0 sha256sum 2>/dev/null
} > "$SNAP" 2>&1

{
  echo "=== build $NAME in env $ENV $(date -Is)"
  for SUB in "$@"; do
    echo "=== pip install $SUB"
    /root/miniforge3/envs/$ENV/bin/python -m pip install "$REPO/$SUB" --no-build-isolation
    echo "=== rc=$? for $SUB"
  done
  echo "BUILD_${NAME}_DONE $(date -Is)"
} >> "$LOG" 2>&1

echo "done $NAME rc=$?"
