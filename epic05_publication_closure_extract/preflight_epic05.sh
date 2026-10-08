#!/usr/bin/env bash
set -euo pipefail
REPO="${EPIC_REPO:-/root/3dgs-renderer-benchmark}"
PY="${EPIC_PY:-/root/miniforge3/envs/gsplat/bin/python}"
OUT="${EPIC_OUT:-/root/3dgs-publication-epic05}"
mkdir -p "$OUT/preflight"

date -Is | tee "$OUT/preflight/date.txt"
hostname | tee "$OUT/preflight/hostname.txt"
uname -a | tee "$OUT/preflight/uname.txt"
nvidia-smi -L | tee "$OUT/preflight/nvidia-smi-L.txt"
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version,pstate,power.limit,clocks.sm,clocks.mem,temperature.gpu \
  --format=csv,noheader | tee "$OUT/preflight/gpu_inventory.csv"
"$PY" - <<'PY' | tee "$OUT/preflight/runtime.txt"
import torch, sys, platform
print("python", sys.version)
print("platform", platform.platform())
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("gpu_count", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p=torch.cuda.get_device_properties(i)
    print(i, p.name, p.total_memory, p.major, p.minor)
PY

test "$(nvidia-smi -L | grep -c 'A100-SXM4-80GB')" -eq 8 || {
  echo "FATAL: expected exactly 8 A100-SXM4-80GB GPUs" >&2; exit 2; }

test -d "$REPO/.git" || { echo "FATAL: repo not found at $REPO" >&2; exit 3; }
git -C "$REPO" rev-parse HEAD | tee "$OUT/preflight/repo_commit.txt"
git -C "$REPO" status --porcelain=v1 | tee "$OUT/preflight/repo_status.txt"

echo "P0 inventory PASS. Do not start publication timing until the trainer/binaries also pass the 2K calibration gate."
