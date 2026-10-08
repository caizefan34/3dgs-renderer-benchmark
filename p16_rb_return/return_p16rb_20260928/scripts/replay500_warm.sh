#!/bin/bash
# P16-RB diagnostic: 500-step replay with a warmed GPU + SM-clock telemetry.
# Purpose: distinguish DVFS/warm-state effects from true machine drift.
D=/root/epic05_p16rb
mkdir -p $D/runs/P12_C0_replay500b/room $D/logs
export CUDA_VISIBLE_DEVICES=2

env -u CUDA_VISIBLE_DEVICES nvidia-smi --query-gpu=index,clocks.sm,clocks.mem,power.draw,utilization.gpu,temperature.gpu \
  --format=csv,noheader -i 2 -l 1 > $D/logs/replay500b.clocks.csv &
POLL=$!
echo "POLLER=$POLL $(date -Is)"

/root/miniforge3/envs/gsplat/bin/python - <<'EOF'
import time, torch
torch.cuda.init()
a = torch.randn(8192, 8192, device="cuda")
b = torch.randn(8192, 8192, device="cuda")
torch.cuda.synchronize()
t0 = time.time(); n = 0
while time.time() - t0 < 60:
    c = a @ b; n += 1
    if n % 20 == 0:
        torch.cuda.synchronize()
torch.cuda.synchronize()
print("warmup_matmuls", n, "elapsed_s", round(time.time() - t0, 1), flush=True)
EOF
echo "WARMUP_RC=$?"
env -u CUDA_VISIBLE_DEVICES nvidia-smi --query-gpu=index,clocks.sm,power.draw --format=csv,noheader -i 2

env -u CUDA_VISIBLE_DEVICES nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $D/logs/replay500b.procs_before.csv
uptime > $D/logs/replay500b.uptime_before.txt
T0=$(date -Is)
/root/epic05_py_wrapper.sh /root/epic05_p12/publication_trainer_p12.py --arm c0 --scene room --iterations 500 \
  --outdir $D/runs/P12_C0_replay500b/room --gpu 2 --lpips --final-eval all \
  --timing-grade PUBLICATION --eps2d 0.3 --loss fused > $D/logs/replay500b.stdout.log 2>&1
RC=$?
T1=$(date -Is)
uptime > $D/logs/replay500b.uptime_after.txt
kill $POLL 2>/dev/null
echo "RC=$RC T0=$T0 T1=$T1"

/root/miniforge3/envs/gsplat/bin/python - <<'EOF'
import json
t = json.load(open("/root/epic05_p16rb/runs/P12_C0_replay500b/room/timing.json"))
print("median_iter_ms", t["median_iter_ms"], "mean_iter_ms", t["mean_iter_ms"], "wall", t["total_wall_s"])
print("phase medians:", {k: v["median_ms"] for k, v in t["phase_ms"].items()})
EOF
echo "-- SM clock histogram during run --"
awk -F, '{gsub(/ /,"",$2); print $2}' $D/logs/replay500b.clocks.csv | sort -n | uniq -c | sort -rn | head -12
echo "REPLAY500B_DONE $(date -Is)"
