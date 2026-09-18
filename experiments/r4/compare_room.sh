#!/bin/bash
echo "=== BASELINE room_30k checkpoints ==="
python3 -c '
import json
d = json.load(open("/home/liaoyuanjun/3dgs-renderer-benchmark/results/reference_v1/room_30k/training_metrics.json"))
for k, v in sorted(d["checkpoints"].items(), key=lambda x: int(x[0])):
    print(f"iter {k}: PSNR={v[\"psnr\"]:.2f} SSIM={v[\"ssim\"]:.4f} N={v[\"N_gaussians\"]}")
'
echo "SEP"
echo "=== CANDIDATE C room timing ==="
python3 -c '
import json
try:
    d = json.load(open("/mnt/storage_pool/liaoyuanjun/r4_13scene_v2/room/candidate_c/timing.json"))
    for k, v in d.items():
        if v["total_ms_mean"] > 0:
            print(f"{k}: total={v[\"total_ms_mean\"]:.1f}ms fwd={v[\"fwd_ms_mean\"]:.1f}ms bwd={v[\"bwd_ms_mean\"]:.1f}ms")
except: print("no timing yet")
'
echo "SEP"
echo "=== CANDIDATE C room checkpoints ==="
python3 -c '
import json
try:
    d = json.load(open("/mnt/storage_pool/liaoyuanjun/r4_13scene_v2/room/candidate_c/training_metrics.json"))
    for k, v in sorted(d["checkpoints"].items(), key=lambda x: int(x[0])):
        print(f"iter {k}: PSNR={v[\"psnr\"]:.2f} SSIM={v[\"ssim\"]:.4f} N={v[\"N_gaussians\"]}")
except: print("no metrics yet")
'
