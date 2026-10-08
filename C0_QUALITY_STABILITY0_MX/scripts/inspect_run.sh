#!/usr/bin/env bash
# Inspect one completed run's artifacts to design diagnostics.
d=/dev/shm/c0_quality_stability/runs/room_s43_a0
echo "=== run dir ==="; ls -la "$d"
echo "=== results.json keys ==="
PY=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python
$PY - "$d/results.json" <<'EOF'
import json,sys
r=json.load(open(sys.argv[1]))
print("top keys:", sorted(r.keys()))
print("final_status:", r.get("final_status"))
print("timing keys:", sorted((r.get("timing") or {}).keys()))
print("phase_ms keys:", sorted((r.get("timing",{}).get("phase_ms") or {}).keys()))
print("timing_grade:", r.get("timing_grade"))
print("initial_N:", r.get("initial_N"), "final_N:", r.get("final_N"))
print("total_splits:", r.get("total_splits"), "total_clones:", r.get("total_clones"), "total_prunes:", r.get("total_prunes"))
fe=r.get("final_eval") or {}
print("final_eval keys:", sorted(fe.keys()))
print("final_eval psnr/ssim/lpips:", fe.get("psnr"), fe.get("ssim"), fe.get("lpips"))
EOF
echo "=== checkpoints-lite present? ==="
ls -la "$d"/*.pt 2>/dev/null || echo "no .pt in run dir (checkpoints saved to campaign checkpoints/)"
