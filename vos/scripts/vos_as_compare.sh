#!/bin/bash
# A_s attribution comparison after the as_room_30k run.
PY=/root/miniforge3/envs/gsplat/bin/python
cd /root/vos_traces
tail -1 logs/vos_as_room_30k.log
$PY - <<'EOF'
import csv, os, json
for d in ["as_room_30k", "c0_room_30k", "c0_room_30k_b",
          "vos_room_30k", "vos_room_30k_b"]:
    p = f"vos_runs/{d}/training_curve.csv"
    if not os.path.exists(p):
        print(f"== {d}: NO CURVE"); continue
    rows = list(csv.DictReader(open(p)))
    pts = [(int(r["step"]), float(r["psnr"]), r["tag"])
           for r in rows if r["tag"] in ("eval", "final")]
    res = json.load(open(f"vos_runs/{d}/results.json"))
    fe = res["final_eval"]
    traj = " ".join(f"t{t}:{p:.2f}" for t, p, _ in pts if t >= 15000)
    print(f"== {d}: final(all-cam) psnr={fe['psnr']:.4f} "
          f"ssim={fe['ssim']:.5f} lpips={fe['lpips']:.5f}")
    print(f"   traj: {traj}")
EOF
