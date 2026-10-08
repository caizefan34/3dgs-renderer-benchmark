#!/bin/bash
# Eval-trajectory comparison via python (CRLF-safe).
PY=/root/miniforge3/envs/gsplat/bin/python
cd /root/vos_traces
$PY - <<'EOF'
import csv
for d in ["c0_room_30k", "vos_room_30k", "vos_room_30k_b", "c0_room_30k_b"]:
    rows = list(csv.DictReader(open(f"vos_runs/{d}/training_curve.csv")))
    pts = [(int(r["step"]), float(r["psnr"]), int(r["n_gaussians"]))
           for r in rows if r["tag"] in ("eval", "final")]
    print(f"== {d} ==")
    for t, p, n in pts:
        print(f"  t={t} psnr={p:.3f} n={n}")
EOF
