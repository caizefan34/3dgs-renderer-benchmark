#!/bin/bash
cat > /tmp/tcmp.py <<'EOF'
import json
on = json.load(open('/root/epic05_p13x/runs/room_diag/timing.json'))
off = json.load(open('/root/epic05_p13x/runs/room_diagoff/timing.json'))
for k in ('mean_iter_ms','median_iter_ms','forward_ms','backward_ms','optimizer_ms','loss_ms','total_wall_s'):
    a, b = on[k], off[k]
    print(f"{k}: diag_on={a:.4f} diag_off={b:.4f} rel_diff={(abs(a-b)/b):.4%}")
EOF
/root/miniforge3/envs/gsplat/bin/python /tmp/tcmp.py
