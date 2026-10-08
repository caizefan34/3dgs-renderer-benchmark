#!/bin/bash
/root/miniforge3/envs/gsplat/bin/python - <<'EOF'
import numpy as np
import json, os, time
p = '/root/epic05_p14/raw/P14_ACTIVE_room.npz'
print("npz mtime:", time.ctime(os.path.getmtime(p)))
z = np.load(p)
for cell in range(3):
    a10 = z[f"cell{cell}_q10"]; a20 = z[f"cell{cell}_q20"]; a25 = z[f"cell{cell}_q25"]
    print(f"cell{cell}: |q10|={len(a10)} |q20|={len(a20)} |q25|={len(a25)} "
          f"min={a10.min()} max={a10.max()}")
    s10, s20, s25 = set(a10.tolist()), set(a20.tolist()), set(a25.tolist())
    print(f"  q10 subset of q20: {s10 <= s20}; q20 subset of q25: {s20 <= s25}")
    print(f"  q10[0:5]={a10[:5]} q20[0:5]={a20[:5]}")
sel = json.load(open('/root/epic05_p14/artifacts/P14_SELECTOR.json'))
print("selector active_counts:", [c["active_counts"] for c in sel["scenes"]["room"]["cells"]])
print("unit count check:")
u10 = json.load(open('/root/epic05_p14/P14_O2_UNITS/room_cell0_q10_H32.json'))
u20 = json.load(open('/root/epic05_p14/P14_O2_UNITS/room_cell0_q20_H32.json'))
print(" q10 pre totals:", u10["pre"])
print(" q20 pre totals:", u20["pre"])
EOF
