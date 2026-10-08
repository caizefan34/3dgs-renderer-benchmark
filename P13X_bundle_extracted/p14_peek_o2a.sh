#!/bin/bash
/root/miniforge3/envs/gsplat/bin/python - <<'EOF'
import json, glob
import numpy as np
rows = []
for p in sorted(glob.glob('/root/epic05_p14/P14_O2_UNITS/*_H32.json')):
    u = json.load(open(p))
    rows.append((u['scene'], u['cell'], u['q'],
                 u['R_stream_pre'], u['R_stream_post'],
                 u['cache_gb_fp32_max'], u['rebuild']['R_rebuild'],
                 u['rebuild']['n_tiles_membership_change'],
                 u['rebuild']['n_tiles_order_only_change'],
                 u['rebuild']['n_tiles']))
print(f"{'scene':9s} {'c':2s} {'q':5s} {'Rstr_pre':9s} {'Rstr_post':9s} {'cacheGB':8s} {'R_rebuild':10s} {'mem/order/tiles'}")
for r in rows:
    print(f"{r[0]:9s} {r[1]:2d} {r[2]:5.2f} {r[3]:9.4f} {r[4]:9.4f} {r[5]:8.2f} {r[6]:10.4f} {r[7]}/{r[8]}/{r[9]}")
for q in (0.10, 0.20, 0.25):
    sub = [r for r in rows if abs(r[2]-q) < 1e-9]
    if sub:
        print(f"q={q}: median R_stream(max)={np.median([max(r[3],r[4]) for r in sub]):.4f} "
              f"median R_rebuild={np.median([r[6] for r in sub]):.4f} "
              f"max cache={max(r[5] for r in sub):.2f} GB")
EOF
