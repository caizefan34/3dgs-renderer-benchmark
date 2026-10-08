#!/bin/bash
/root/miniforge3/envs/gsplat/bin/python - <<'EOF'
import json, glob
import numpy as np
print("=== corrected O2 units (H=32) ===")
rows = []
for p in sorted(glob.glob('/root/epic05_p14/P14_O2_UNITS/*_H32.json')):
    u = json.load(open(p))
    rows.append(u)
    t_pre, t_post = u["pre"], u["post"]
    print(f"{u['scene']:9s} c{u['cell']} q={u['q']:.2f}: "
          f"R_stream pre={u['R_stream_pre']:.4f} post={u['R_stream_post']:.4f} "
          f"act_frac={t_pre['active_fraction']:.3f} "
          f"n_seg={t_pre['n_frozen_segments']:,} "
          f"cache={u['cache_gb_fp32_max']:.2f}GB "
          f"R_rebuild={u['rebuild']['R_rebuild']:.4f}")
print()
for q in (0.10, 0.20, 0.25):
    sub = [u for u in rows if abs(u["q"]-q) < 1e-9]
    if sub:
        print(f"q={q}: median R_stream(max)={np.median([u['R_stream_max'] for u in sub]):.4f} "
              f"median R_rebuild={np.median([u['rebuild']['R_rebuild'] for u in sub]):.4f} "
              f"max cache={max(u['cache_gb_fp32_max'] for u in sub):.2f}GB")
print()
print("=== O1 Stage A re-run (R_train vs D1) ===")
units = {}
for p in sorted(glob.glob('/root/epic05_p14/P14_O1_UNITS/*_h32.json')):
    u = json.load(open(p))
    units[(u['scene'], u['cell'], u.get('arm'), u.get('q'))] = u
for q in (0.10, 0.20, 0.25):
    rts, pas = [], []
    for sc in ('room', 'bicycle'):
        for c in range(3):
            d1 = units[(sc, c, 'd1', None)]
            f = units[(sc, c, 'f', q)]
            rt = f['G_train'] / d1['G_train']
            pa = (f['L_audit_after'] - d1['L_audit_after']) / f['L_audit_before']
            rts.append(rt); pas.append(pa)
    print(f"q={q}: median R_train={np.median(rts):.4f} min={min(rts):.4f} "
          f"P_audit range [{min(pas):.4f}, {max(pas):.4f}]")
EOF
