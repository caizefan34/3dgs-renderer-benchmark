#!/bin/bash
# Region (15K->30K] optimizer-stage A/B for the finished arms.
PY=/root/miniforge3/envs/gsplat/bin/python
cd /root/vos_traces
$PY - <<'EOF'
import numpy as np, json, os
for s in ["room", "bicycle", "garden"]:
    for arm in ["c0", "vos"]:
        p = f"vos_runs/{arm}_{s}_30k/vos_timing.npz"
        if not os.path.exists(p):
            continue
        z = np.load(p)
        t, inner, vos = z["t"], z["inner_ms"], z["vos_ms"]
        reg = (t > 15000) & (t <= 30000)
        pre = t <= 15000
        st = json.load(open(f"vos_runs/{arm}_{s}_30k/vos_stats.json"))
        tim = json.load(open(f"vos_runs/{arm}_{s}_30k/timing.json"))
        print(f"{s}/{arm}: region_inner={np.mean(inner[reg]):.3f}ms "
              f"region_vos={np.mean(vos[reg]):.3f}ms "
              f"region_total={np.mean(inner[reg]+vos[reg]):.3f}ms "
              f"pre_inner={np.mean(inner[pre]):.3f}ms "
              f"wall={tim['total_wall_s']:.1f}s "
              f"steps={st['n_vos_steps']} fences={len(st['fences'])} "
              f"replay={st['replay_steps_sum']} mode={st['vos_mode']}")
EOF
