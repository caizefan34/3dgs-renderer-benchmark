#!/bin/bash
cd /root/epic05_p11
/root/miniforge3/envs/gsplat/bin/python - <<'EOF'
import json
d = json.load(open('dumps/P11_DENSIFICATION_TIMING.json'))
print("=== D-GATE summary ===")
g = d["d_gate"]["summary"]
for k, v in g.items():
    print(f"  {k}: {v}")
print("=== rows ===")
for r in d["d_gate"]["rows"]:
    print(f"  {r['scene']}@{r['iteration']}: N={r['N']:,} cons={r['conservative_ms']*1000:.0f}us "
          f"({r['conservative_pct_titer']:.2f}%T {r['conservative_pct_campaign_e2e']:.2f}%E2E) "
          f"mod={r['moderate_ms']*1000:.0f}us ({r['moderate_pct_titer']:.2f}%T "
          f"{r['moderate_pct_campaign_e2e']:.2f}%E2E) opt={r['optimistic_pct_titer']:.2f}%T")
print("=== per-state detail (absgrad aux + chain) ===")
for scene, rec in d["per_scene"].items():
    for it, s in rec["states"].items():
        print(f"  {scene}@{it}: absgrad_aux={s['absgrad_aux_cost_ms']*1000:.0f}us "
              f"chain_total={s['measured_total_stat_update_ms']*1000:.0f}us")
        steps = s.get("steps_ms", s)
        if isinstance(steps, dict):
            for k2, v2 in steps.items():
                if isinstance(v2, (int, float)):
                    pass
EOF
