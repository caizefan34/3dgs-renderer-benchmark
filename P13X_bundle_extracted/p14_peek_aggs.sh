#!/bin/bash
/root/miniforge3/envs/gsplat/bin/python - <<'EOF'
import json
a = json.load(open('/root/epic05_p14/artifacts/P14_O1_AGG.json'))
print("O1_AGG top-level keys:", list(a.keys()))
print(json.dumps({k: v for k, v in a.items()
                  if k not in ("units", "by_H", "q_stats", "per_arm")},
                 indent=1, default=str)[:3000])
o = json.load(open('/root/epic05_p14/artifacts/P14_O2_AGG.json'))
print("\nO2_AGG keys:", list(o.keys()))
print("decision:", o["decision"], "| O2_PASS_q:", o["O2_PASS_q"])
for qs, rec in o["by_q"].items():
    for H in ("H32", "H128"):
        st = rec.get(H)
        if st is None:
            print(f"q={qs} {H}: no data")
            continue
        print(f"q={qs} {H}: medRstream={st['median_R_stream_max']:.4f} "
              f"medRrebuild={st['median_R_rebuild']:.4f} "
              f"maxCache={st['max_cache_gb_fp32']:.2f}GB "
              f"cells>24GB={st['n_cells_cache_gt_24gb']} "
              f"frozen={st['all_frozen_bitwise']} "
              f"O2_PASS={rec['O2_PASS']}")
EOF
