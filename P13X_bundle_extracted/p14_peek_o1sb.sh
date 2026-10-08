#!/bin/bash
/root/miniforge3/envs/gsplat/bin/python - <<'EOF'
import json
a = json.load(open('/root/epic05_p14/artifacts/P14_O1_AGG.json'))
print("O1_PASS_by_q:", json.dumps(a["O1_PASS_by_q"], indent=1))
sb = a.get("stage_B")
if sb:
    for rec in sb.get("per_q", []):
        print(f"\nq={rec['q']}: n_viable={rec['n_viable']}/6 "
              f"median R_train={rec['median_R_train']:.4f}")
        for cs, v in rec["per_cell"].items():
            print(f"  {cs}: H={v.get('H')} R_train={v['R_train']:.4f} "
                  f"P_audit={v['P_audit']:.4f} viable={v['viable']}")
EOF
