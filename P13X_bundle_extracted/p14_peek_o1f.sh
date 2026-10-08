#!/bin/bash
/root/miniforge3/envs/gsplat/bin/python - <<'EOF'
import json
a = json.load(open('/root/epic05_p14/artifacts/P14_O1_AGG.json'))
print("stage_B_allowed:", a.get("stage_B_allowed"))
for qs, rec in a["by_q"].items():
    print(f"\nq={qs}: O1_PASS={rec.get('O1_PASS')}")
    for H in ("H8", "H32", "H128"):
        st = rec.get(H)
        if st is None:
            print(f"  {H}: no data")
            continue
        print(f"  {H}: median R_train={st['median_R_train']:.4f} "
              f"n_viable={st['n_viable']}/6 "
              f"P_audit_max={st['max_P_audit']:.4f} "
              f"all_frozen={st['all_frozen_bitwise']}")
EOF
