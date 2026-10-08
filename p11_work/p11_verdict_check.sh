#!/bin/bash
cd /root/epic05_p11
/root/miniforge3/envs/gsplat/bin/python - <<'EOF'
import json
v = json.load(open('artifacts/P11_VERDICT.json'))
print("DECISION:", v["decision"])
print("NEXT ACTION:", v["next_action"])
print("SELECTED:", v["selected_candidate"])
print()
print("REASON:")
print(v["reason"])
print()
m = json.load(open('artifacts/P11_CANDIDATE_MATRIX.json'))
print("=== CANDIDATE DECISIONS ===")
for c in m["candidates"]:
    print(f"  {c['name']}: cons={c['conservative_removable_pct']} "
          f"mod={c['moderate_removable_pct']} -> {c['decision']}")
    fr = c.get("final_rule_check", {})
    print(f"      final_rule: {fr}")
EOF
