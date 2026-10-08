#!/bin/bash
/root/miniforge3/envs/gsplat/bin/python - <<'EOF'
import json
A = "/root/epic05_n1/artifacts"
h = json.load(open(f"{A}/N1_ACTION_HETEROGENEITY.json"))
f = json.load(open(f"{A}/N1_BUDGET_FRONTIER.json"))
for s in ("room", "bicycle"):
    hs = h["scenes"][s]
    print(f"=== {s}: H1={hs['H1_skip_share']:.4f} (gate {hs['H1_gate_20pct_pass']}) "
          f"H2={hs['H2_share']:.4f} (gate {hs['H2_gate_15pct_pass']}) "
          f"H3={hs['H3_prefer_share']} gates={hs['H3_gates']}")
    print(f"    tiers(best)={hs['A2_best_tier_histogram']} "
          f"secondary_H1={hs['H1_secondary_share_sigma_onestep']:.4f}")
    for k, v in f["scenes"][s].items():
        sg = v.get("strong_gate_pass")
        print(f"    {k}: prog={v['assignment_progress']:.6f} "
              f"allA1={v['all_A1_progress']:.6f} fixed01={v['baseline_fixed01_progress']:.6f} "
              f"oracle01={v['baseline_oracle01_control_progress']:.6f} "
              f"ratio_maxctrl={v['assignment_over_max_control']:.4f} "
              f"shares={ {k2: (round(x,3) if x else x) for k2, x in v['action_shares'].items()} }"
              + (f" STRONG={sg}" if sg is not None else ""))
# bicycle per-ckpt heterogeneity detail
for ck, v in h["scenes"]["bicycle"]["per_checkpoint"].items():
    print(f"bicycle@{ck}: medA1={v['median_dA1']:.2e} sign={v['sign_rate_dA1']:.3f} "
          f"medA2r={v['median_dA2_random']:.2e} sigma1step={v['sigma_onestep']:.2e}")
# timing table
for s in ("room", "bicycle"):
    t = json.load(open(f"{A}/N1_ACTION_TIMING_{s}.json"))
    for cell, v in sorted(t["cells"].items()):
        print(f"{s} {cell}: median={v['total_wall_ms']['median']:.3f}ms "
              f"phases={ {k: round(p['median'],2) for k,p in v['phases_ms'].items()} }")
EOF
