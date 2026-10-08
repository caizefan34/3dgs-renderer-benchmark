#!/bin/bash
cd /root/epic05_p11
/root/miniforge3/envs/gsplat/bin/python - <<'EOF'
import json
d = json.load(open('dumps/P11_ACTIVE_INTERACTION_AUDIT.json'))
BW = 1555e9
for scene, cells in d["per_scene"].items():
    for label, c in cells.items():
        m = c.get("compact_worklist_memory_lower_bound_bytes", {})
        exact = c["exact_contributing_interactions"]
        res_pct = c["conservative_residual_removable_pct_e2e"]
        titer = c.get("T_iter_ms")
        res_ms = res_pct/100*titer if titer else None
        # minimal build floor: write + read of a 4B-entry compact list
        floor4 = 2*exact*4/BW*1000
        floor8 = 2*exact*8/BW*1000
        mf = c.get("validation_mismatch_fraction_gt_1e-3")
        print(f"{scene}/{label}: exact={exact:,} res={res_pct:.2f}% "
              f"T_iter={titer:.1f}ms res_ms={res_ms:.2f}ms "
              f"floor4B={floor4:.2f}ms floor8B={floor8:.2f}ms "
              f"net4={res_ms-floor4:.2f}ms mismatch_frac={mf}")
        for k, v in m.items():
            print(f"    worklist[{k}] = {v/2**30:.2f} GiB")
EOF
