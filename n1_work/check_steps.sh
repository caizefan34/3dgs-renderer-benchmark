#!/bin/bash
PY=/root/miniforge3/envs/gsplat/bin/python
$PY - <<'EOF'
import json
for sc in ("room", "bicycle"):
    d = json.load(open(f"/root/epic05_n1/artifacts/N1_CHECKPOINT_IDENTITY_{sc}.json"))
    print(f"=== {sc}")
    for it, rec in d["checkpoints"].items():
        print(f"ckpt {it}: steps={rec['adam_step_values']} N={rec['N']:,} "
              f"restore={rec['optimizer_state_restorable_exactly']} "
              f"switchback={rec.get('switchback_heldout_bit_exact')} "
              f"inv={rec['restore_render_invariance_bit_exact']}")
EOF
