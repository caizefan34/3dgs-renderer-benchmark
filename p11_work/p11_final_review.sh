#!/bin/bash
cd /root/epic05_p11
/root/miniforge3/envs/gsplat/bin/python - <<'EOF'
import json
d = json.load(open('dumps/P11_L3_CLOSURE_STREAMING.json'))
print("=== STREAMING ADDITIVE CLOSURE (official) ===")
for scene, rec in d["per_scene"].items():
    print(f"{scene}: rbwd={rec['renderer_backward_external']['median_ms']:.2f} "
          f"lbwd={rec['loss_to_image_backward']['median_ms']:.2f} "
          f"sum={rec['rbwd_plus_lbwd_ms']:.2f} full={rec['full_backward']['median_ms']:.2f} "
          f"gap={rec['closure_gap_pct']:.2f}%")
EOF
echo
echo "=== REPORT (head 60) ==="
head -n 60 artifacts/P11_REPORT.md
