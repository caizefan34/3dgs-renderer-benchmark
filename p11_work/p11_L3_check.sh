#!/bin/bash
cd /root/epic05_p11
/root/miniforge3/envs/gsplat/bin/python - <<'EOF'
import json
d = json.load(open('dumps/P11_L3_RENDERER_BWD.json'))
print("=== L3 per scene/state ===")
for scene, srec in d["per_scene"].items():
    for label, cell in srec.items():
        s = cell["state_summary"]
        t = cell["titer"]
        print(f"{scene}/{label}: fwd={s['renderer_forward_ms']:.2f} "
              f"rbwd={s['renderer_backward_external_ms']:.2f} "
              f"lbwd={s['loss_to_image_bwd_ms']:.2f} "
              f"full={s['full_backward_ms']:.2f} "
              f"gap={s['mean_abs_closure_gap_pct']:.2f}%")
        print(f"    T_iter={t['T_iter_ms']:.2f} (fwd {t['forward']:.2f} loss {t['loss']:.2f} "
              f"bwd {t['backward']:.2f} dens {t['densify']:.2f} opt {t['optimizer']:.2f})")
EOF
