#!/bin/bash
cd /root/epic05_p11
/root/miniforge3/envs/gsplat/bin/python - <<'EOF'
import json
d = json.load(open('dumps/P11_LOSS_EQUIVALENCE.json'))
print("OVERALL:", d["overall_classification"])
print("CONSEQUENCE:", d["consequence"])
for s, p in d["provisional_classification"].items():
    print(f"{s}: class={p['classification']}")
    print(f"   worst_grad_rel_l2={p['worst_grad_rel_l2']:.3e} (thr {p['grad_threshold']:.3e})")
    print(f"   worst_loss_rel={p['worst_loss_rel_diff']:.3e} (thr {p['loss_threshold']:.3e})")
    print(f"   finite={p['worst_finite_fraction']}")
    print(f"   calibration: repeat={p['repeat_noise_rel_l2_max']:.2e} "
          f"sep64={p['sep_fp64_rel_l2_max']:.2e} fused64={p['fused_fp64_rel_l2_max']:.2e}")
    print(f"   clamp census: sp2<0={p.get('clamp_census', {}).get('total_sp2_negative', 'n/a')} "
          f"st2<0={p.get('clamp_census', {}).get('total_st2_negative', 'n/a')}")
EOF
