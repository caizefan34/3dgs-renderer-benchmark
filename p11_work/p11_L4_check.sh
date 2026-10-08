#!/bin/bash
cd /root/epic05_p11
/root/miniforge3/envs/gsplat/bin/python - <<'EOF'
import json
d = json.load(open('dumps/P11_LOSS_KERNEL_PROFILE.json'))
for scene, rec in d["per_scene"].items():
    b = rec["fusion_bounds_fused_baseline"]
    print(f"=== {scene} ===")
    print(f"  loss-only GPU ms/iter: sep={rec['loss_only_sepssim']['total_gpu_ms_per_iter']:.3f} "
          f"fused={rec['loss_only_fused']['total_gpu_ms_per_iter']:.3f}")
    print(f"  bounds on fused baseline (% of T_iter={b['T_iter_ms']:.2f}ms): "
          f"cons={b['conservative_pct_titer']:.2f}% "
          f"mod={b['moderate_pct_titer']:.2f}% "
          f"opt={b['optimistic_pct_titer']:.2f}%")
    print(f"  not counted: {b.get('not_counted', 'see json')}")
    fs = rec["full_step_fused"]
    print(f"  full-step fused kernel categories (ms/iter):")
    cats = fs.get("categories_ms_per_iter", fs.get("kernel_categories", {}))
    for k, v in sorted(cats.items(), key=lambda kv: -kv[1] if isinstance(kv[1], (int, float)) else 0)[:8]:
        print(f"    {k}: {v}")
EOF
echo "=== D1 progress ==="
tail -n 8 logs/D1.log 2>/dev/null
echo "=== resume ==="
tail -n 3 logs/resume.log
