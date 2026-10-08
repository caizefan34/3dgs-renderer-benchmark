#!/bin/bash
# Bicycle re-pair A/B on GPU 3 (MR0 CPU-load symmetric, GPU exclusive).
PY=/root/miniforge3/envs/gsplat/bin/python
cd /root/vos_traces
$PY - <<'EOF'
import numpy as np, json
def arm(d):
    z = np.load(f"vos_runs/{d}/vos_timing.npz")
    t, inner, vos = z["t"], z["inner_ms"], z["vos_ms"]
    reg = (t > 15000) & (t <= 30000)
    pre = t <= 15000
    tim = json.load(open(f"vos_runs/{d}/timing.json"))
    res = json.load(open(f"vos_runs/{d}/results.json"))
    fe = res["final_eval"]
    return {"reg": float(np.mean(inner[reg]+vos[reg])),
            "pre": float(np.mean(inner[pre])),
            "wall": tim["total_wall_s"], "psnr": fe["psnr"],
            "ssim": fe["ssim"], "lpips": fe["lpips"], "N": res["final_N"]}
for s in ["bicycle"]:
    a_c0, a_vos = arm(f"c0_{s}_30k"), arm(f"vos_{s}_30k")
    b_vos, b_c0 = arm(f"vos_{s}_30k_b"), arm(f"c0_{s}_30k_b")
    print(f"== {s} order-A (3-way co-tenant) ==")
    print(f"  C0_a : wall={a_c0['wall']:.1f} pre={a_c0['pre']:.3f} reg={a_c0['reg']:.3f} psnr={a_c0['psnr']:.4f}")
    print(f"  VOS_a: wall={a_vos['wall']:.1f} pre={a_vos['pre']:.3f} reg={a_vos['reg']:.3f} psnr={a_vos['psnr']:.4f}")
    print(f"  -> stage {a_c0['reg']/a_vos['reg']:.2f}x wall {(a_c0['wall']-a_vos['wall'])/a_c0['wall']*100:+.2f}%")
    print(f"== {s} order-B (GPU3-exclusive, MR0 CPU load) ==")
    print(f"  VOS_b: wall={b_vos['wall']:.1f} pre={b_vos['pre']:.3f} reg={b_vos['reg']:.3f} psnr={b_vos['psnr']:.4f}")
    print(f"  C0_b : wall={b_c0['wall']:.1f} pre={b_c0['pre']:.3f} reg={b_c0['reg']:.3f} psnr={b_c0['psnr']:.4f}")
    print(f"  -> stage {b_c0['reg']/b_vos['reg']:.2f}x wall {(b_c0['wall']-b_vos['wall'])/b_c0['wall']*100:+.2f}%")
    print(f"  twins: C0 dPSNR={b_c0['psnr']-a_c0['psnr']:+.4f} VOS dPSNR={b_vos['psnr']-a_vos['psnr']:+.4f}")
    print(f"  order-B d_psnr(VOS-C0)={b_vos['psnr']-b_c0['psnr']:+.4f} "
          f"d_ssim={b_vos['ssim']-b_c0['ssim']:+.5f} d_lpips={b_vos['lpips']-b_c0['lpips']:+.5f}")
EOF
