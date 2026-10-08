#!/bin/bash
# Full final A/B across all scenes and both orders + twin spreads.
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
            "ssim": fe["ssim"], "lpips": fe["lpips"], "N": res["final_N"],
            "mean_iter": tim["mean_iter_ms"]}
for s in ["room", "bicycle", "garden"]:
    a_c0, a_vos = arm(f"c0_{s}_30k"), arm(f"vos_{s}_30k")
    b_vos, b_c0 = arm(f"vos_{s}_30k_b"), arm(f"c0_{s}_30k_b")
    print(f"===== {s} =====")
    for tag, x in [("C0_a", a_c0), ("VOS_a", a_vos),
                   ("VOS_b", b_vos), ("C0_b", b_c0)]:
        print(f"  {tag}: wall={x['wall']:.1f} pre={x['pre']:.3f} "
              f"reg={x['reg']:.3f} psnr={x['psnr']:.4f} "
              f"ssim={x['ssim']:.5f} lpips={x['lpips']:.5f} N={x['N']}")
    for tag, c0, vo in [("A", a_c0, a_vos), ("B", b_c0, b_vos)]:
        sp = c0["reg"] / vo["reg"]
        wl = (c0["wall"] - vo["wall"]) / c0["wall"] * 100
        dp = vo["psnr"] - c0["psnr"]
        shr = (c0["reg"] - vo["reg"]) / c0["mean_iter"] * 100
        print(f"  order-{tag}: stage={sp:.2f}x wall={wl:+.2f}% "
              f"region_share={shr:.1f}% dPSNR={dp:+.4f}")
    print(f"  twins: C0 dPSNR={b_c0['psnr']-a_c0['psnr']:+.4f} "
          f"VOS dPSNR={b_vos['psnr']-a_vos['psnr']:+.4f} "
          f"C0 dWall={b_c0['wall']-a_c0['wall']:+.1f}s")
EOF
