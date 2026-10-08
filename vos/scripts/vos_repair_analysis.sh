#!/bin/bash
# Re-pair analysis: drift, twin envelope, and order-swapped A/B for room.
PY=/root/miniforge3/envs/gsplat/bin/python
cd /root/vos_traces
$PY - <<'EOF'
import numpy as np, json
def arm(d):
    z = np.load(f"vos_runs/{d}/vos_timing.npz")
    t, inner, vos = z["t"], z["inner_ms"], z["vos_ms"]
    reg = (t > 15000) & (t <= 30000)
    tim = json.load(open(f"vos_runs/{d}/timing.json"))
    res = json.load(open(f"vos_runs/{d}/results.json"))
    fe = res["final_eval"]
    return {"region_total": float(np.mean(inner[reg]+vos[reg])),
            "region_inner": float(np.mean(inner[reg])),
            "pre_inner": float(np.mean(inner[t <= 15000])),
            "wall": tim["total_wall_s"],
            "psnr": fe["psnr"], "ssim": fe["ssim"], "lpips": fe["lpips"],
            "N": res["final_N"]}
a_c0, a_vos = arm("c0_room_30k"), arm("vos_room_30k")
b_vos, b_c0 = arm("vos_room_30k_b"), arm("c0_room_30k_b")
print("order A: C0 then VOS")
print(f"  C0_a : wall={a_c0['wall']:.1f} pre={a_c0['pre_inner']:.3f} "
      f"reg={a_c0['region_total']:.3f} psnr={a_c0['psnr']:.4f} N={a_c0['N']}")
print(f"  VOS_a: wall={a_vos['wall']:.1f} pre={a_vos['pre_inner']:.3f} "
      f"reg={a_vos['region_total']:.3f} psnr={a_vos['psnr']:.4f} N={a_vos['N']}")
print("order B: VOS then C0")
print(f"  VOS_b: wall={b_vos['wall']:.1f} pre={b_vos['pre_inner']:.3f} "
      f"reg={b_vos['region_total']:.3f} psnr={b_vos['psnr']:.4f} N={b_vos['N']}")
print(f"  C0_b : wall={b_c0['wall']:.1f} pre={b_c0['pre_inner']:.3f} "
      f"reg={b_c0['region_total']:.3f} psnr={b_c0['psnr']:.4f} N={b_c0['N']}")
print(f"TWIN (C0_a vs C0_b): dWall={b_c0['wall']-a_c0['wall']:+.1f}s "
      f"dPre={b_c0['pre_inner']-a_c0['pre_inner']:+.4f}ms "
      f"dReg={b_c0['region_total']-a_c0['region_total']:+.4f}ms "
      f"dPSNR={b_c0['psnr']-a_c0['psnr']:+.4f}dB")
print(f"TWIN (VOS_a vs VOS_b): dWall={b_vos['wall']-a_vos['wall']:+.1f}s "
      f"dPSNR={b_vos['psnr']-a_vos['psnr']:+.4f}dB")
print(f"A/B order B: stage {b_c0['region_total']/b_vos['region_total']:.2f}x "
      f"wall {(b_c0['wall']-b_vos['wall'])/b_c0['wall']*100:+.2f}% "
      f"dPSNR={b_vos['psnr']-b_c0['psnr']:+.4f}dB")
EOF
