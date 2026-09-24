#!/usr/bin/env python3
import json, sys

path = sys.argv[1] if len(sys.argv) > 1 else "/home/liaoyuanjun/3dgs-renderer-benchmark/results/a100/phase-c42/c42_p2_training_validation_30k.json"
d = json.load(open(path))
print("Keys:", list(d.keys()))
for vk in ["variant_A_baseline", "variant_B_downsampled_0.75"]:
    if vk in d:
        v = d[vk]
        fe = v.get("final_eval", {})
        print(f"\n{vk}:")
        print(f"  final: PSNR={fe.get('psnr','?')} SSIM={fe.get('ssim','?')} GS={fe.get('n_gaussians','?')}")
        for ep in v.get("eval_points", []):
            if ep["iter"] in [0, 5000, 10000, 15000, 20000, 25000, 30000]:
                gs = ep.get("n_gaussians", "?")
                print(f"  iter={ep['iter']:>6d}: PSNR={ep['psnr']:.2f} SSIM={ep['ssim']:.4f} GS={gs}")
