#!/usr/bin/env python3
import json, sys

# Check s22 Room baseline
paths = [
    "/home/liaoyuanjun/3dgs-renderer-benchmark/results/reference_v1/s22/room/baseline_30k.json",
    "/home/liaoyuanjun/3dgs-renderer-benchmark/results/reference_v1/s22/room/c42_30k.json",
]
for path in paths:
    print(f"\n=== {path.split('/')[-1]} ===")
    try:
        d = json.load(open(path))
        print("Keys:", list(d.keys()))
        if "results" in d:
            eps = d["results"].get("eval_points", [])
            fe = d["results"].get("final_eval")
            print(f"eval_points: {len(eps)}, final_eval: {fe}")
            for ep in eps:
                if ep["iter"] in [0, 5000, 10000, 15000, 20000, 25000, 30000]:
                    print(f"  iter={ep['iter']:>6d}: PSNR={ep['psnr']:.2f} SSIM={ep['ssim']:.4f} GS={ep.get('n_gaussians','?')}")
        elif "eval_points" in d:
            eps = d.get("eval_points", [])
            fe = d.get("final_eval")
            print(f"eval_points: {len(eps)}, final_eval: {fe}")
            for ep in eps:
                if ep["iter"] in [0, 5000, 10000, 15000, 20000, 25000, 30000]:
                    print(f"  iter={ep['iter']:>6d}: PSNR={ep['psnr']:.2f} SSIM={ep['ssim']:.4f} GS={ep.get('n_gaussians','?')}")
        elif "per_iter" in d:
            pi = d.get("per_iter", [])
            print(f"per_iter: {len(pi)} entries")
            for p in pi[:3]:
                print(f"  {p}")
    except Exception as e:
        print(f"  ERROR: {e}")
