#!/usr/bin/env python3
import json
for scene in ["bicycle", "garden", "room"]:
    path = f"/mnt/storage_pool/liaoyuanjun/r4_13scene_v2/{scene}/candidate_c/training_metrics.json"
    try:
        d = json.load(open(path))
        ckpts = d.get("checkpoints", {})
        print(f"\n=== {scene} candidate_c ===")
        for k in sorted(ckpts.keys(), key=int):
            v = ckpts[k]
            print(f"  iter {k}: PSNR={v['psnr']:.2f} SSIM={v['ssim']:.4f} N={v['N_gaussians']}")
    except Exception as e:
        print(f"\n{scene}: {e}")
