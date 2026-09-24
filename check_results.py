#!/usr/bin/env python3
import json, sys
for name in sys.argv[1:]:
    path = f"/home/liaoyuanjun/3dgs-renderer-benchmark/results/c42_adaptive/completion_batch/{name}.json"
    try:
        with open(path) as f:
            d = json.load(f)
        fe = d["results"]["final_eval"]
        t = d["results"]["timing"]
        prov = d.get("provenance", {})
        print(f"{name}:")
        print(f"  PSNR={fe['psnr']:.4f} SSIM={fe['ssim']:.4f} LPIPS={fe.get('lpips','N/A')}")
        print(f"  GS={fe['n_gaussians']:,} cams={fe['n_eval_cameras']}")
        print(f"  wall={t['total_wall_min']:.1f}min mean_iter={t['mean_iter_ms']:.2f}ms steady={t['steady_state_mean_iter_ms']:.2f}ms")
        print(f"  GPU={prov.get('gpu_name','?')} gsplat={prov.get('gsplat_version','?')} seed={d['results']['seed']}")
        # Show eval trajectory at key points
        for ep in d["results"]["eval_points"]:
            if ep["iter"] in [0, 5000, 10000, 15000, 20000, 25000, 30000]:
                lpips_str = f" LPIPS={ep.get('lpips','?')}"
                if isinstance(ep.get('lpips'), float):
                    lpips_str = f" LPIPS={ep['lpips']:.4f}"
                print(f"  iter={ep['iter']:>6d}: PSNR={ep['psnr']:.2f} SSIM={ep['ssim']:.4f}{lpips_str} GS={ep['n_gaussians']:,}")
    except Exception as e:
        print(f"{name}: ERROR - {e}")
