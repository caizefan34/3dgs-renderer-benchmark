import json

d = json.load(open('/mnt/storage_pool/liaoyuanjun/strong_baseline_results/faster-gs/all_metrics.json'))
for k, v in d.items():
    print(f"{k:18s} wall={v['wall_time_min']:6.2f}min PSNR={v['PSNR']:6.2f} "
          f"SSIM={v['SSIM']:.3f} LPIPS={v['LPIPS']:.3f} N={v['n_gaussians']:,} rc={v['exit_code']}")
print("===")
import os
for b in ("fastgs", "speedy-splat"):
    p = f"/mnt/storage_pool/liaoyuanjun/strong_baseline_results/{b}"
    for root, dirs, files in os.walk(p):
        for f in files:
            if f.endswith(".json"):
                print(os.path.join(root, f))
print("===")
# provenance: run dirs with logs
for b in ("faster-gs", "fastgs", "speedy-splat"):
    p = f"/mnt/storage_pool/liaoyuanjun/strong_baseline_results/{b}"
    for root, dirs, files in os.walk(p):
        if files and root.count(os.sep) < 8:
            print(root, "->", sorted(files)[:6])
