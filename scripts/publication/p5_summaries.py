import json
import math

ALL13 = ["bicycle", "bonsai", "counter", "drjohnson", "flowers", "garden",
         "kitchen", "playroom", "room", "stump", "train", "treehill", "truck"]

for b in ("fastgs", "speedy-splat"):
    d = json.load(open(f"/mnt/storage_pool/liaoyuanjun/strong_baseline_results/{b}/all_metrics.json"))
    print(f"== {b} ==")
    walls, psnrs = [], []
    for s in ALL13:
        n = d.get(f"{s}_native")
        m = d.get(f"{s}_c42")
        if not (n and m):
            continue
        walls.append(m["wall_time_min"])
        psnrs.append(m["PSNR"])
        flag = " <-- low" if m["PSNR"] < 20 else ""
        print(f"  {s:10s} c42: wall={m['wall_time_min']:5.1f}m PSNR={m['PSNR']:6.2f} "
              f"SSIM={m['SSIM']:.3f} N={m['n_gaussians']:,}{flag}")
    print(f"  c42 geomean wall: {math.exp(sum(math.log(x) for x in walls)/len(walls)):.1f}m, "
          f"mean PSNR: {sum(psnrs)/len(psnrs):.2f}")
