import json
import math

ALL13 = ["bicycle", "bonsai", "counter", "drjohnson", "flowers", "garden",
         "kitchen", "playroom", "room", "stump", "train", "treehill", "truck"]

for b in ("faster-gs", "fastgs", "speedy-splat"):
    d = json.load(open(f"/mnt/storage_pool/liaoyuanjun/strong_baseline_results/{b}/all_metrics.json"))
    walls, psnrs, ns, missing = [], [], [], []
    for s in ALL13:
        m = d.get(f"{s}_c42")
        if not m:
            missing.append(s)
            continue
        if m["wall_time_min"] > 0.01:
            walls.append(m["wall_time_min"])
        else:
            missing.append(f"{s}(wall)")
        psnrs.append(m["PSNR"])
        ns.append(m["n_gaussians"])
    gm = math.exp(sum(math.log(x) for x in walls) / len(walls)) if walls else 0
    print(f"{b}: c42 geomean_wall={gm:.1f}min (n={len(walls)}) mean_PSNR={sum(psnrs)/len(psnrs):.2f} "
          f"N_range={min(ns):,}-{max(ns):,} missing={missing}")
