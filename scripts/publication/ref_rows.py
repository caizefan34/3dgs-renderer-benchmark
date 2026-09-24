import json
import math

M = json.load(open("/mnt/storage_pool/liaoyuanjun/pubphase/aggregates/runs_master.json"))

for v, e in (("b1a", 0.1), ("c0", 0.3)):
    runs = [r for r in M["runs"] if r["variant"] == v and r["seed"] == 42
            and r["eps2d"] == e and not r.get("contaminated")]
    walls = [r["wall_s"] / 60 for r in runs]
    psnrs = [r["psnr"] for r in runs]
    ns = [r["n_gaussians"] for r in runs]
    print(f"{v}: n={len(runs)} geomean_wall={math.exp(sum(math.log(x) for x in walls)/len(walls)):.1f}min "
          f"mean_PSNR={sum(psnrs)/len(psnrs):.2f} N_range={min(ns):,}-{max(ns):,}")
