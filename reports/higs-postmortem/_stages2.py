import json, glob, os, statistics
from collections import defaultdict

def clean(x):
    try:
        v = float(x)
        return v if v > 0 else None
    except Exception:
        return None

# Log files summary
out = {}
for f in sorted(glob.glob(r"results/training/*30k*.json")):
    r = json.load(open(f))
    cfg = r.get("config", {}) or {}
    scene = cfg.get("scene", "?")
    tile = cfg.get("tile_size", "?")
    ml = r.get("metrics_log", [])
    sums = {k: 0.0 for k in ["fwd_ms", "bwd_ms", "opt_ms", "topology_ms", "iteration_ms"]}
    counts = dict(sums)
    for e in ml:
        for k in sums:
            v = clean(e.get(k))
            if v is not None:
                sums[k] += v
                counts[k] += 1
    acc = sums["fwd_ms"] + sums["bwd_ms"] + sums["opt_ms"] + sums["topology_ms"]
    print("%-8s t%-3d logged=%d iter_sum=%8.1fs acc_sum=%7.1fs accshare=%.1f%% fwd=%6.1fs(%4.1f) bwd=%6.1fs(%4.1f) opt=%6.1fs(%4.1f) topo=%6.1fs(%4.1f)" %
          (scene, tile, len(ml), sums["iteration_ms"]/1000.0, acc/1000.0,
           100.0*acc/sums["iteration_ms"] if sums["iteration_ms"] else 0,
           sums["fwd_ms"]/1000.0, 100.0*sums["fwd_ms"]/sums["iteration_ms"] if sums["iteration_ms"] else 0,
           sums["bwd_ms"]/1000.0, 100.0*sums["bwd_ms"]/sums["iteration_ms"] if sums["iteration_ms"] else 0,
           sums["opt_ms"]/1000.0, 100.0*sums["opt_ms"]/sums["iteration_ms"] if sums["iteration_ms"] else 0,
           sums["topology_ms"]/1000.0, 100.0*sums["topology_ms"]/sums["iteration_ms"] if sums["iteration_ms"] else 0))

# Now 500-step logs
print()
for f in sorted(glob.glob(r"results/training/*500step*.json")):
    r = json.load(open(f))
    cfg = r.get("config", {}) or {}
    scene = cfg.get("scene", "?")
    tile = cfg.get("tile_size", "?")
    ml = r.get("metrics_log", [])
    sums = {k: 0.0 for k in ["fwd_ms", "bwd_ms", "opt_ms", "topology_ms", "iteration_ms"]}
    counts = dict(sums)
    for e in ml:
        for k in sums:
            v = clean(e.get(k))
            if v is not None:
                sums[k] += v
                counts[k] += 1
    acc = sums["fwd_ms"] + sums["bwd_ms"] + sums["opt_ms"] + sums["topology_ms"]
    print("%-10s t%-3d logged=%d iter=%.1fms acc=%.1fms accshare=%.1f%%" %
          (scene, tile, len(ml), sums["iteration_ms"]/len(ml) if len(ml) else 0,
           acc/len(ml) if len(ml) else 0,
           100.0*acc/sums["iteration_ms"] if sums["iteration_ms"] else 0))
