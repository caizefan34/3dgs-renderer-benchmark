import json, glob, os, statistics

def clean(x):
    try:
        return float(x)
    except Exception:
        return None

stats = {}
for f in sorted(glob.glob(r"results/training/*30k*.json")):
    r = json.load(open(f))
    cfg = r.get("config", {}) or {}
    scene = cfg.get("scene", "?")
    tile = cfg.get("tile_size", "?")
    key = (scene, tile)
    ml = r.get("metrics_log", [])
    if not ml:
        continue
    # accumulate per-stage sums across all logged iterations
    sums = {k: 0.0 for k in ["fwd_ms", "bwd_ms", "opt_ms", "topology_ms", "iteration_ms"]}
    for e in ml:
        for k in sums:
            v = clean(e.get(k))
            if v is not None:
                sums[k] += v
    # Also count number of entries with each stage timed
    counts = {k: sum(1 for e in ml if clean(e.get(k)) is not None) for k in sums}
    s = stats.setdefault(key, {})
    s["n_logged"] = len(ml)
    s["sums"] = sums
    s["counts"] = counts

print("=== 30k-run stage totals (ms over logged iterations) ===")
for (scene, tile) in sorted(stats):
    s = stats[(scene, tile)]
    sums = s["sums"]
    acc = sums["fwd_ms"] + sums["bwd_ms"] + sums["opt_ms"] + sums["topology_ms"]
    print("%-10s tile=%-3d n_log=%d  fwd=%8.1f bwd=%8.1f opt=%8.1f topo=%8.1f | acc=%9.1f iter=%9.1f | gap=%9.1f acc%%=%.1f" %
          (scene, tile, s["n_logged"], sums["fwd_ms"], sums["bwd_ms"], sums["opt_ms"], sums["topology_ms"],
           acc, sums["iteration_ms"], sums["iteration_ms"] - acc,
           100.0 * acc / sums["iteration_ms"] if sums["iteration_ms"] else 0))

# Per-entry statistics for the first logged tile of each scene
print()
print("=== per-entry stage stats (median), from 30k runs ===")
for (scene, tile) in sorted(stats):
    f = [i for i in [glob.glob(rf"results/training/*30k_{scene}_t{tile}*.json")] if i]
    if not f:
        continue
    r = json.load(open(f[0][0]))
    ml = r["metrics_log"]
    stages = ["fwd_ms", "bwd_ms", "opt_ms", "topology_ms", "iteration_ms"]
    med = {}
    for k in stages:
        vals = sorted(clean(e[k]) for e in ml if clean(e[k]) is not None)
        med[k] = vals[len(vals)//2] if vals else float("nan")
    print("  %-10s tile=%d: fwd=%.2f bwd=%.2f opt=%.2f topo=%.4f iter=%.2f" %
          (scene, tile, med["fwd_ms"], med["bwd_ms"], med["opt_ms"], med["topology_ms"], med["iteration_ms"]))
