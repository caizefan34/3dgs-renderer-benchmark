import json, glob, os, statistics, math
from collections import defaultdict

def md(v): return "%.1f" % v if not isinstance(v, str) else v
def clean(x):
    if isinstance(x, str):
        return float(x) if x.replace('.','',1).replace('-','',1).isdigit() else float('nan')
    try: return float(x)
    except Exception: return float('nan')

def load_grid(d):
    rows = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        r = json.load(open(f))
        rows[(r.get("method"), r.get("scene"), r.get("seed"))] = r
    return rows

def load_logs(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        r = json.load(open(f))
        cfg = r.get("config", {})
        scene = cfg.get("scene", "?")
        tile = cfg.get("tile_size", "?")
        out[(scene, tile)] = r
    return out

# 1) grid summary
for label, d in [("PAPER", r"artifacts\training-paper\results")]:
    g = load_grid(d)
    by = defaultdict(list)
    for (m, s, seed), r in g.items():
        by[m].append(r)
    print("grid:", label, "| methods:", sorted(by.keys()))
    for m in sorted(by):
        rs = by[m]
        walls = [clean(r["performance"]["wall_time_seconds"]) for r in rs]
        ttqs = [clean(r["performance"]["time_to_quality_seconds"]) for r in rs]
        for metric, vals in [("wall", walls), ("ttq", ttqs)]:
            vals = [v for v in vals if math.isfinite(v)]
            gmean = math.exp(sum(math.log(v) for v in vals)/len(vals)) if vals else float('nan')
            print("  %-16s %-4s mean=%.1f gmean=%.1f" % (m, metric, statistics.mean(vals), gmean))
    # per-scene wall
    sc = defaultdict(dict)
    for (m, s, seed), r in g.items():
        sc[s][m] = clean(r["performance"]["wall_time_seconds"])
    print("\n  per-scene wall (mean over seeds):")
    for s, d2 in sorted(sc.items()):
        line = "  %-28s" % s
        for m in sorted(d2):
            line += "  %s=%.1f" % (m, d2[m])
        print(line)

# 2) training logs
logs = load_logs(r"results\training")
print("\ntraining logs:", sorted(logs.keys()))
for key, r in sorted(logs.items()):
    scene, tile = key
    ml = r.get("metrics_log", [])
    if not ml:
        continue
    fields = ["fwd_ms", "bwd_ms", "opt_ms", "topology_ms", "iteration_ms"]
    agg = {}
    for fld in fields:
        vals = [clean(e[fld]) for e in ml if fld in e and clean(e[fld]) > 0]
        if vals:
            agg[fld] = (statistics.mean(vals), statistics.median(vals))
    print("  scene=%s tile=%s n=%d" % (scene, tile, len(ml)))
    for fld in fields:
        if fld in agg:
            print("      %-14s mean=%.3f med=%.3f" % (fld, agg[fld][0], agg[fld][1]))

# 3) compute ratio gslam/higs_proposed per scene
print("\nper-scene wall ratio and delta quality:")
by2 = defaultdict(lambda: defaultdict(dict))
for (m, s, seed), r in g.items():
    by2[s][m][seed] = (clean(r["performance"]["wall_time_seconds"]), clean(r["quality"]["psnr_db"]))
for s in sorted(by2):
    out = []
    for m in sorted(by2[s]):
        vals = by2[s][m]
        w = statistics.mean([v[0] for v in vals.values()]) if vals else float('nan')
        p = statistics.mean([v[1] for v in vals.values()]) if vals else float('nan')
        out.append((m, w, p))
    line = "  %-28s" % s
    for m, w, p in out:
        line += "  %s(%.0fs,%.2f)" % (m[:8], w, p)
    print(line)
