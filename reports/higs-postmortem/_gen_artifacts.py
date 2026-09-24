#!/usr/bin/env python3
"""Generate artifacts/higs-trainable-regression/*.{json,csv}"""
import json, glob, os, statistics, math
from collections import defaultdict

def clean(x):
    if isinstance(x, str):
        try: return float(x)
        except Exception: return None
    try: return float(x)
    except Exception: return None

def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None

def gmean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs)/len(xs)) if xs else None

def load(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        out[(r.get("method"), r.get("scene"), r.get("seed"))] = r
    return out

G = load(r"artifacts\training-paper\results")
methods = sorted({m for (m,_,_) in G})
scenes  = sorted({s for (_,s,_) in G})

# ---- master csv ----
rows = []
for (m, s, seed), r in sorted(G.items()):
    P = r.get("performance", {})
    Q = r.get("quality", {})
    R = r.get("resources", {})
    rows.append({
        "method": m, "scene": s, "seed": seed,
        "wall_s": clean(P.get("wall_time_seconds")),
        "ttq_s":  clean(P.get("time_to_second", P.get("time_to_quality_seconds"))),
        "psnr":   clean(Q.get("psnr_df", Q.get("psnr_db"))),
        "ssim":   clean(Q.get("ssim")),
        "lpips":  clean(Q.get("lpips")),
        "gauss":  clean(R.get("final_gaussian_count")),
        "peak_mib": clean(R.get("peak_gpu_memory_mib", R.get("peak_memory_mib"))),
    })
with open(r"artifacts\higs-trainable-regression\master_results.csv","w") as f:
    import csv
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# ---- scene-level means & speedups ----
def agg_by_scene(m):
    sc = defaultdict(list)
    for r in rows:
        if r["method"] == m:
            sc[r["scene"]].append(r)
    return {s: {k: mean([r[k] for r in rs]) for k in ["wall_s","ttq_s","psnr","ssim","gauss","peak_mib"]} for s, rs in sc.items()}

agg_g = agg_by_scene("gsplat")
out = []
for s in scenes:
    d = {"scene": s, "gsplat_wall_s": agg_g[s]["wall_s"], "gsplat_ttq_s": agg_g[s]["ttq_s"]}
    for m in [mm for mm in methods if mm != "gsplat"]:
        a = agg_by_scene(m)[s]
        d[m+"_wall_s"] = a["wall_s"]
        d[m+"_ttq_s"]  = a["ttq_s"]
        d[m+"_wall_ratio"] = round(agg_g[s]["wall_s"]/a["wall_s"], 4)
    out.append(d)
with open(r"artifacts\higs-trainable-regression\scene_speedup.csv","w") as f:
    import csv
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader()
    w.writerows(out)

# ---- stage timing from logs ----
logs = {}
for f in sorted(glob.glob(r"results/training/*.json")):
    r = json.load(open(f))
    cfg = r.get("config", {}) or {}
    logs[(cfg.get("scene"), cfg.get("tile_size"))] = r

stage_rows = []
stage_by_scene = defaultdict(lambda: defaultdict(list))
for (scene, tile), r in logs.items():
    ml = r.get("metrics_log", [])
    if not ml:
        continue
    row = {"scene": scene, "tile": tile}
    for k in ["fwd_ms", "bwd_ms", "opt_ms", "topology_ms", "iteration_ms"]:
        vals = [clean(e.get(k)) for e in ml]
        vals = [v for v in vals if v is not None]
        if vals:
            row[k] = round(mean(vals), 3)
            stage_by_scene[scene][k].append(mean(vals))
    stage_rows.append(row)

with open(r"artifacts\higs-trainable-regression\stage_timing_by_tile.json","w") as f:
    json.dump(stage_rows, f, indent=2)

summary = {}
for scene in sorted(stage_by_scene):
    d = {k: round(mean(v), 3) for k, v in stage_by_scene[scene].items()}
    # share of accounted time
    acc = sum(d.get(k, 0) for k in ["fwd_ms", "bwd_ms", "opt_ms", "topology_ms"])
    tot = d.get("iteration_ms", 0)
    d["accounted_ms"] = round(acc, 3)
    d["accounted_share"] = round(100.0 * acc / tot, 1) if tot else None
    d["gap_ms"] = round(max(tot - acc, 0), 3) if tot else None
    d["gap_share"] = round(100.0 * max(tot - acc, 0) / tot, 1) if tot else None
    d["fwd_share"] = round(100.0 * d.get("fwd_ms", 0) / tot, 1) if tot else None
    d["bwd_share"] = round(100.0 * d.get("bwd_ms", 0) / tot, 1) if tot else None
    summary[scene] = d
with open(r"artifacts\higs-trainable-regression\stage_timing_summary.json","w") as f:
    json.dump(summary, f, indent=2)

print("stage summary:")
for s, d in summary.items():
    print(" ", s, d)

# ---- break-even & sensitivity ----
# Per-scene model: use per-iteration wall from logs? Use first file per scene.
# Use grid: mean wall per method across all (scene, seed) -> geomeans
import collections
geomean = {}
for m in methods:
    ws = [r["wall_s"] for r in rows if r["method"] == m and r["wall_s"]]
    geomean[m] = math.exp(sum(math.log(w) for w in ws)/len(ws))
print("\nwall geomeans:", {k: round(v,1) for k,v in geomean.items()})
for m in methods:
    if m != "gsplat":
        print("geomean gain (gsplat/hid): %.4fx" % (geomean["gsplat"]/geomean[m]))

# Break-even logic (analytic):
# Let T_total = 1.0 (normalized). Stage model unaccounted G.
# For a stage cost S and gain fraction g, speedup factor = 1/(1 - S*g).
# If we remove the whole "gap" (unaccounted) we recover G.
# If we halve backward (bwd share b), we recover 0.5b.
# Compute for each scene from stage summary.
print("\nstage shares by scene (of total iteration):")
for s, d in summary.items():
    tot = d["iteration_ms"]
    f_b = 100.0 * (d.get("bwd_ms",0)) / tot
    f_o = 100.0 * (d.get("opt_ms",0)) / tot
    f_t = 100.0 * (d.get("topology_ms",0)) / tot
    f_f = 100.0 * (d.get("fwd_ms",0)) / tot
    f_g = 100.0 * (d.get("gap_ms",0)) / tot
    print("  %s: fwd=%.1f%% bwd=%.1f%% opt=%.1f%% topo=%.1f%% gap=%.1f%%" % (s, f_f, f_b, f_o, f_t, f_g))

# Sensitivity table: projected total-time reduction with bwd cut
print("\nbackward sensitivity (projected % reduction of iteration time):")
for s, d in summary.items():
    tot = d["iteration_ms"]
    b = 100.0 * d.get("bwd_ms", 0) / tot
    print("  %s: bwd share=%.1f%%; -10%% bwd -> %.2f%% total; -20%% bwd -> %.2f%% total" % (s, b, 0.1*b, 0.2*b))

with open(r"artifacts\higs-trainable-regression\break_even_analysis.json","w") as f:
    json.dump({
        "wall_geomean_seconds": geomean,
        "method_speedups": {m: geomean["gsplat"]/geomean[m] for m in methods if m != "gsplat"},
    }, f, indent=2)
