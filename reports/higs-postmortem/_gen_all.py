#!/usr/bin/env python3
"""Generate all postmortem artifacts."""
import json, glob, os, statistics, math
from collections import defaultdict

def clean(x):
    if isinstance(x, str):
        try: return float(x)
        except Exception: return float("nan")
    try: return float(x)
    except Exception: return float("nan")

def mean(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return statistics.fmean(xs) if xs else float("nan")

def stdev(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return statistics.fmean(xs) if xs else 0.0

def gmean(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x) and x > 0]
    return math.exp(sum(math.log(x) for x in xs)/len(xs)) if xs else float("nan")

def load_grid(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        r = json.load(open(f))
        out[(r.get("method"), r.get("scene"), r.get("seed"))] = r
    return out

g = load_grid(r"artifacts\training-paper\results")
methods = sorted({m for (m,_,_) in g})
scenes  = sorted({s for (_,s,_) in g})
print("methods:", methods)
print("scenes :", scenes)

# ---- MASTER TABLE ----
master = []
for s in scenes:
    row = {"scene": s}
    for m in methods:
        rs = [r for (mm,ss,_), r in g.items() if mm==m and ss==s]
        w   = [clean(r["performance"]["wall_time_seconds"]) for r in rs]
        t   = [clean(r["performance"]["time_to_quality_seconds"]) for r in rs]
        p   = [clean(r["quality"].get("psnr_db")) for r in rs]
        si  = [clean(r["quality"].get("ssim")) for r in rs]
        gs  = [clean(r["resources"].get("final_gaussian_count")) for r in rs]
        row[m + "_wall_s"] = round(mean(w),1)
        row[m + "_ttq_s"]  = round(mean(t),1)
        row[m + "_psnr"]   = round(mean(p),2)
        row[m + "_ssim"]   = round(mean(si),4)
        row[m + "_gauss"]  = int(mean(gs))
    master.append(row)
    print(row)

# ratios vs gsplat for each other method
others = [m for m in methods if m != "gsplat"]
print("\nper-scene wall ratio gsplat/speedup:")
for s in scenes:
    line = s
    for o in others:
        r = next(x for x in master if x["scene"] == s)
        gw = r["gsplat_wall_s"]; ow = r[o + "_wall_s"]
        line += "  %s=%.3fx" % (o, gw/ow if ow else float("nan"))
    print(line)

# stage breakdown
print("\n=== STAGE BREAKDOWN (30k logs) ===")
stage_agg = defaultdict(lambda: defaultdict(list))
for f in sorted(glob.glob(r"results/training/*30k*.json")):
    r = json.load(open(f))
    cfg = r.get("config", {}) or {}
    scene = cfg.get("scene", "?")
    tile = cfg.get("tile_size", "?")
    ml = r.get("metrics_log", [])
    for e in ml:
        for k, v in e.items():
            if isinstance(k, str) and k.endswith("_ms") and isinstance(v, (int,float)) and v > 0:
                stage_agg[scene][k].append(float(v))
for s in sorted(stage_agg):
    tot = sum(max(v) for v in stage_agg[s].values()) or 1.0
    parts = []
    for k in sorted(stage_agg[s]):
        vals = stage_agg[s][k]
        parts.append("%s=%.1fms(%.1f%%)" % (k, mean(vals), 100.0*mean(vals)/tot))
    print(" %-10s %s" % (s, " ".join(parts)))
