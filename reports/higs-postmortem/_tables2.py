#!/usr/bin/env python3
"""Final data tables for the regression postmortem."""
import json, glob, os, statistics, math
from collections import defaultdict

def clean(x):
    if isinstance(x, str):
        try:
            return float(x)
        except Exception:
            return float("nan")
    try:
        return float(x)
    except Exception:
        return float("nan")

def gm(xs):
    xs = [x for x in xs if x and math.isfinite(x) and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")

def mean(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return statistics.fmean(xs) if xs else float("nan")

def load_grid(d):
    rows = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        r = json.load(open(f))
        rows[(r.get("method"), r.get("scene"), r.get("seed"))] = r
    return rows

def per_scene_table(g, methods=None):
    by = defaultdict(list)
    for (m, s, seed), r in g.items():
        by[(m, s)].append(r)
    scenes = sorted({s for (_, s) in by if s})
    if methods is None:
        methods = sorted({m for (m, _) in by if m})
    print("scene;", end="")
    for m in methods:
        print(";" + m + "_wall;" + m + "_psnr", end="")
    print()
    for s in scenes:
        print(s, end="")
        for m in methods:
            rs = by.get((m, s), [])
            w = mean([clean(r["performance"]["wall_time_seconds"]) for r in rs]) if rs else float("nan")
            p = mean([clean(r["quality"]["psnr_db"]) for r in rs]) if rs else float("nan")
            print(";%.1f;%.2f" % (w, p), end="")
        print()
    print("geomeans;")
    for m in methods:
        ws = [mean([clean(r["performance"]["wall_time_seconds"]) for r in by[(m, s)]]) for s in scenes if (m, s) in by]
        ttqs = [mean([clean(r["performance"]["time_to_quality_seconds"]) for r in by[(m, s)]]) for s in scenes if (m, s) in by]
        p = [mean([clean(r["quality"]["psnr_db"]) for r in by[(m, s)]]) for s in scenes if (m, s) in by]
        print("%s;%.1f;%.1f;%.2f" % (m, gm(ws), gm(ttqs), mean(p)))
    return by

for label, d in [("PAPER", r"artifacts\training-paper\results"), ("ALL", r"artifacts\training-all\results")]:
    print("=" * 140)
    print("###", label, "###")
    g = load_grid(d)
    by = per_scene_table(g)
    methods = sorted({m for (m, _, _) in g})
    print("ratios;")
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            m1, m2 = methods[i], methods[j]
            scenes = sorted({s for (_, s) in by if s})
            ratios = []
            for s in scenes:
                w1 = mean([clean(r["performance"]["wall_time_seconds"]) for r in by.get((m1, s), [])])
                w2 = mean([clean(r["performance"]["wall_time_seconds"]) for r in by.get((m2, s), [])])
                if w1 and w2:
                    ratios.append(w1 / w2)
            print("%s/%s;%.4f;%d" % (m1, m2, gm(ratios), len(ratios)))
