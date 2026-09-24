#!/usr/bin/env python3
"""Definitive aggregation: compute report tables and geomean ratios."""
import json, glob, os, math, statistics
from collections import defaultdict

def load_dir(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        try:
            with open(f) as fh:
                r = json.load(fh)
            out[(r.get("method"), r.get("scene"), r.get("seed"))] = r
        except Exception:
            pass
    return out

def gm(vals):
    vals = [v for v in vals if v and math.isfinite(v) and v > 0]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else float("nan")

def mean(vals):
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    return statistics.fmean(vals) if vals else float("nan")

for label, d in [
    ("PAPER", r"C:\Users\36570\3dgs-renderer-benchmark\artifacts\training-paper\results"),
    ("ALL", r"C:\Users\36570\3dgs-renderer-benchmark\artifacts\training-all\results"),
]:
    rows = load_dir(d)
    print(f"\n===== {label}: {len(rows)} rows =====")
    scenes = sorted({s for (_, s, _) in rows if s})
    methods = sorted({m for (m, _, _) in rows if m})
    print("methods:", methods)
    print("scenes :", scenes)
    by = {}
    for (m, s, seed), r in rows.items():
        by.setdefault((m, s), []).append(r)
    print()
    for (m, s), rs in sorted(by.items()):
        wall = mean([r["performance"]["wall_time_seconds"] for r in rs])
        ttq = mean([r["performance"]["time_to_quality_seconds"] for r in rs])
        psnr = mean([r["quality"]["psnr_db"] for r in rs])
        ss = mean([r["quality"]["ssim"] for r in rs])
        lp = mean([r["quality"]["lpips"] for r in rs])
        g = mean([r["resources"]["final_gaussian_count"] for r in rs])
        print(f"  {m.strip():16s} {s.strip():28s} n={len(rs)} wall={wall:9.1f}s ttq={ttq:9.1f}s psnr={psnr:6.2f} ssim={ss:.4f} lpips={lp:.4f} gauss={g:9,.0f}")
    print()
    for m in methods:
        wg = [mean([r["performance"]["wall_time_seconds"] for r in by[(m, s)]]) for s in scenes if (m, s) in by]
        tg = [mean([r["performance"]["time_to_quality_seconds"] for r in by[(m, s)]]) for s in scenes if (m, s) in by]
        print(f"  {m.strip():16s} wall_gmean={gm(wg):9.1f}s  ttq_gmean={gm(tg):9.1f}s")
