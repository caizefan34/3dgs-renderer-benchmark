#!/usr/bin/env python3
"""Full analysis: per-scene x per-seed tables for the 13-scene x 3-method trainable HiGS study."""
import json, glob, os, math, statistics
from collections import defaultdict

def load_dir(d):
    rows = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        try:
            with open(f) as fh:
                r = json.load(fh)
            key = (r.get("method"), r.get("scene"), r.get("seed"))
            rows[key] = r
        except Exception as e:
            print(f"  ERR {os.path.basename(f)}: {e}")
    return rows

def gmean(vals):
    vals = [v for v in vals if v and math.isfinite(v) and v > 0]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else float("nan")

for label, d in [
    ("TRAINING-PAPER", r"C:\Users\36570\3dgs-renderer-benchmark\artifacts\training-paper\results"),
    ("TRAINING-ALL", r"C:\Users\36570\3dgs-renderer-benchmark\artifacts\training-all\results"),
]:
    rows = load_dir(d)
    print(f"\n================ {label}: {len(rows)} rows ================")
    by = defaultdict(list)
    for (m, s, seed), r in rows.items():
        by[(m, s)].append(r)
    scenes = sorted({s for (_, s) in by if s})
    methods = sorted({m for (m, _) in by if m})
    print("methods:", methods)
    print("scenes :", scenes)
    print()
    hdr = f"{'scene':<28s}"
    for m in methods:
        hdr += f" | {m:>10s}_wall {m:>10s}_ttq {m:>8s}_psnr {m:>8s}_gauss"
    print(hdr)
    print("-" * len(hdr))
    for s in scenes:
        line = f"{s:<28s}"
        for m in methods:
            rs = by.get((m, s), [])
            wall = statistics.fmean([r["performance"]["wall_time_seconds"] for r in rs]) if rs else float("nan")
            ttq = statistics.fmean([r["performance"]["time_to_quality_seconds"] for r in rs]) if rs else float("nan")
            psnr = statistics.fmean([r["quality"]["psnr_df"] if "psnr_df" in r["quality"] else r["quality"]["psnr_db"] for r in rs]) if rs else float("nan")
            gauss = statistics.fmean([r["resources"]["final_gaussian_count"] for r in rs]) if rs else float("nan")
            line += f" | {wall:10.1f} {ttq:10.1f} {psnr:8.2f} {gauss:6.0f}"
        print(line)
    print()
    for m in methods:
        gws, gts, gps, gss = [], [], [], []
        for s in scenes:
            rs = by.get((m, s), [])
            if not rs:
                continue
            gws.append(statistics.fmean([r["performance"]["wall_time_seconds"] for r in rs]))
            gts.append(statistics.fmean([r["performance"]["time_to_quality_seconds"] for r in rs]))
            gps.append(statistics.fmean([r["quality"]["psnr_db"] for r in rs]))
            gss.append(statistics.fmean([r["quality"]["ssim"] for r in rs]))
        print(f"  {m:16s} geomean_wall={gmean(gws):.1f}s  geomean_ttq={gmean(gts):.1f}s  mean_psnr={statistics.fmean(gps):.2f}  mean_ssim={statistics.fmean(gss):.4f}")
    print()
    if "gslam" in methods and "higs_full" in methods and "higs_proposed" in methods:
        for pair in [("gslam", "higs_full"), ("gslam", "higs_proposed"), ("higs_full", "higs_proposed")]:
            base, cand = pair
            wall_ratios, ttq_ratios = [], []
            for s in scenes:
                rb = by.get((base, s), [])
                rc = by.get((cand, s), [])
                if not rb or not rc:
                    continue
                bw = statistics.fmean([r["performance"]["wall_time_seconds"] for r in rb])
                cw = statistics.fmean([r["performance"]["wall_time_seconds"] for r in rc])
                bt = statistics.fmean([r["performance"]["time_to_quality_seconds"] for r in rb])
                ct = statistics.fmean([r["performance"]["time_to_quality_seconds"] for r in rc])
                if bw > 0 and bt > 0:
                    wall_ratios.append(bw / cw)
                    ttq_ratios.append(bt / ct)
            print(f"  wall-speedup {base}/{cand} geomean = {gmean(wall_ratios):.4f}x   ttq-speedup = {gmean(ttq_ratios):.4f}x   (n={len(wall_ratios)})")
