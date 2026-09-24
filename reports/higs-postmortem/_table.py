#!/usr/bin/env python3
"""Full analysis: per-scene x per-seed tables for the 13-scene x 3-method trainable HiGS study."""
import json, glob, os, math, statistics
from collections import defaultdict

def anchor(results_dir):
    """Load, validate, and key result JSONs."""
    out = {}
    for f in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(f) as fh:
            j = json.load(fh)
        key = (j.get("job_id", ""), j.get("method"), j.get("scene"), j.get("seed"))
        out[key] = j
    return out

def num(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else float("nan")
    except TypeError:
        return float("nan")

def gmean(vals):
    vals = [v for v in vals if v and v > 0 and math.isfinite(v)]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else float("nan")

def mean(vals):
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    return statistics.fmean(vals) if vals else float("nan")

for label, results_dir in [
    ("TRAINING-PAPER", r"C:\Users\36570\3dgs-renderer-benchmark\artifacts\training-paper\results"),
    ("TRAINING-ALL", r"C:\Users\36570\3dgs-renderer-benchmark\artifacts\training-all\results"),
]:
    rows = anchor(results_dir)
    print(f"\n{'='*110}\n### {label}: {len(rows)} rows\n{'='*110}")
    # key rows by method x scene
    by = defaultdict(list)
    for (jid, method, scene, seed), j in rows.items():
        p = j["performance"]
        q = j["quality"]
        r = j["resources"]
        by[(method, scene)].append({
            "seed": seed,
            "wall": num(p.get("wall_time_seconds")),
            "ttq": num(p.get("time_to_quality_seconds")),
            "psnr": num(q.get("psnr_db")),
            "ssim": num(q.get("ssim")),
            "lpips": num(q.get("lpips")),
            "gauss": num(r.get("final_gaussian_count")),
            "mem": num(r.get("peak_gpu_memory_mib")),
            "energy": num(r.get("energy_joules")),
        })
    # method x scene summary
    scenes_all = sorted({s for (m, s) in by if s})
    methods_all = sorted({m for (m, s) in by if m})
    print(f"  methods: {methods_all}")
    print(f"  scenes : {scenes_all}")
    print("\n  per-method x scene: mean wall (s)  |  mean psnr (dB)  |  mean gauss")
    hdr = f"  {'scene':<28s}"
    for m in methods_all:
        hdr += f" | {m:>10s}_wall | {m:>10s}_psnr | {m:>10s}_gauss"
    print(hdr)
    print("  " + "-" * (len(hdr)))
    for s in scenes_all:
        row = f"  {s:<28s}"
        for m in methods_all:
            entries = by.get((m, s), [])
            w = mean([e["wall"] for e in entries])
            p = mean([e["psnr"] for e in entries])
            g = mean([e["gauss"] for e in entries])
            row += f" | {w:10.1f} | {p:10.2f} | {g:10,.0f}"
        print(row)
    # geomeans
    print("\n  geomean over scenes of scene-mean wall time:")
    for m in methods_all:
        scene_means = []
        for s in scenes_all:
            entries = by.get((m, s), [])
            if entries:
                scene_means.append(mean([e["wall"] for e in entries]))
        print(f"    {m:16s}: gmean={gmean(scene_means):.1f}s  (n={len(scene_means)})")
    # ratio table
    print("\n  scene-mean wall ratio (gslam / higs_full, gslam / higs_proposed):")
    for s in scenes_all:
        wg = mean([e["wall"] for e in by.get(("gslam", s), [])])
        whf = mean([e["wall"] for e in by.get(("higs_full", s), [])])
        whp = mean([e["wall"] for e in by.get(("higs_proposed", s), [])])
        row = f"  {s:<28s}"
        if wg and whf:
            row += f" gslam/higs_full = {wg/whf:.4f}x"
        if wg and whp:
            row += f" gslam/higs_proposed = {wg/whp:.4f}x"
        print(row)
