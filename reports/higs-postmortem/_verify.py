#!/usr/bin/env python3
"""Compute per-method geomean wall times and ratios from result JSONs."""
import json, glob, os, math, statistics
from collections import defaultdict

def load_all(results_dir):
    out = []
    for f in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(f) as fh:
            out.append(json.load(fh))
    return out

def gm(vals):
    vals = [v for v in vals if v and v > 0]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else float("nan")

for label, results_dir in [
    ("TRAINING-PAPER", r"C:\Users\36570\3dgs-renderer-benchmark\artifacts\training-paper\results"),
    ("TRAINING-ALL", r"C:\Users\36570\3dgs-renderer-benchmark\artifacts\training-all\results"),
]:
    rows = load_all(results_dir)
    print(f"\n=== {label}: {len(rows)} rows")
    groups = defaultdict(list)
    for r in rows:
        method = r["method"]
        perf = r["performance"]
        groups[method].append((r["scene"], r["seed"], perf["wall_time_seconds"], perf["time_to_quality_seconds"]))
    for method in sorted(groups):
        scenes = {}
        for scene, seed, wall, ttq in groups[method]:
            scenes.setdefault(scene, []).append((seed, wall, ttq))
        scene_wall_gms = []
        scene_ttq_gms = []
        for scene, entries in sorted(scenes.items()):
            walls = [e[1] for e in entries]
            ttqs = [e[2] for e in entries]
            scene_wall_gms.append(gm(walls))
            scene_ttq_gms.append(gm(ttqs))
        print(f"  {method:16s}: n={len(groups[method])} scenes={len(scene_wall_gms)}")
        print(f"      geom_wall={gm(scene_wall_gms):.1f}s  geom_ttq={gm(scene_ttq_gms):.1f}s")
        for scene, entries in sorted(scenes.items()):
            walls = [e[1] for e in entries]
            ttqs = [e[2] for e in entries]
            print(f"        {scene:28s} wall={gm(walls):.1f}s ttq={gm(ttqs):.1f}s seeds={len(entries)}")
    # ratios: baseline gslam vs others
    if "gslam" in groups and "higs_proposed" in groups:
        def scene_ratio_geomean(m1, m2):
            # geomean over scenes of (mean over seeds for m1) / (mean over seeds for m2)
            ratios = []
            s1 = {}
            s2 = {}
            for scene, seed, wall, ttq in groups[m1]:
                s1.setdefault(scene, []).append(wall)
            for scene, seed, wall, ttq in groups[m2]:
                s2.setdefault(scene, []).append(wall)
            for scene in s1:
                if scene in s2 and s1[scene] and s2[scene]:
                    ratios.append(statistics.fmean(s1[scene]) / statistics.fmean(s2[scene]))
            return gm(ratios), len(ratios)
        r_wall, n = scene_ratio_geomean("gslam", "higs_proposed")
        print(f"  gslam/higs_proposed wall geomean = {r_wall:.4f}x ({n} scenes)")
        print(f"  interpretation: higs_proposed speedup = {r_wall:.4f}x  (1/r_wall = {1/r_wall:.4f} = time of higs / gslam)")
