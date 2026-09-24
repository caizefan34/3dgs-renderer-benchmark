#!/usr/bin/env python3
"""Aggregate result JSONs into per-method x per-scene tables and geomeans."""
import json, glob, os, math, statistics, sys

def load_all(results_dir):
    out = []
    for f in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(f) as fh:
            j = json.load(fh)
        out.append(j)
    return out

def gmean(vals):
    vals = [v for v in vals if v and v > 0]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else float("nan")

for label, results_dir in [
    ("TRAINING-ALL", r"C:\Users\36570\3dgs-renderer-benchmark\artifacts\training-all\results"),
    ("TRAINING-PAPER", r"C:\Users\36570\3dgs-renderer-benchmark\artifacts\training-paper\results"),
]:
    if not os.path.isdir(results_dir):
        print(f"### {label}: missing")
        continue
    rows = load_all(results_dir)
    print(f"\n### {label}: {len(rows)} rows")
    # group by method then scene
    per = {}
    for j in rows:
        method = j.get("method", "?")
        scene = j.get("scene", "?")
        seed = j.get("seed", "?")
        per.setdefault((method, scene), []).append(j)
    for (method, scene), js in sorted(per.items()):
        walls = [j["performance"]["wall_time_seconds"] for j in js]
        ttqs = [j["performance"]["time_to_quality_seconds"] for j in js]
        psnrs = [j["quality"]["psnr_db"] for j in js]
        ss = [j["quality"]["ssim"] for j in js]
        lp = [j["quality"]["lpips"] for j in js]
        mems = [j["resources"]["peak_gpu_memory_mib"] for j in js]
        cnt = [j["resources"]["final_gaussian_count"] for j in js]
        def m(xs): return statistics.fmean(xs)
        def s(xs): return statistics.stdev(xs) if len(xs) > 1 else 0.0
        print(f"  {method:16s} {scene:28s} n={len(js)}  wall={m(walls):.1f}+-{s(walls):.1f}s  ttq={m(ttqs):.1f}+-{s(ttqs):.1f}s  psnr={m(psnrs):.2f}+-{s(psnrs):.2f}  ssim={m(ss):.4f}+-{s(ss):.4f}  lpips={m(lp):.4f}  mem={m(mems):.1f}MiB  gauss={m(cnt):,.0f}")
    # geomean of wall times by method across all scenes/seeds (matching paper convention: per-scene mean then geomean)
    by_scene_method = {}
    for (method, scene), js in per.items():
        by_scene_method.setdefault(method, {}).setdefault(scene, []).extend(js)
    print("  --- geomeans ---")
    for method in sorted(by_scene_method):
        scene_means = []
        for scene, js in by_scene_method[method].items():
            scene_means.append(statistics.fmean([j["performance"]["wall_time_seconds"] for j in js]))
        if scene_means:
            gm = gmean(scene_means)
            print(f"  {method:16s} geomean_wall={gm:.1f}s  (n_scenes={len(scene_means)})")
