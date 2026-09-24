#!/usr/bin/env python3
"""Comprehensive result-grid analysis."""
import json, glob, os, math, statistics
from collections import defaultdict

def load_all(results_dir):
    out = []
    for f in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        try:
            with open(f) as fh:
                out.append(json.load(fh))
        except Exception as e:
            print(f"  ERR {os.path.basename(f)}: {e}")
    return out

for label, results_dir in [
    ("TRAINING-PAPER", r"C:\Users\36570\3dgs-renderer-benchmark\artifacts\training-paper\results"),
    ("TRAINING-ALL", r"C:\Users\36570\3dgs-renderer-benchmark\artifacts\training-all\results"),
]:
    rows = load_all(results_dir)
    print(f"\n### {label}: {len(rows)} rows")
    by_ws = defaultdict(lambda: defaultdict(set))  # (workload, method) -> scene -> seeds
    for j in rows:
        wl = j["job_id"].split("--")[0]
        m = j["method"]
        s = j["scene"]
        sd = j["seed"]
        by_ws[(wl, m)][s].add(sd)
    for (wl, m) in sorted(by_ws):
        scenes = by_ws[(wl, m)]
        nscenes = len(scenes)
        wants = ["mipnerf360/bicycle","mipnerf360/bonsai","mipnerf360/counter","mipnerf360/garden","mipnerf360/kitchen","mipnerf360/room","mipnerf360/stump","tanks_and_temples/train","tanks_and_temples/truck","deep_blending/drjohnson","deep_blending/playroom"]
        missing = [s for s in wants if s not in scenes]
        seedcnt = set()
        for s, sds in scenes.items():
            seedcnt |= sds
        nseeds = len(seedcnt)
        print(f"  {wl} / {m}: scenes={nscenes} seeds={nseeds} missing={missing if missing else '-'}")
