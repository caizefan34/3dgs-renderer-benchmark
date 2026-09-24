#!/usr/bin/env python3
"""Definitive: inspect grid files & training logs, compute stats."""
import json, glob, os

print("=== grid file keys ===")
candidates = glob.glob(r"C:\Users\36570\3dgs-renderer-benchmark\artifacts\training-paper\results\primary_full_convergence--higs_proposed--*.json")
print("higs_proposed primary files:", len(candidates))
if candidates:
    d = json.load(open(candidates[0]))
    print("first:", candidates[0])
    print("top keys:", list(d.keys()))
    ml = d.get("metrics_log", [])
    print("metrics_log len:", len(ml))
    if ml:
        print("first entry:", json.dumps(ml[0], indent=2)[:800])
    for k in ("config", "milestones", "resources"):
        print(k, "=>", json.dumps(d.get(k))[:500])

print()
print("=== training log files ===")
for f in sorted(glob.glob(r"C:\Users\36570\3dgs-renderer-benchmark\results\training\*.json")):
    d = json.load(open(f))
    print(" -", os.path.basename(f), "| keys:", list(d.keys()))
    print("   config:", json.dumps(d.get("config", {}))[:300])
