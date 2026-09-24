#!/usr/bin/env python3
"""Count matrix composition of result corpora."""
import json, glob, os
from collections import defaultdict

for root in [
    r"C:\Users\36570\3dgs-renderer-benchmark\artifacts\training-all\results",
    r"C:\Users\36570\3dgs-renderer-benchmark\artifacts\training-paper\results",
]:
    if not os.path.isdir(root):
        continue
    files = sorted(glob.glob(os.path.join(root, "*.json")))
    counts = defaultdict(int)
    by_pair = defaultdict(list)
    for f in files:
        with open(f) as fh:
            j = json.load(fh)
        wl = j.get("job_id", os.path.basename(f)).split("--")[0]
        method = j.get("method", "?")
        scene = j.get("scene", "?")
        seed = j.get("seed", "?")
        status = j.get("status", "?")
        pair = (wl, method)
        counts[pair] += 1
        by_pair[(wl, method)].append((scene, seed, status))
    print(f"\n### {root}")
    for (wl, method), n in sorted(counts.items()):
        items = by_pair[(wl, method)]
        scenes = sorted(set(s for s, _, _ in items))
        seeds = sorted(set(sd for _, sd, _ in items))
        statuses = sorted(set(st for _, _, st in items))
        print(f"  {wl} / {method}: n={n}  scenes={len(scenes)}  seeds={seeds}  statuses={statuses}")
