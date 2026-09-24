#!/usr/bin/env python3
"""Extract densification event trajectories from training JSONs."""
import json
from pathlib import Path

result_dir = Path("/home/liaoyuanjun/3dgs-renderer-benchmark/results/a100/phase-c51-stage4a")
for f in sorted(result_dir.glob("training_5k_*.json")):
    name = f.stem.replace("training_5k_", "")
    if name == "analysis":
        continue
    data = json.load(open(f))
    events = data.get("densification_events", [])
    print(f"=== {name} ({len(events)} events) ===")
    for e in events[:3]:
        print(f"  iter={e['iter']}: clone={e['cloned']}, split={e['split']}, prune={e['pruned']}, GS={e['gaussians']}")
    if len(events) > 6:
        print(f"  ... ({len(events)-6} more)")
    if len(events) > 3:
        for e in events[-3:]:
            print(f"  iter={e['iter']}: clone={e['cloned']}, split={e['split']}, prune={e['pruned']}, GS={e['gaussians']}")
    print()
