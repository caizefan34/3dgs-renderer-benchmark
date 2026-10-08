"""Summarize and validate scalar pixel-work counts for the room split event."""

import json
import statistics
from pathlib import Path


HERE = Path(__file__).parent
source = json.loads((HERE / "fastgs_shadow_work_result.json").read_text(encoding="utf-8"))
rows = []
for view in source["views"]:
    work = view["work"]
    before, after = work["native_before"], work["native_after"]
    birth, half, release = work["shadow_birth"], work["shadow_half"], work["shadow_release"]
    if birth["nonzero_alpha_accepted"] != before["nonzero_alpha_accepted"]:
        raise AssertionError(f"birth active-work parity failed: {view['camera']}")
    if release["nonzero_alpha_accepted"] != after["nonzero_alpha_accepted"]:
        raise AssertionError(f"release active-work parity failed: {view['camera']}")
    rows.append({
        "camera": view["camera"],
        "tile_instance_union_over_native_after": view["tile_instances"]["union"] / view["tile_instances"]["after"],
        "half_visited_over_native_after": half["visited_tile_rows"] / after["visited_tile_rows"],
        "half_raw_over_native_after": half["raw_above_cutoff"] / after["raw_above_cutoff"],
        "half_active_over_native_after": half["nonzero_alpha_accepted"] / after["nonzero_alpha_accepted"],
        "shadow_share_of_half_raw": half["shadow_raw_above_cutoff"] / half["raw_above_cutoff"],
        "shadow_raw_millions": half["shadow_raw_above_cutoff"] / 1e6,
    })
keys = [key for key in rows[0] if key != "camera"]
summary = {key: {"min": min(row[key] for row in rows),
                 "max": max(row[key] for row in rows),
                 "median": statistics.median(row[key] for row in rows)} for key in keys}
out = {"views": rows, "summary": summary,
       "endpoint_active_work_parity": True,
       "scope": "Four room train views, one native 4,100 split event, scalar pixel work "
                "on author-projected/sorted buffers. Counts do not predict CUDA fused VJP "
                "wall time or long-run reconstruction utility."}
(HERE / "fastgs_shadow_work_analysis.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
print(json.dumps(out, indent=2))
