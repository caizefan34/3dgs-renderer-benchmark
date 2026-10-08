#!/usr/bin/env python3
"""P16-RB: evidence record for the curve x-axis correction (v1 -> v2).

Compares archived v1 curve JSONs, rebuilt v2 curve JSONs, and primary walls; pulls the
timings.txt cross-check totals from the v2 protocol blocks. Writes
artifacts/P16_CURVE_XAXIS_CORRECTION.json. Run after p16_rebuild_curves.sh.
"""
import json
import time
from pathlib import Path

R = Path("/root/epic05_p16rb")
METHODS = ["FasterGS", "FasterGSFused", "FastGS", "SADGS"]
SCENES = ["room", "bicycle", "garden"]
rows = []
for m in METHODS:
    for s in SCENES:
        v1 = R / "curves_v1_archive_20260928" / f"{m}__{s}.json"
        v2 = R / "curves" / m / f"{s}.json"
        pr = R / "runs" / m / s / "P16_RUN.json"
        row = {"method": m, "scene": s}
        if v1.exists():
            a = json.loads(v1.read_text())
            row["v1_x_s"] = [pt["training_elapsed_s"] for pt in a["points"]]
        if v2.exists():
            b = json.loads(v2.read_text())
            row["v2_x_s"] = [pt["training_elapsed_s"] for pt in b["points"]]
            proto = b["protocol"]
            row["cb_total_s"] = proto.get("callbacks_subtracted_total_s")
            row["timings_save_total_s"] = proto.get("timings_txt_save_intermediate_total_s")
            row["timings_vs_tqdm_delta_s"] = proto.get("timings_vs_tqdm_delta_s")
            row["callbacks_measured"] = proto.get("callbacks_measured")
            gc = proto.get("graphdeco_calibration")
            if gc:
                row["graphdeco_calibration"] = gc
        if pr.exists():
            rec = json.loads(pr.read_text())
            wall = rec["training_wall_s"]
            row["primary_wall_s"] = wall
            if "v2_x_s" in row:
                row["v2_final_drift_pct"] = round((row["v2_x_s"][-1] - wall) / wall * 100, 2)
            if "v1_x_s" in row:
                row["v1_final_drift_pct"] = round((row["v1_x_s"][-1] - wall) / wall * 100, 2)
            gc = row.get("graphdeco_calibration") or {}
            if gc.get("event_total_s") is not None:
                row["event_total_vs_primary_pct"] = round(
                    (gc["event_total_s"] - wall) / wall * 100, 2)
        rows.append(row)

out = {
    "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "issue": "v1 curve x-axis anchored on raw tqdm elapsed minus only the mtime-measured "
             "inline render pause; the nerficg curve protocol's inline callback also includes "
             "checkpoint serialization and metric evaluation (and each save point's own "
             "callback landed inside its own x), inflating x by ~9-23% vs the primary wall.",
    "fix": "v2 anchors x on the first tqdm redraw at/behind each save (pre-callback) and "
           "subtracts the full measured per-save callback cost (first-match tqdm deltas). "
           "Cross-checked against timings.txt _save_intermediate_checkpoint totals. "
           "graphdeco pairs additionally scale the save-subtracted wall by "
           "alpha = native CUDA-event Training time: / wall-after-saves, so the endpoint "
           "matches the same semantics as the primary runs (uniform per-iteration "
           "overhead assumption; alpha and raw wall recorded as evidence).",
    "validation": rows,
}
p = R / "artifacts" / "P16_CURVE_XAXIS_CORRECTION.json"
p.write_text(json.dumps(out, indent=2), encoding="utf-8")
print("WROTE", p)
for r in rows:
    print(r["method"], r["scene"], "v1_drift", r.get("v1_final_drift_pct"),
          "v2_drift", r.get("v2_final_drift_pct"), "timings_delta", r.get("timings_vs_tqdm_delta_s"))
