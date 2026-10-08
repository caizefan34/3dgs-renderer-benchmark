#!/usr/bin/env python3
import argparse, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--scene-summary", required=True)
ap.add_argument("--predictability", required=False)
ap.add_argument("--out", required=True)
args = ap.parse_args()

scenes = json.loads(Path(args.scene_summary).read_text())
by = {x["scene"]: x for x in scenes}
primary = [by.get("room"), by.get("bicycle")]

if any(x is None for x in primary):
    verdict = "D0_INCONCLUSIVE"
    reason = "missing_primary_scene"
elif any(not x.get("trace_valid", False) for x in primary):
    verdict = "D0_INCONCLUSIVE"
    reason = "invalid_trace"
elif any(x.get("sample_reconstruction_error", 1.0) > 0.03 for x in primary):
    verdict = "D0_INCONCLUSIVE"
    reason = "renderer_cost_accounting_invalid"
elif any(x.get("R_upper_ci95_low", 0.0) < 0.60 for x in primary):
    verdict = "D0_NO_3X_HEADROOM"
    reason = "headroom_lower_bound_below_0.60"
elif all(x.get("R_upper_ci95_low", 0.0) >= 0.70 for x in primary):
    pred = []
    if args.predictability and Path(args.predictability).exists():
        pred = json.loads(Path(args.predictability).read_text())
    logreg = [x for x in pred if x.get("model") == "logreg"]
    need = {("room","bicycle"),("bicycle","room")}
    got = {(x.get("train_scene"),x.get("test_scene")) for x in logreg}
    if got != need:
        verdict = "D0_HEADROOM_BUT_NOT_PREDICTABLE"
        reason = "missing_cross_scene_predictability"
    else:
        strong = all(
            x.get("auroc",0.0) >= 0.85 and
            x.get("test_survivor_fpr",1.0) <= 0.02 and
            x.get("test_dead_cost_weighted_recall",0.0) >= 0.50
            for x in logreg
        )
        if strong:
            verdict = "D0_HEADROOM_AND_PREDICTABLE"
            reason = "headroom_and_cross_scene_prediction_strong"
        else:
            verdict = "D0_HEADROOM_BUT_NOT_PREDICTABLE"
            reason = "cross_scene_prediction_failed"
else:
    verdict = "D0_INCONCLUSIVE"
    reason = "headroom_borderline_0.60_to_0.70"

out = {"verdict": verdict, "reason": reason}
Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
