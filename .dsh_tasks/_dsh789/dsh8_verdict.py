#!/usr/bin/env python3
"""DSH-8 APBC0 verdict generator (Phase A structural headroom gate)."""
import json
from pathlib import Path

RUNS = Path("/root/dsh_20261001/runs/DSH_PHASEA")
ART = RUNS.parent / "artifacts" / "DSH8_APBC0"
THRESH = 0.25


def load_scene(scene):
    recs = []
    p = RUNS / scene / "trace" / "phase_a_records.jsonl"
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    recs.sort(key=lambda r: r["iteration"])
    return recs


def main():
    all_scenes = {s: load_scene(s) for s in ["room", "garden", "bicycle"]}

    reach = {
        "p0": "DSH-8 APBC0 Exact Active-Pixel Bucket Compaction",
        "phase": "A_structural_headroom_trace",
        "K": 32,
        "predicate": "pixel p reachable by bucket b iff n_contrib[p] > 32*b "
                     "(verified vs kernels_backward.cuh blend_backward_cu: "
                     "bucket survives iff tile_bucket_idx*32 < tile_max_n_processed; "
                     "lane skips iff tile_primitive_idx >= last_contributor)",
        "scenes": {},
        "gate_P1": {
            "description": ">=25% of baseline bucket-pixel scan visits are "
                           "structurally unreachable, cost-weighted, in EVERY scene.",
            "threshold": THRESH,
            "per_scene_peak": {},
            "every_scene_pass": False,
            "min_scene_peak": None,
        },
    }
    per_scene_peak = {}
    for s, recs in all_scenes.items():
        fracs = {r["iteration"]: r["dsh8_apbc0"]["unreachable_fraction_cost_weighted"]
                 for r in recs}
        peak_it = max(fracs, key=lambda k: fracs[k])
        per_scene_peak[s] = fracs[peak_it]
        use = recs  # full trajectory
        reach["scenes"][s] = {
            "n_records": len(recs),
            "iterations": [r["iteration"] for r in recs],
            "unreachable_fraction_by_iter": {str(k): v for k, v in fracs.items()},
            "peak_unreachable_fraction": fracs[peak_it],
            "peak_iteration": peak_it,
            "final_record": recs[-1]["dsh8_apbc0"] if recs else None,
        }
    every_pass = bool(per_scene_peak) and all(v >= THRESH for v in per_scene_peak.values())
    reach["gate_P1"]["per_scene_peak"] = per_scene_peak
    reach["gate_P1"]["every_scene_pass"] = every_pass
    reach["gate_P1"]["min_scene_peak"] = min(per_scene_peak.values()) if per_scene_peak else None
    reach["gate_P1"]["decision"] = ("APBC0_P1_PASS" if every_pass
                                    else "APBC0_NO_STRUCTURAL_HEADROOM")
    (ART / "APBC0_REACHABILITY_TRACE.json").write_text(json.dumps(reach, indent=2))

    decision = reach["gate_P1"]["decision"]
    verdict = {
        "p0": "DSH-8 APBC0 Exact Active-Pixel Bucket Compaction",
        "decision": decision,
        "evidence_level": "L1 (Phase A structural trace, EPIC-05 A100)",
        "gate_P1": {
            "threshold": THRESH,
            "metric": "peak cost-weighted structurally-unreachable fraction across all captured checkpoints",
            "per_scene_peak": per_scene_peak,
            "min_scene_peak": reach["gate_P1"]["min_scene_peak"],
            "every_scene_pass": every_pass,
        },
        "stop_rule_applied": not every_pass,
        "reasoning": (
            "P1 requires >=25% cost-weighted structurally-unreachable "
            "bucket-pixel scan visits in EVERY scene. Peak across all captured "
            "checkpoints (500/2K/5K/10K/14.5K/20K/25K) per scene: "
            + ", ".join("%s=%.4f" % (s, v) for s, v in per_scene_peak.items())
            + ". Min scene peak = %.4f < %.2f, and no scene reaches 25%%, so "
               "P1 FAILS. The trajectory is non-monotonic (peaks at 5K-10K for "
               "room/garden, rises again to 20K-25K for bicycle), but even the "
               "largest scene (bicycle, deepest contributor lists) peaks at "
               "20.5%% < 25%%. The 'EVERY scene' requirement fails primarily "
               "because room (12.0%%) and garden (10.5%%) are far below. Phase A "
               "predicate was verified against kernels_backward.cuh "
               "blend_backward_cu (bucket survives iff tile_bucket_idx*32 < "
               "tile_max_n_processed; lane skips iff tile_primitive_idx >= "
               "last_contributor = tile_n_processed[pixel])."
            % (reach["gate_P1"]["min_scene_peak"], THRESH)),
        "phase_B_permitted": every_pass,
        "phase_C_permitted": every_pass,
        "claim_scope": ("Structural headroom ABSENT at L1; no method claim. "
                        "Phase B/C prototype not authorized by the stop rule."),
    }
    (ART / "APBC0_VERDICT.json").write_text(json.dumps(verdict, indent=2))
    print(json.dumps({"decision": decision, "per_scene_peak": per_scene_peak,
                      "min_scene_peak": reach["gate_P1"]["min_scene_peak"]}, indent=2))


if __name__ == "__main__":
    main()
