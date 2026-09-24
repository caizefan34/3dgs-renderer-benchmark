#!/usr/bin/env python3
"""final30k_aggregate.py — Driver for the FINAL-30K aggregate artifacts.

Reuses the FROZEN harness logic verbatim (harness/final30k/aggregate_results.py
and harness/final30k/derive_ttq.py) — imported, not reimplemented.

Produces (under artifacts/final-30k/):
  aggregate_performance.json   frozen aggregate_results output (speedup/reduction)
  aggregate_quality.json       frozen quality deltas
  ttq.json                     frozen derive_ttq output (R1 censoring semantics)
  n_gs_comparison.json         per-scene final/initial N + totals
  phase_timing.json            per-scene phase means + aggregate
  failure_manifest.json        every non-SUCCESS or contaminated run
  final_verdict.json           FINAL_PASS / FINAL_MIXED / FINAL_FAIL

Verdict rule (PRE-REGISTERED before any 30K result was seen, 2026-09-23):
  FINAL_PASS  := 26/26 SUCCESS, zero contamination, geomean speedup > 1.00,
                 aggregate |mean dPSNR| <= 0.25 dB, no scene dPSNR < -0.50 dB.
  FINAL_FAIL  := any run not SUCCESS, OR geomean speedup <= 1.00 with
                 aggregate dPSNR < 0, OR aggregate dPSNR <= -0.50 dB.
  FINAL_MIXED := everything else (e.g. faster but a quality dip beyond the
                 bound on 1-2 scenes; or quality neutral but not faster).
"""
import json
import math
import os
import sys
from pathlib import Path

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "harness", "final30k"))
sys.path.insert(0, REPO)

import aggregate_results as frozen_agg   # noqa: E402  (frozen harness)
import derive_ttq as frozen_ttq          # noqa: E402  (frozen harness)

F30K = Path(os.path.join(REPO, "artifacts", "final-30k"))
SCENES = ["bicycle", "bonsai", "counter", "flowers", "garden", "kitchen",
          "room", "stump", "treehill", "train", "truck", "drjohnson", "playroom"]
BASE = "B1A_ACCUTILE"
CAND = "C0_V3_FINAL30K"


def jload(p):
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def jdump(obj, name):
    p = os.path.join(F30K, name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    print(f"wrote {p}")


def geomean(xs):
    xs = [x for x in xs if x is not None and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main():
    # ---- frozen aggregate (performance + quality in one structure) ----
    agg = frozen_agg.aggregate(CAND, BASE, F30K)
    jdump(agg, "aggregate_performance.json")
    jdump({"quality_delta": agg["quality_delta"],
           "note": "dLPIPS = candidate - baseline; NEGATIVE is improvement; sign NOT flipped."},
          "aggregate_quality.json")

    # ---- frozen TTQ ----
    ttq = frozen_ttq.derive(CAND, BASE, os.path.join(F30K))
    jdump(ttq, "ttq.json")

    # ---- N_GS comparison ----
    n_gs = {"per_scene": {}, "note": "N_GS = gaussian count after 30000 steps (densification end)."}
    finals_c, finals_b = [], []
    for s in SCENES:
        c = jload(os.path.join(F30K, CAND, s, "training_results.json"))
        b = jload(os.path.join(F30K, BASE, s, "training_results.json"))
        row = {
            "initial_N": {"b1a": b.get("initial_N"), "c0": c.get("initial_N")},
            "final_N": {"b1a": b.get("final_N"), "c0": c.get("final_N")},
            "ratio_c0_over_b1a": (c["final_N"] / b["final_N"]) if c.get("final_N") and b.get("final_N") else None,
            "total_clones": {"b1a": b.get("total_clones"), "c0": c.get("total_clones")},
            "total_splits": {"b1a": b.get("total_splits"), "c0": c.get("total_splits")},
            "total_prunes": {"b1a": b.get("total_prunes"), "c0": c.get("total_prunes")},
        }
        n_gs["per_scene"][s] = row
        finals_c.append(row["final_N"]["c0"])
        finals_b.append(row["final_N"]["b1a"])
    n_gs["aggregate"] = {
        "mean_final_N": {"b1a": mean(finals_b), "c0": mean(finals_c)},
        "geomean_ratio": geomean([n_gs["per_scene"][s]["ratio_c0_over_b1a"] for s in SCENES]),
    }
    jdump(n_gs, "n_gs_comparison.json")

    # ---- phase timing ----
    ph = {"per_scene": {}, "aggregate": {}}
    for s in SCENES:
        tc = jload(os.path.join(F30K, CAND, s, "timing.json"))
        tb = jload(os.path.join(F30K, BASE, s, "timing.json"))
        ph["per_scene"][s] = {
            "b1a": tb.get("phase_ms") or {}, "c0": tc.get("phase_ms") or {},
            "nested_fb_ms": {"b1a": tb.get("nested_fb_ms"), "c0": tc.get("nested_fb_ms")},
            "mean_iter_ms": {"b1a": tb.get("mean_iter_ms"), "c0": tc.get("mean_iter_ms")},
        }
    for phase in ("forward", "backward", "loss", "densify", "optimizer"):
        cb = [(ph["per_scene"][s]["b1a"].get(phase) or {}).get("mean_ms") for s in SCENES]
        cc = [(ph["per_scene"][s]["c0"].get(phase) or {}).get("mean_ms") for s in SCENES]
        ph["aggregate"][phase] = {"b1a_mean_ms": mean(cb), "c0_mean_ms": mean(cc)}
    jdump(ph, "phase_timing.json")

    # ---- failure manifest ----
    failures = []
    for cand_id in (BASE, CAND):
        for s in SCENES:
            fs_p = os.path.join(F30K, cand_id, s, "final_status.json")
            if not os.path.exists(fs_p):
                failures.append({"candidate": cand_id, "scene": s, "status": "MISSING"})
                continue
            fs = jload(fs_p)
            if fs.get("status") != "SUCCESS" or fs.get("contaminated"):
                failures.append({"candidate": cand_id, "scene": s,
                                 "status": fs.get("status"),
                                 "contaminated": fs.get("contaminated", False)})
    jdump({"failures": failures, "n_failures": len(failures)}, "failure_manifest.json")

    # ---- verdict (pre-registered rule) ----
    n_success_pairs = agg.get("n_success_pairs", 0)
    gm_speedup = agg.get("speedup", {}).get("geomean")
    mean_dpsnr = agg.get("quality_delta", {}).get("mean_dPSNR")
    per_scene_dpsnr = {x["scene"]: x["dPSNR"] for x in agg.get("quality_delta", {}).get("per_scene", [])}
    worst_scene_dpsnr = min(per_scene_dpsnr.values()) if per_scene_dpsnr else None
    n_contaminated = sum(1 for f in failures if f.get("contaminated"))

    reasons = []
    if n_success_pairs != len(SCENES) or failures:
        reasons.append(f"execution: {n_success_pairs}/{len(SCENES)} success pairs; "
                       f"{len(failures)} failure/contamination entries")
    if gm_speedup is not None and gm_speedup <= 1.0:
        reasons.append(f"geomean speedup {gm_speedup:.4f} <= 1.00")
    if mean_dpsnr is not None and mean_dpsnr <= -0.50:
        reasons.append(f"aggregate dPSNR {mean_dpsnr:.3f} dB <= -0.50 dB")
    if worst_scene_dpsnr is not None and worst_scene_dpsnr < -0.50:
        reasons.append(f"worst scene dPSNR {worst_scene_dpsnr:.3f} dB < -0.50 dB")

    if reasons and (failures or (gm_speedup is not None and gm_speedup <= 1.0 and (mean_dpsnr or 0) < 0)
                    or (mean_dpsnr is not None and mean_dpsnr <= -0.50)):
        verdict = "FINAL_FAIL"
    elif not reasons:
        verdict = "FINAL_PASS"
    else:
        verdict = "FINAL_MIXED"

    jdump({
        "verdict": verdict,
        "rule": ("PRE-REGISTERED 2026-09-23 before any 30K result was seen: "
                 "FINAL_PASS = 26/26 SUCCESS, zero contamination, geomean speedup > 1.00, "
                 "|mean dPSNR| <= 0.25 dB, no scene dPSNR < -0.50 dB; "
                 "FINAL_FAIL = any run not SUCCESS, or geomean speedup <= 1.00 with dPSNR < 0, "
                 "or mean dPSNR <= -0.50 dB; FINAL_MIXED = otherwise."),
        "inputs": {
            "n_success_pairs": n_success_pairs,
            "n_failures_or_contaminated": len(failures),
            "n_contaminated": n_contaminated,
            "geomean_speedup": gm_speedup,
            "mean_dPSNR": mean_dpsnr,
            "worst_scene_dPSNR": worst_scene_dpsnr,
            "per_scene_dPSNR": per_scene_dpsnr,
        },
        "reasons": reasons,
    }, "final_verdict.json")
    print(f"\nVERDICT: {verdict}")
    if reasons:
        for r in reasons:
            print(f"  - {r}")


if __name__ == "__main__":
    main()
