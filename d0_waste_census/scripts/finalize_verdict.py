#!/usr/bin/env python3
"""D0 verdict assembly: builds the canonical scene summary and then invokes
the bundle-provided verdict tool VERBATIM (d0_verdict.py, sha256
393af7e019a587e220a44d39b3aeb89d6329f6ffb4ecef3bb2b0199fc2ddca96).

Inputs (artifacts dir):
  D0_TRACE_VALIDITY_30k.json   per-scene trace validity
  D0_COST_ACCOUNTING.json      sample reconstruction error
  D0_HEADROOM.json             R_upper + bootstrap CI
  D0_PREDICTABILITY_ROWS.json  optional, flat logreg rows

Outputs:
  D0_SCENE_SUMMARY.json        canonical tool input (also a return artifact)
  D0_VERDICT.json              canonical tool output
"""
import argparse
import json
import os
import subprocess
import sys

ART = "/root/epic05_d0/artifacts"
HERE = os.path.dirname(os.path.abspath(__file__))


def load(name, art=ART):
    with open(os.path.join(art, name)) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art-dir", default=ART)
    args = ap.parse_args()
    art = args.art_dir
    tv = load("D0_TRACE_VALIDITY_30k.json", art)
    cost = load("D0_COST_ACCOUNTING.json", art)
    head = load("D0_HEADROOM.json", art)

    reasons = tv.get("reasons") or {}
    summary = []
    for s in ("room", "bicycle"):
        h = head["runs"][s]
        a = cost["runs"][s]["accounting"]
        summary.append({
            "scene": s,
            "trace_valid": bool(not reasons.get(s)),
            "trace_valid_reasons": reasons.get(s, []),
            "trace_validity_verdict": tv.get("verdict"),
            "sample_reconstruction_error": a["reconstruction_error"],
            "reconstruction_gate_ok": a["reconstruction_gate_ok"],
            "sumcheck_gate_ok": a["sumcheck_gate_ok"],
            "R_upper": h["R_upper"],
            "R_upper_ci95_low": h["R_upper_ci95"][0],
            "R_upper_ci95_high": h["R_upper_ci95"][1],
            "S_upper_at_ci95_low": h["S_upper_at_ci95_low"],
            "t_star": h["t_star"],
            "r_I_dead": h["r_I_dead"], "r_N_dead": h["r_N_dead"],
        })
    with open(os.path.join(art, "D0_SCENE_SUMMARY.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Canonical alias: prereg §13 lists D0_TRACE_VALIDITY.json; the 30K
    # analyzer writes the validated artifact as D0_TRACE_VALIDITY_30k.json.
    with open(os.path.join(art, "D0_TRACE_VALIDITY.json"), "w") as f:
        json.dump(tv, f, indent=2, default=str)

    pred = os.path.join(art, "D0_PREDICTABILITY_ROWS.json")
    cmd = [sys.executable, os.path.join(HERE, "d0_verdict.py"),
           "--scene-summary", os.path.join(art, "D0_SCENE_SUMMARY.json"),
           "--out", os.path.join(art, "D0_VERDICT.json")]
    if os.path.exists(pred):
        cmd += ["--predictability", pred]
    rc = subprocess.call(cmd)
    with open(os.path.join(art, "D0_VERDICT.json")) as f:
        print(f.read())
    return rc


if __name__ == "__main__":
    sys.exit(main())
