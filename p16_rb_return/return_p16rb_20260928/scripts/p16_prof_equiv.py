#!/usr/bin/env python3
"""P16-RB: camera/loss sequence neutrality check for the profiling instrumentation.

Three short seeded legs on the copied (instrumented) tree:
  A1, A2 = control   (P16_PROF=0, spans/profiler inert)
  B      = enabled   (P16_PROF=1, windows disabled by out-of-range native N)
If the camera sequence differs at all, or the A1-B loss divergence exceeds the
A1-A2 control scatter by a wide margin, the instrumented run is flagged.
Stdlib only.
"""
import argparse
import csv
import json
import sys
from pathlib import Path


def load(p):
    cams, losses, iters = [], [], []
    with open(p, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            iters.append(int(row["iteration"]))
            cams.append(row["cameras"])
            try:
                losses.append(float(row["loss"]))
            except ValueError:
                losses.append(float("nan"))
    return iters, cams, losses


def diff(a, b):
    n = min(len(a), len(b))
    ds = [abs(a[i] - b[i]) for i in range(n)]
    return {"n_compared": n, "max_abs": max(ds) if ds else None,
            "mean_abs": (sum(ds) / n) if n else None,
            "first_abs": ds[0] if ds else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a1", required=True)
    ap.add_argument("--a2", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    it1, c1, l1 = load(args.a1)
    it2, c2, l2 = load(args.a2)
    it3, c3, l3 = load(args.b)

    cam_12 = c1 == c2
    cam_13 = c1 == c3
    cam_123_len = len(c1) == len(c2) == len(c3)

    d12 = diff(l1, l2)
    d13 = diff(l1, l3)

    scatter = d12["max_abs"] or 0.0
    effect = d13["max_abs"] or 0.0
    threshold = max(20.0 * scatter, 1e-4)

    if not (cam_12 and cam_13 and cam_123_len):
        status = "FAIL_CAMERA_SEQUENCE"
    elif effect <= threshold:
        status = "PASS"
    else:
        status = "REVIEW_LOSS_DIVERGENCE"

    rep = {
        "label": args.label,
        "status": status,
        "legs": {"a1": args.a1, "a2": args.a2, "b": args.b},
        "n_iterations": {"a1": len(c1), "a2": len(c2), "b": len(c3)},
        "camera_sequence": {"a1_eq_a2": cam_12, "a1_eq_b": cam_13, "equal_lengths": cam_123_len},
        "loss_control_scatter_a1_a2": d12,
        "loss_instrumentation_effect_a1_b": d13,
        "decision_rule": {
            "pass_if": "camera sequences identical AND max_abs(a1-b) <= max(20*max_abs(a1-a2), 1e-4)",
            "threshold_used": threshold,
        },
    }
    Path(args.out).write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("label", "status", "camera_sequence")}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
