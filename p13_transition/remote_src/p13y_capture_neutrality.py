#!/usr/bin/env python3
"""P13-Y Sec.4 neutrality: capture OFF vs ON vs OFF-rerun + replay-vs-P12.

Trio (short paired replay, room): three runs of the SAME command, differing
only in whether the P13-Y capture side channel is enabled:
  OFF1 (capture off) vs ON (capture on) -> the capture effect
  OFF1 vs OFF2 (capture off rerun)      -> intrinsic cross-process noise floor
Criterion: the ON-vs-OFF1 divergence must not exceed the OFF1-vs-OFF2 noise
floor (per-metric, small tolerance). Also optionally compares a full capture
replay's per-iteration loss/N trajectory against the ORIGINAL P12 30K run's
per-100 rows over the replay prefix [1, replay_iters].
"""
import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np

ART = "/root/epic05_p13y/artifacts"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_log(p):
    d = json.load(open(p))
    return d, {int(r[0]): (float(r[1]), int(r[2])) for r in d["iter_log"]}


def pair_metrics(a, b, iters=None):
    common = sorted(set(a) & set(b) if iters is None else
                    (set(a) & set(b) & set(iters)))
    nd = sum(1 for i in common if a[i][0] != b[i][0])
    mx = max((abs(a[i][0] - b[i][0]) for i in common), default=None)
    first = next((i for i in common if a[i][0] != b[i][0]), None)
    first_ge = next((i for i in common if a[i][0] != b[i][0] and i >= 10), None)
    return {"n_common": len(common), "n_loss_exact_equal": len(common) - nd,
            "n_loss_differing": nd, "max_abs_loss_diff": mx,
            "first_differing_iter": first,
            "first_differing_iter_ge10": first_ge,
            "final_n_gs_equal": (a[common[-1]][1] == b[common[-1]][1]) if common else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--off1", required=True)
    ap.add_argument("--on", required=True)
    ap.add_argument("--off2", required=True)
    ap.add_argument("--full-run", default=None, help="full capture replay run dir")
    ap.add_argument("--p12-run", default=None, help="original P12 run dir")
    ap.add_argument("--out", default=f"{ART}/P13Y_CAPTURE_NEUTRALITY.json")
    args = ap.parse_args()

    d_off1, l_off1 = load_log(os.path.join(args.off1, "p13y_iter_log.json"))
    d_on, l_on = load_log(os.path.join(args.on, "p13y_iter_log.json"))
    d_off2, l_off2 = load_log(os.path.join(args.off2, "p13y_iter_log.json"))

    rep = {"phase": "P13-Y Sec.4 capture neutrality", "date": "2026-09-27",
           "runs": {
               "off1": {"dir": args.off1, "iters": d_off1["iterations"],
                        "final_N": d_off1["final_N"],
                        "camera_sequence_sha256": d_off1["camera_sequence_sha256"],
                        "final_model_param_sha256": d_off1["final_model_param_sha256"]},
               "on": {"dir": args.on, "iters": d_on["iterations"],
                      "final_N": d_on["final_N"],
                      "camera_sequence_sha256": d_on["camera_sequence_sha256"],
                      "final_model_param_sha256": d_on["final_model_param_sha256"]},
               "off2": {"dir": args.off2, "iters": d_off2["iterations"],
                        "final_N": d_off2["final_N"],
                        "camera_sequence_sha256": d_off2["camera_sequence_sha256"],
                        "final_model_param_sha256": d_off2["final_model_param_sha256"]}}
           }
    rep["camera_sequence_identical_all_three"] = bool(
        len({rep["runs"][k]["camera_sequence_sha256"] for k in rep["runs"]}) == 1)
    rep["final_params_identical_off1_off2"] = bool(
        rep["runs"]["off1"]["final_model_param_sha256"]
        == rep["runs"]["off2"]["final_model_param_sha256"])
    rep["final_params_identical_on_off1"] = bool(
        rep["runs"]["on"]["final_model_param_sha256"]
        == rep["runs"]["off1"]["final_model_param_sha256"])
    rep["noise_floor_off1_vs_off2"] = pair_metrics(l_off1, l_off2)
    rep["capture_effect_on_vs_off1"] = pair_metrics(l_on, l_off1)
    rep["capture_effect_on_vs_off2"] = pair_metrics(l_on, l_off2)

    nf, ce = rep["noise_floor_off1_vs_off2"], rep["capture_effect_on_vs_off1"]
    def le(a, b, tol=1.0):
        if a is None or b is None:
            return None
        return bool(a <= b * tol + 1e-12)
    checks = {
        "n_loss_differing": le(ce["n_loss_differing"], max(nf["n_loss_differing"], 1)),
        "max_abs_loss_diff": le(ce["max_abs_loss_diff"],
                                max(nf["max_abs_loss_diff"] or 0.0, 1e-12)),
        "first_differing_iter": le(ce["first_differing_iter_ge10"],
                                   max(nf["first_differing_iter_ge10"] or 0, 1)),
    }
    rep["criterion"] = {
        "rule": ("capture effect (ON vs OFF1) must not exceed the intrinsic "
                 "cross-process noise floor (OFF1 vs OFF2) per metric"),
        "checks": checks, "pass": all(v for v in checks.values() if v is not None),
    }

    if args.full_run and args.p12_run:
        d_full, l_full = load_log(os.path.join(args.full_run, "p13y_iter_log.json"))
        p12 = json.load(open(os.path.join(args.p12_run, "results.json")))
        p12_rows = {int(r["iter"]): r for r in p12.get("per_iter", [])}
        iters = sorted(i for i in p12_rows if i <= d_full["iterations"])
        cmp_rows, nd, mx, first = [], 0, 0.0, None
        for i in iters:
            if i not in l_full:
                continue
            lv = l_full[i][0]
            dv = abs(lv - float(p12_rows[i]["loss"]))
            n_gs_eq = l_full[i][1] == int(p12_rows[i]["n_gs"])
            if lv != float(p12_rows[i]["loss"]):
                nd += 1
                if first is None:
                    first = i
            mx = max(mx, dv)
            cmp_rows.append({"iter": i, "replay_loss": lv,
                             "p12_loss": float(p12_rows[i]["loss"]),
                             "abs_diff": dv,
                             "replay_n_gs": l_full[i][1],
                             "p12_n_gs": int(p12_rows[i]["n_gs"]),
                             "n_gs_equal": bool(n_gs_eq)})
        p12_npy = os.path.join(args.p12_run, "camera_sequence.npy")
        rep["full_replay_vs_p12_original"] = {
            "replay_run": args.full_run, "p12_run": args.p12_run,
            "n_iter_rows_compared": len(cmp_rows),
            "n_loss_differing": nd,
            "max_abs_loss_diff": mx,
            "first_differing_iter": first,
            "n_gs_equal_all_rows": bool(all(r["n_gs_equal"] for r in cmp_rows)),
            "camera_sequence_sha256_match": bool(
                sha256_file(p12_npy) == d_full["camera_sequence_sha256"])
            if os.path.exists(p12_npy) else None,
            "first_rows": cmp_rows[:5], "last_rows": cmp_rows[-5:],
        }
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps({k: rep[k] for k in (
        "camera_sequence_identical_all_three", "final_params_identical_off1_off2",
        "final_params_identical_on_off1", "noise_floor_off1_vs_off2",
        "capture_effect_on_vs_off1", "criterion")}, indent=2))
    if "full_replay_vs_p12_original" in rep:
        fr = rep["full_replay_vs_p12_original"]
        print(json.dumps({k: fr[k] for k in (
            "n_iter_rows_compared", "n_loss_differing", "max_abs_loss_diff",
            "first_differing_iter", "n_gs_equal_all_rows",
            "camera_sequence_sha256_match")}, indent=2))
    print("WROTE", args.out)
    sys.exit(0 if rep["criterion"]["pass"] else 4)


if __name__ == "__main__":
    main()
