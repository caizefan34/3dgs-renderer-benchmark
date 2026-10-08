#!/usr/bin/env python3
"""P13-Y Sec.4 supplementary DIAGNOSTIC (post-hoc, NOT a pre-registered gate).

The frozen neutrality criterion compares ONE capture-ON run against ONE
capture-OFF run using ONE capture-OFF rerun as the "noise floor". In a chaotically
amplifying system a single-sample floor is a noisy estimate of its own tail, so
this script characterizes the intrinsic cross-process divergence DISTRIBUTION:
5 capture-OFF replays (identical command) + 1 capture-ON replay, all 400 iters,
all with bitwise-identical camera sequences.

Reported per metric: every OFF-OFF pair value, every ON-OFF pair value, and the
rank of each ON-OFF value inside the OFF-OFF set. Also compares end-state eval
quality (PSNR/SSIM/LPIPS) of ON against the OFF ensemble spread.

This does NOT replace or amend the frozen criterion verdict; both are reported.
"""
import argparse
import json
import os

ART = "/root/epic05_p13y/artifacts"
METRICS = ("max_abs_loss_diff", "n_loss_differing", "first_differing_iter")


def load_log(run_dir):
    with open(os.path.join(run_dir, "p13y_iter_log.json")) as f:
        d = json.load(f)
    return {int(r[0]): (float(r[1]), int(r[2])) for r in d["iter_log"]}


def pair_metrics(a, b):
    common = sorted(set(a) & set(b))
    nd = sum(1 for i in common if a[i][0] != b[i][0])
    mx = max((abs(a[i][0] - b[i][0]) for i in common), default=None)
    first = next((i for i in common if a[i][0] != b[i][0]), None)
    return {"max_abs_loss_diff": mx, "n_loss_differing": nd,
            "first_differing_iter": first, "n_common": len(common)}


def quality(run_dir):
    with open(os.path.join(run_dir, "quality.json")) as f:
        q = json.load(f)
    return {k: q.get(k) for k in ("psnr", "ssim", "lpips", "n_gaussians")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--off-dirs", nargs="+", required=True)
    ap.add_argument("--on-dir", required=True)
    ap.add_argument("--out", default=f"{ART}/P13Y_NEUTRALITY_ENSEMBLE.json")
    args = ap.parse_args()

    offs = {os.path.basename(d.rstrip("/")): load_log(d) for d in args.off_dirs}
    on_log = load_log(args.on_dir)
    on_name = os.path.basename(args.on_dir.rstrip("/"))

    off_off = {}
    names = sorted(offs)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            off_off[f"{names[i]}|{names[j]}"] = pair_metrics(offs[names[i]],
                                                            offs[names[j]])
    on_off = {f"{on_name}|{n}": pair_metrics(on_log, offs[n]) for n in names}

    def values(pairs, metric):
        return [v[metric] for v in pairs.values() if v[metric] is not None]

    dist = {}
    for m in METRICS:
        ov = values(off_off, m)
        nv = values(on_off, m)
        rank_frac = [sum(1 for x in ov if x <= y) / len(ov) for y in nv] if ov else []
        dist[m] = {
            "off_off_values": sorted(ov),
            "off_off_min": min(ov) if ov else None,
            "off_off_max": max(ov) if ov else None,
            "off_off_median": sorted(ov)[len(ov) // 2] if ov else None,
            "on_off_values": nv,
            "on_off_max": max(nv) if nv else None,
            "on_off_rank_fraction_inside_off_off": rank_frac,
            "on_off_within_off_off_envelope": bool(
                all(min(ov) <= y <= max(ov) for y in nv)) if ov and nv else None,
        }

    q_off = {n: quality(os.path.join(os.path.dirname(args.on_dir.rstrip("/")),
                                     n)) for n in names}
    q_on = quality(args.on_dir)
    qcmp = {}
    for k in ("psnr", "ssim", "lpips"):
        ov = [q_off[n][k] for n in names if q_off[n].get(k) is not None]
        v = q_on.get(k)
        qcmp[k] = {"off_values": ov, "on_value": v,
                   "off_min": min(ov) if ov else None,
                   "off_max": max(ov) if ov else None,
                   "on_inside_off_envelope": bool(ov and v is not None
                                                  and min(ov) <= v <= max(ov)),
                   "on_rank_frac_inside_off": (sum(1 for x in ov if x <= v) / len(ov)
                                               if ov and v is not None else None)}

    rep = {"phase": "P13-Y Sec.4 neutrality supplementary diagnostic",
           "date": "2026-09-27",
           "label": ("POST-HOC DIAGNOSTIC, not a pre-registered gate; the frozen "
                     "criterion verdict is in P13Y_CAPTURE_NEUTRALITY_SHORT.json "
                     "and P13Y_CAPTURE_NEUTRALITY_trio3500.json and is unchanged"),
           "design": {"n_off_replays": len(names), "n_off_off_pairs": len(off_off),
                      "n_on_off_pairs": len(on_off), "off_runs": names,
                      "on_run": on_name, "iterations": 400},
           "off_off_pairs": off_off, "on_off_pairs": on_off,
           "distribution": dist, "end_state_quality": qcmp,
           "reading": ("If every ON-OFF value lies inside the OFF-OFF envelope "
                       "(and the ON end-state quality lies inside the OFF ensemble "
                       "spread), the capture side channel shows no deviation beyond "
                       "the intrinsic cross-process divergence of the C0 recipe.")}
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps({"distribution": dist, "end_state_quality": qcmp}, indent=2))
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
