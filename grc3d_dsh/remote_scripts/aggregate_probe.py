#!/usr/bin/env python3
"""M1 aggregation: per-group kappa, analytic variances, block bootstrap CIs.

Consumes the 32-block probe JSONs (one per bundle) and produces:
  - per (scene, iteration, lag): pooled-over-cameras per-group totals,
    kappa = sum||g_f-g_p||^2 / sum||g_f||^2 with a block bootstrap 95% CI,
    raw (un-anchored coarse) kappa, analytic variances for p=1/2,1/4,1/8,
    and the pre-registered ratio anchored p=1/8 vs plain p=1/4 = 7*kappa/3.
  - per (scene, iteration, camera, lag) point estimates (no CI).
  - near-zero denominator accounting (None is never treated as 0).
  - preconditioned kappa merge from precond_*.json when present.
  - a machine-readable pre-registration evaluation (GO / DROP trigger checks).

The bootstrap resamples the pooled block rows within a (scene, iteration, lag)
cell; it characterizes sampling variability over the fixed block grid, not the
whole-image exact kappa.
"""
import argparse
import csv
import glob as globmod
import json
import os
import re

import numpy as np

GROUPS = ["means", "quats", "log_scales", "opacity_logits", "sh0", "shN"]
ALL_UNITS = "all_raw_units"
P_LEVELS = [0.5, 0.25, 0.125]
THRESH = 3.0 / 7.0
NAME_RE = re.compile(r"probe_(?P<scene>[a-z0-9]+)_it(?P<it>\d+)_cam(?P<cam>\d+)"
                     r"_lag(?P<lag>\d+)_b(?P<b>\d+)\.json$")
PRECOND_RE = re.compile(r"precond_(?P<scene>[a-z0-9]+)_it(?P<it>\d+)_cam(?P<cam>\d+)"
                        r"_lag(?P<lag>\d+)_curv_b(?P<b>\d+)\.json$")


def load_probe(path):
    with open(path) as f:
        d = json.load(f)
    rows = []
    for blk in d["per_block"]:
        row = {"index": blk["index"], "center": blk["center"]}
        for g in GROUPS:
            row[g] = blk["groups"][g]
        rows.append(row)
    return d, rows


def cell_stats(rows):
    """Point totals + bootstrap CI over pooled block rows."""
    out = {}
    for g in GROUPS + [ALL_UNITS]:
        if g == ALL_UNITS:
            fine = np.array([sum(r[x]["fine_energy"] for x in GROUPS) for r in rows],
                            dtype=np.float64)
            diff = np.array([sum(r[x]["difference_energy"] for x in GROUPS) for r in rows],
                            dtype=np.float64)
            raw = np.array([sum(r[x]["raw_difference_energy"] for x in GROUPS) for r in rows],
                           dtype=np.float64)
        else:
            fine = np.array([r[g]["fine_energy"] for r in rows], dtype=np.float64)
            diff = np.array([r[g]["difference_energy"] for r in rows], dtype=np.float64)
            raw = np.array([r[g]["raw_difference_energy"] for r in rows], dtype=np.float64)
        out[g] = {"fine_energy": float(fine.sum()), "difference_energy": float(diff.sum()),
                  "raw_difference_energy": float(raw.sum()),
                  "n_blocks": int(len(rows)),
                  "n_blocks_fine_energy_lt_1e-30": int((fine < 1e-30).sum()),
                  "n_blocks_diff_energy_lt_1e-30": int((diff < 1e-30).sum())}
    out[ALL_UNITS]["note"] = ("sum over the six parameter groups' raw units; not a physical "
                              "quantity, reported as context only")
    return out


def _arrays(rows, key):
    out = {}
    for g in GROUPS:
        out[g] = np.array([r[g][key] for r in rows], dtype=np.float64)
    out[ALL_UNITS] = np.sum([out[g] for g in GROUPS], axis=0)
    return out


def bootstrap_cis(rows, rng, B):
    n = len(rows)
    idx_all = np.arange(n)
    fine = _arrays(rows, "fine_energy")
    diff = _arrays(rows, "difference_energy")
    raw = _arrays(rows, "raw_difference_energy")
    samples = {g: np.full(B, np.nan) for g in GROUPS + [ALL_UNITS]}
    samples_raw = {g: np.full(B, np.nan) for g in GROUPS + [ALL_UNITS]}
    for b in range(B):
        sel = rng.choice(idx_all, size=n, replace=True)
        for g in GROUPS + [ALL_UNITS]:
            den = fine[g][sel].sum()
            if den >= 1e-30:
                samples[g][b] = diff[g][sel].sum() / den
                samples_raw[g][b] = raw[g][sel].sum() / den
    out = {}
    for g in GROUPS + [ALL_UNITS]:
        s = samples[g][~np.isnan(samples[g])]
        sr = samples_raw[g][~np.isnan(samples_raw[g])]
        out[g] = {
            "kappa_ci95": ([float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]
                           if len(s) > 0 else None),
            "raw_kappa_ci95": ([float(np.percentile(sr, 2.5)), float(np.percentile(sr, 97.5))]
                               if len(sr) > 0 else None),
            "n_bootstrap_valid": int(len(s)),
            "n_bootstrap_denominator_lt_1e-30":
                int(B - len(s)) if len(s) < B else 0,
        }
    return out


def analytic_block(cell, cis):
    """Per-group analytic variance tables and pre-registered ratios."""
    out = {}
    for g in GROUPS + [ALL_UNITS]:
        c = cell[g]
        den = c["fine_energy"]
        entry = dict(c)
        kappa = c["difference_energy"] / den if den >= 1e-30 else None
        raw_kappa = c["raw_difference_energy"] / den if den >= 1e-30 else None
        entry["kappa"] = kappa
        entry["raw_kappa"] = raw_kappa
        entry["ci"] = cis[g]
        entry["non_degenerate"] = bool(den >= 1e-30)
        for p in P_LEVELS:
            f = (1.0 - p) / p
            entry[f"var_plain_p{p:g}"] = f * c["fine_energy"] if den >= 1e-30 else None
            entry[f"var_anchored_p{p:g}"] = (f * c["difference_energy"]
                                             if den >= 1e-30 else None)
            entry[f"var_plain_raw_p{p:g}"] = f * c["fine_energy"] if den >= 1e-30 else None
            entry[f"var_raw_coarse_p{p:g}"] = (f * c["raw_difference_energy"]
                                               if den >= 1e-30 else None)
        if kappa is not None:
            entry["ratio_anchored_p1_8_vs_plain_p1_4"] = 7.0 * kappa / 3.0
            entry["threshold_3_over_7"] = THRESH
            entry["point_le_3over7"] = bool(kappa <= THRESH)
            hi = cis[g]["kappa_ci95"]
            entry["ci_upper_le_3over7"] = bool(hi is not None and hi[1] <= THRESH)
            entry["ci_lower_gt_3over7"] = bool(hi is not None and hi[0] > THRESH)
            entry["kappa_ge_1"] = bool(kappa >= 1.0)
        else:
            entry["ratio_anchored_p1_8_vs_plain_p1_4"] = None
            entry["point_le_3over7"] = None
            entry["ci_upper_le_3over7"] = None
            entry["ci_lower_gt_3over7"] = None
            entry["kappa_ge_1"] = None
        out[g] = entry
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="outputs/probe_*_b32.json")
    ap.add_argument("--precond-glob", default="outputs/precond_*_b32.json")
    ap.add_argument("--bootstrap", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    files = sorted(globmod.glob(args.glob))
    assert files, "no probe files matched"

    out = {"bootstrap_B": args.bootstrap, "bootstrap_seed": args.seed,
           "threshold_3_over_7": THRESH, "cells": {}, "per_camera": {},
           "preconditioned": {}, "preregistered_evaluation": {}, "files": files}

    cells = {}
    for f in files:
        m = NAME_RE.search(os.path.basename(f))
        assert m, f"unparseable probe filename {f}"
        scene, it, cam, lag = m["scene"], int(m["it"]), int(m["cam"]), int(m["lag"])
        d, rows = load_probe(f)
        key = (scene, it, lag)
        cells.setdefault(key, {"rows": [], "cameras": {}})
        cells[key]["cameras"].setdefault(cam, []).extend(rows)
        cells[key]["rows"].extend(rows)
        out["per_camera"][f"{scene}_it{it}_cam{cam}_lag{lag}"] = {
            "file": f, "bundle": d.get("bundle_sha256"), "energy_scope": d.get("energy_scope"),
            "num_gaussians": d.get("num_gaussians"), "sh_degree": d.get("sh_degree"),
            "groups": cell_stats(rows)}

    for (scene, it, lag), c in sorted(cells.items()):
        cis = bootstrap_cis(c["rows"], rng, args.bootstrap)
        entry = {"scene": scene, "iteration": it, "lag": lag,
                 "n_block_rows": len(c["rows"]),
                 "cameras": sorted(c["cameras"]),
                 "groups": analytic_block(cell_stats(c["rows"]), cis)}
        out["cells"][f"{scene}_it{it}_lag{lag}"] = entry

    # ---------------- preconditioned (anchor-state) kappa -------------------
    for f in sorted(globmod.glob(args.precond_glob)):
        m = PRECOND_RE.search(os.path.basename(f))
        if not m:
            continue
        with open(f) as fh:
            d = json.load(fh)
        key = f"{m['scene']}_it{m['it']}_cam{m['cam']}_lag{m['lag']}"
        entry = {"file": f, "v_source": d.get("v_source", {}),
                 "state": "current_state_current_v (topology-exact; anchor-state "
                          "variant is a mathematical identity, see JSON)",
                 "groups": {}}
        for g in GROUPS:
            t = d["totals"][g]
            den, den_v = t["fine_energy"], t["precond_fine_energy"]
            entry["groups"][g] = {
                "fine_energy": den, "difference_energy": t["difference_energy"],
                "precond_fine_energy": den_v,
                "precond_difference_energy": t["precond_difference_energy"],
                "kappa": (t["difference_energy"] / den) if den >= 1e-30 else None,
                "precond_kappa": (t["precond_difference_energy"] / den_v)
                                 if den_v >= 1e-30 else None,
                "n_blocks": t.get("n_blocks"),
                "n_blocks_v_zero_fine_lt_1e-30": t.get("n_blocks_precond_fine_lt_1e-30"),
            }
        out["preconditioned"][key] = entry

    # ---------------- pre-registered evaluation -----------------------------
    ev = out["preregistered_evaluation"]
    ev["drop_trigger_q1_4"] = {"rule": ("both scenes at lag16 have >=1 non-degenerate "
                                        "group with kappa>=1 and CI lower bound > 3/7"),
                               "per_scene": {}, "triggered": None}
    ev["go_check"] = {"rule": ("M0 pass AND both scenes at lag16 every non-degenerate "
                               "group has kappa CI upper bound <= 3/7 AND cost feasible"),
                      "per_scene": {}, "kappa_criterion_met": None}
    scenes = sorted({k[0] for k in cells})
    for scene in scenes:
        key = f"{scene}_it5000_lag16"
        cell = out["cells"].get(key)
        if not cell:
            continue
        grp = cell["groups"]
        nondeg = [g for g in GROUPS if grp[g]["non_degenerate"]]
        ge1 = [g for g in nondeg if grp[g]["kappa_ge_1"]]
        drop = [g for g in ge1 if grp[g]["ci_lower_gt_3over7"]]
        ev["drop_trigger_q1_4"]["per_scene"][scene] = {
            "non_degenerate_groups": nondeg, "groups_kappa_ge_1": ge1,
            "groups_kappa_ge_1_and_ci_lo_gt_3over7": drop}
        ev["go_check"]["per_scene"][scene] = {
            "non_degenerate_groups": nondeg,
            "groups_ci_upper_le_3over7": [g for g in nondeg
                                          if grp[g]["ci_upper_le_3over7"]],
            "groups_ci_upper_gt_3over7": [g for g in nondeg
                                          if grp[g]["ci_upper_le_3over7"] is False]}
    drop_scenes = {s: bool(v["groups_kappa_ge_1_and_ci_lo_gt_3over7"])
                   for s, v in ev["drop_trigger_q1_4"]["per_scene"].items()}
    ev["drop_trigger_q1_4"]["triggered"] = (all(drop_scenes.values())
                                            if drop_scenes else None)
    go_scenes = {s: (len(v["groups_ci_upper_gt_3over7"]) == 0 and v["non_degenerate_groups"])
                 for s, v in ev["go_check"]["per_scene"].items()}
    ev["go_check"]["kappa_criterion_met"] = (all(go_scenes.values())
                                             if go_scenes else None)
    ev["note"] = ("kappa criterion only; cost feasibility is evaluated in the cost "
                  "profile and the final verdict")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, allow_nan=False)
    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "iteration", "lag", "group", "fine_energy",
                    "difference_energy", "raw_difference_energy", "kappa", "raw_kappa",
                    "kappa_ci_lo", "kappa_ci_hi", "ci_upper_le_3over7",
                    "ratio_anchored_p1_8_vs_plain_p1_4", "var_plain_p1_8",
                    "var_anchored_p1_8", "var_plain_p0.5", "var_anchored_p0.5",
                    "var_plain_p0.25", "var_anchored_p0.25",
                    "n_blocks_fine_energy_lt_1e-30"])
        for key, cell in sorted(out["cells"].items()):
            for g in GROUPS + [ALL_UNITS]:
                e = cell["groups"][g]
                ci = (e["ci"]["kappa_ci95"] or [None, None])
                w.writerow([cell["scene"], cell["iteration"], cell["lag"], g,
                            e["fine_energy"], e["difference_energy"],
                            e["raw_difference_energy"], e["kappa"], e["raw_kappa"],
                            ci[0], ci[1], e["ci_upper_le_3over7"],
                            e["ratio_anchored_p1_8_vs_plain_p1_4"],
                            e["var_plain_p0.125"], e["var_anchored_p0.125"],
                            e["var_plain_p0.5"], e["var_anchored_p0.5"],
                            e["var_plain_p0.25"], e["var_anchored_p0.25"],
                            e["n_blocks_fine_energy_lt_1e-30"]])
    print(f"[agg] wrote {args.out} and {args.csv}")
    for key, cell in sorted(out["cells"].items()):
        if cell["lag"] != 16:
            continue
        line = [f"{key}:"]
        for g in GROUPS:
            e = cell["groups"][g]
            ci = e["ci"]["kappa_ci95"]
            line.append(f"{g}={e['kappa'] if e['kappa'] is None else round(e['kappa'], 4)}"
                        f"{'' if ci is None else f'[{ci[0]:.3f},{ci[1]:.3f}]'}")
        print("  " + " ".join(line))


if __name__ == "__main__":
    main()
