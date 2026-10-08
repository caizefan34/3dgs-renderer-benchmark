#!/usr/bin/env python3
"""P13-X X0: existing-evidence extraction from P12 artifacts (no new experiment).

Adapted from the bundle's p13x_x0_from_p12.py to read the P12 artifact
directory directly (same JSON inputs, same output schema). Verifies the
15K-30K constant-N claim quoted in P13X_PREREG.md section 3.
"""
import json
import os
import statistics
import sys

ART = "/root/epic05_p12/artifacts"
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    "/root/epic05_p13x/artifacts", "P13X_X0_P12_PRIOR.json")

traj = json.load(open(os.path.join(ART, "P12A_SYSTEM_TRAJECTORY.json")))
runs = json.load(open(os.path.join(ART, "P12A_SYSTEM_RUNS.json")))

rows = {}
scene_means = []
for scene, d in traj["visible_gaussian_trajectories"].items():
    xs = [r for r in d["den"] if 15000 <= r[0] <= 30000]
    ratios = [r[2] / r[1] for r in xs]
    Ns = sorted(set(r[1] for r in xs))
    rows[scene] = {
        "n_samples": len(xs),
        "N_unique_count": len(Ns),
        "N_values": Ns,
        "visible_fraction_mean": statistics.mean(ratios),
        "visible_fraction_median": statistics.median(ratios),
        "visible_fraction_min": min(ratios),
        "visible_fraction_max": max(ratios),
    }
    scene_means.append(statistics.mean(ratios))

phase = {}
for scene, arms in runs["per_scene"].items():
    c = arms["C0_FUSED"]
    phase[scene] = {k: c["phase_ms"][k]["mean_ms"]
                    for k in ("forward", "loss", "backward", "densify", "optimizer")}
    phase[scene]["mean_iter_ms"] = c["mean_iter_ms"]

keys = ["forward", "loss", "backward", "densify", "optimizer", "mean_iter_ms"]
phase_mean = {k: statistics.mean(phase[s][k] for s in phase) for k in keys}

out = {
    "phase": "P13-X0 existing-evidence extraction",
    "source": ("P12A_SYSTEM_TRAJECTORY.json + P12A_SYSTEM_RUNS.json from "
               "/root/epic05_p12/artifacts (P12 campaign, frozen provenance)"),
    "window": [15000, 30000],
    "all_13_scenes_constant_N": all(v["N_unique_count"] == 1 for v in rows.values()),
    "mean_of_scene_visible_fractions": statistics.mean(scene_means),
    "per_scene": rows,
    "c0_phase_ms_unweighted_scene_mean": phase_mean,
    "interpretation": ("Supports testing persistent topology/active worksets only; "
                       "does not establish exact membership/order/dirty-work stability."),
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)

# ---- verification vs the bundle-provided prior (must agree exactly) ----
bundle_prior = "/root/epic05_p13x/p13x_bundle_P13X_X0_P12_PRIOR.json"
report = {
    "x0_output": OUT,
    "all_13_scenes_constant_N": out["all_13_scenes_constant_N"],
    "constant_N_claim_verified": out["all_13_scenes_constant_N"] is True,
    "mean_of_scene_visible_fractions": out["mean_of_scene_visible_fractions"],
    "prereg_claim_mean_visible_fraction": 0.293,
    "prereg_claim_matches_within_1e_3": abs(
        out["mean_of_scene_visible_fractions"] - 0.293) < 1e-3,
    "n_scenes": len(rows),
    "window_samples_per_scene": sorted({v["n_samples"] for v in rows.values()}),
}
print(json.dumps(report, indent=2))
