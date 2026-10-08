"""Supplementary P-TRANSITION summary: preregistered state-level items not emitted
by the frozen analyzer (median audit_per_train, per-seed R_eta lists).

Derived directly from the 12 raw unit JSONs; no thresholds, no selection changes.
"""
import glob
import json


def median(xs):
    ys = sorted(xs)
    n = len(ys)
    return ys[n // 2] if n % 2 else (ys[n // 2 - 1] + ys[n // 2]) / 2


rows = []
for f in sorted(glob.glob("/root/epic05_ptransition/results/transition_*.json")):
    d = json.load(open(f))
    if d.get("status") != "P_TRANSITION_UNIT_COMPLETE":
        continue
    rows.append({
        "scene": d["scene"], "iteration": int(d["iteration"]), "seed": int(d["seed"]),
        "R_eta": float(d["R_eta"]),
        "screen_lr": d["best_by_eta"]["screen"]["lr"],
        "geometry_lr": d["best_by_eta"]["geometry"]["lr"],
        "screen_audit_per_train": d["best_by_eta"]["screen"]["audit_per_train"],
        "geometry_audit_per_train": d["best_by_eta"]["geometry"]["audit_per_train"],
        "screen_memory_ratio": d["screen_memory_ratio"],
    })

groups = {}
for r in rows:
    groups.setdefault((r["scene"], r["iteration"]), []).append(r)

summary = {}
for (scene, it), rs in sorted(groups.items()):
    summary[f"{scene}:{it}"] = {
        "n": len(rs),
        "median_R_eta": median([r["R_eta"] for r in rs]),
        "seeds_R_gt_1": sum(r["R_eta"] > 1.0 for r in rs),
        "median_screen_audit_per_train": median([r["screen_audit_per_train"] for r in rs]),
        "median_geometry_audit_per_train": median([r["geometry_audit_per_train"] for r in rs]),
        "screen_audit_no_worse_count": sum(
            r["screen_audit_per_train"] <= r["geometry_audit_per_train"] for r in rs),
        "median_screen_memory_ratio": median([r["screen_memory_ratio"] for r in rs]),
        "per_seed_R_eta": {str(r["seed"]): r["R_eta"] for r in rs},
        "screen_best_lrs": sorted({r["screen_lr"] for r in rs}),
        "geometry_best_lrs": sorted({r["geometry_lr"] for r in rs}),
    }

out = {
    "note": "derived from raw unit JSONs; complements the frozen analyzer output",
    "n_units": len(rows),
    "groups": summary,
}
dest = "/root/epic05_ptransition/PTRANSITION_SUPP_SUMMARY.json"
open(dest, "w").write(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
