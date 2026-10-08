import json, os, math, hashlib
OUT = "/root/ig0_run/out"
REF = "/root/ig0_run"

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

def write(name, obj):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=2)

probes = json.load(open(os.path.join(OUT, "IG0_PROBES.json")))
th = json.load(open(os.path.join(OUT, "IG0_TRUE_HOLDOUT.json")))

cells = ["room@10000", "room@15000", "bicycle@10000", "bicycle@15000"]

# ---- camera semantics (authoritative from probe/trueholdout artifacts) ----
cam_sem = {"artifact": "IG0_CAMERA_SEMANTICS.json", "prereg_section": "2,4",
           "true_holdout_rule": "original camera index i % 8 == 0 (corrected SB0)",
           "train_pool_rule": "all remaining cameras (i % 8 != 0)",
           "selection": "midpoint-stratified, deterministic", "cells": {}}
for cell in cells:
    recs = [r for r in probes if r["cell"] == cell]
    src = sorted(set(int(r["source_cam"]) for r in recs))
    hold = sorted(set(int(x) for x in th["per_cell"][cell]["holdout"]))
    n = max(int(r["n_gaussians"]) for r in recs)  # placeholder; n_cameras below
    ncam = max(src + hold) + 1 if src else 0
    cam_sem["cells"][cell] = {
        "n_source_train": len(src), "selected_source_train_ids": src,
        "n_true_holdout": len(hold), "selected_true_holdout_ids": hold,
        "train_intersect_holdout": sorted(set(src) & set(hold)),
        "true_holdout_used_for_gradient_or_update": False,
    }
write("IG0_CAMERA_SEMANTICS.json", cam_sem)

# ---- GPU radius parity aggregation (real native Adam displacement, per probe) ----
rp = {"artifact": "IG0_RADIUS_PARITY.json", "prereg_section": "10.4,10.5,15",
      "gradient_source": "frozen_renderer_native_Adam_displacement",
      "per_probe": {}, "aggregates": {}}
for cell in cells:
    recs = [r for r in probes if r["cell"] == cell]
    rp["per_probe"][cell] = [{
        "probe": r["probe"], "source_cam": r["source_cam"],
        "fr_global_radius_rel": r["fr_global_radius_rel"],
        "fr_local_radius_median": r["fr_local_radius_median"],
        "fr_local_radius_p99": r["fr_local_radius_p99"],
        "fr_local_n_compared": r["fr_local_n_compared"],
        "correctness_pass": r["correctness_pass"],
    } for r in recs]
    rp["aggregates"][cell] = {
        "fr_global_radius_rel_worst": max(r["fr_global_radius_rel"] for r in recs),
        "fr_local_radius_median_worst": max(r["fr_local_radius_median"] for r in recs),
        "fr_local_radius_p99_worst": max(r["fr_local_radius_p99"] for r in recs),
        "all_correctness_pass": all(r["correctness_pass"] for r in recs),
        "all_nan_inf_zero": all(r["nan_inf_count"] == 0 for r in recs),
    }
write("IG0_RADIUS_PARITY.json", rp)

# ---- copy CPU-reference artifacts into out/ ----
import shutil
for nm in ["IG0_FISHER_REFERENCE.json", "IG0_QUATERNION_CHECK.json"]:
    src = os.path.join(REF, nm)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(OUT, nm))

# ---- validate all JSON parse + no NaN/Inf ----
def guard(x):
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            raise ValueError("NaN/Inf in JSON")
    elif isinstance(x, dict):
        for v in x.values():
            guard(v)
    elif isinstance(x, list):
        for v in x:
            guard(v)
names = ["IG0_SOURCE_IDENTITY.json","IG0_ENVIRONMENT.json","IG0_CAMERA_SEMANTICS.json",
         "IG0_FISHER_REFERENCE.json","IG0_QUATERNION_CHECK.json","IG0_RADIUS_PARITY.json",
         "IG0_PROBES.json","IG0_TRUE_HOLDOUT.json","IG0_SUMMARY.json","IG0_VERDICT.json"]
allok = True
for nm in names:
    p = os.path.join(OUT, nm)
    if not os.path.exists(p):
        print("MISSING", nm); allok = False; continue
    try:
        d = json.load(open(p), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
        guard(d)
        print("OK", nm)
    except Exception as e:
        print("BAD", nm, e); allok = False
print("ALL_JSON_OK", allok)
