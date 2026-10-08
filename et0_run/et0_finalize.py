#!/usr/bin/env python3
"""ET0 finalize (ET0_PREREG.md S13): emit the required named artifacts by slicing
the frozen per-cell SWEEP/SAFESTEP/ENVIRONMENT JSONs. Pure post-processing; NO
GPU. Does NOT alter any measured value or the analyzer-owned verdict."""
import json, os, sys, hashlib, glob

OUT = sys.argv[1] if len(sys.argv) > 1 else "/root/epic05_moonshot/et0/full"
CELLS = ["room_10000", "room_15000", "room_20000",
         "bicycle_10000", "bicycle_15000", "bicycle_20000"]
AT = "@"


def jload(name):
    with open(os.path.join(OUT, name)) as f:
        return json.load(f)


def jdump(obj, name):
    path = os.path.join(OUT, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: float(o))
    print("[wrote]", path, flush=True)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


env = {c: jload(f"ET0_ENVIRONMENT_{c}.json") for c in CELLS}
swp = {c: jload(f"ET0_SWEEP_{c}.json") for c in CELLS}
saf = {c: jload(f"ET0_SAFESTEP_{c}.json") for c in CELLS}

# ---- ET0_SOURCE_IDENTITY.json ----
ident = {"prereg_pins": {
             "c0_ext_sha256": "d26cccf46c7932bfb5fd44ee05e553af27e17c93eaffb7dacf3197258a513f11",
             "core_so_sha256": "935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e"},
         "cells": {}}
for c in CELLS:
    e = env[c]
    ident["cells"][c.replace("_", AT)] = {
        "ckpt_path": e["ckpt_path"], "ckpt_sha256": e["ckpt_sha256"],
        "ckpt_iter": e["ckpt_iter"], "num_points": e["num_points"],
        "so_sha256": e["so_sha256"], "core_sha256": e["core_sha256"],
        "active_sh_degree": e["active_sh_degree"],
        "spatial_lr_scale": e["spatial_lr_scale"],
        "group_order": e["group_order"]}
jdump(ident, "ET0_SOURCE_IDENTITY.json")

# ---- ET0_CAMERA_SEMANTICS.json ----
cam = {"rule": env[CELLS[0]]["camera_rule"],
       "split": "train: i%8!=0 ; holdout: i%8==0 ; midpoint-stratified(8)",
       "cells": {}}
for c in CELLS:
    e = env[c]
    cam["cells"][c.replace("_", AT)] = {
        "n_cams": e["n_cams"], "train_pool_size": e["train_pool_size"],
        "holdout_size": e["holdout_size"],
        "source_cams": e["source_cams"], "holdout_cams": e["holdout_cams"]}
jdump(cam, "ET0_CAMERA_SEMANTICS.json")

# ---- ET0_ADAM_DIRECTIONS.json ----
adam = {"beta1": 0.9, "beta2": 0.999, "eps": 1e-15,
        "definition": "d = theta_adam - theta_0 (one native Adam step on clone)",
        "cells": {}}
for c in CELLS:
    e = env[c]
    adam["cells"][c.replace("_", AT)] = {
        "per_group_lr": e["adam"]["per_group_lr"],
        "per_group_step": e["adam"]["per_group_step"],
        "probes": [{"probe": p["probe"], "cam": p["cam"],
                    "gd": p["gd"], "L_src_theta0": p["L_src_theta0"]}
                   for p in swp[c]["probes"]]}
jdump(adam, "ET0_ADAM_DIRECTIONS.json")

# ---- ET0_ALPHA_LOSSES.json ----
losses = {"alphas": swp[CELLS[0]]["alphas"], "cells": {}}
for c in CELLS:
    losses["cells"][c.replace("_", AT)] = [
        {"probe": p["probe"], "cam": p["cam"],
         "rows": [{"alpha": r["alpha"], "L_src": r["L_src"],
                   "pred": r["pred"], "real": r["real"]} for r in p["rows"]]}
        for p in swp[c]["probes"]]
jdump(losses, "ET0_ALPHA_LOSSES.json")

# ---- ET0_RENDERER_CHURN.json ----
churn = {"tau_C": 0.10,
         "score": "C = 0.5*C_tile + 0.5*C_order (frozen, untuned)",
         "base_key": "N+1 (fixed total Gaussians + 1)", "cells": {}}
for c in CELLS:
    churn["cells"][c.replace("_", AT)] = [
        {"probe": p["probe"], "cam": p["cam"],
         "rows": [{"alpha": r["alpha"], "C_tile": r["C_tile"],
                   "C_order": r["C_order"], "C": r["C"],
                   "insdel": r["insdel"]} for r in p["rows"]]}
        for p in swp[c]["probes"]]
jdump(churn, "ET0_RENDERER_CHURN.json")

# ---- ET0_HELLINGER.json ----
hel = {"metric": "squared Hellinger N(mu0,S0)||N(mu1,S1), fp64; H=mean_i H2_i",
       "cells": {}}
for c in CELLS:
    hel["cells"][c.replace("_", AT)] = [
        {"probe": p["probe"], "cam": p["cam"],
         "H_at_alpha1": p["H_at_alpha1"],
         "rows": [{"alpha": r["alpha"], "H": r["H"]} for r in p["rows"]]}
        for p in swp[c]["probes"]]
jdump(hel, "ET0_HELLINGER.json")

# ---- ET0_MODEL_ERROR.json ----
err = {"definition": "E = |pred-real|/(|pred|+|real|+1e-12); "
         "breakdown = E>=0.50 OR sign_failure",
       "cells": {}}
for c in CELLS:
    err["cells"][c.replace("_", AT)] = [
        {"probe": p["probe"], "cam": p["cam"],
         "rho_EC": p["rho_EC"], "rho_EH": p["rho_EH"],
         "delta_rho": p["delta_rho"],
         "AUC_C": p["AUC_C"], "AUC_H": p["AUC_H"],
         "delta_AUC": p["delta_AUC"],
         "n_positive_labels": p["n_positive_labels"],
         "rows": [{"alpha": r["alpha"], "E": r["E"],
                   "sign_failure": r["sign_failure"]} for r in p["rows"]]}
        for p in swp[c]["probes"]]
jdump(err, "ET0_MODEL_ERROR.json")

# ---- ET0_TRUE_HOLDOUT.json ----
hold = {"cells": {}}
for c in CELLS:
    hold["cells"][c.replace("_", AT)] = saf[c]["probes"]
jdump(hold, "ET0_TRUE_HOLDOUT.json")

# ---- SHA256 manifest over all ET0_*.json artifacts ----
arts = sorted(glob.glob(os.path.join(OUT, "ET0_*.json")))
man = {"artifacts": {os.path.basename(a): sha256_file(a) for a in arts}}
jdump(man, "ET0_ARTIFACT_MANIFEST.json")
print("finalize done:", len(arts), "artifacts")
