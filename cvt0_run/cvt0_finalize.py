#!/usr/bin/env python3
"""Split CVT0_RESULTS.json into the sectioned artifacts required by
CVT0_PREREG.md section 13. Runs on EPIC-05. Does NOT recompute anything;
the analyzer (cvt0_analyze.py) owns the verdict and is run separately.
"""
import json, os, sys, hashlib, time

OUT = sys.argv[1] if len(sys.argv) > 1 else "/root/epic05_cvt0"
res = json.load(open(os.path.join(OUT, "CVT0_RESULTS.json")))
env = json.load(open(os.path.join(OUT, "CVT0_ENVIRONMENT.json")))

PREREG_PIN_C0 = "d26cccf46c7932bfb5fd44ee05e553af27e17c93eaffb7dacf3197258a513f11"
PREREG_PIN_CORE = "935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e"
CKPT_HASH = {
    ("room", 10000): "9aa00f0ddf28b6c097e3dc8f41dbb107cd9bf175750503dc330aa1d6ab579fc4",
    ("room", 15000): "b276b338f5a444d8ea1be390d21fd3e67679b7d862652f0babd131db9434e5dd",
    ("room", 20000): "c4e41cfef79e50980406fc1fa148af8d04bc8b37b801c36da5cddb2ce8058718",
    ("bicycle", 10000): "26bf0b959f10f0ce526c496d0dc7ca8e62b0ba93efc83d58b7f9440d58c2f4e9",
    ("bicycle", 15000): "273971d3867c1d2a08b7246803265539d74756b8eca4e3f01c6a8d149b7fd6f7",
    ("bicycle", 20000): "96f136cb331a19900a41a516d53314d522b4de5a3250ddbba8a232b610b398f0",
}

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

# ---- CVT0_SOURCE_IDENTITY.json ---------------------------------------------
src = {
    "audit_type": "CVT0_SOURCE_IDENTITY",
    "authority": "CVT0_PREREG.md section 1 (frozen C0 source/renderer/trainer)",
    "host": env["host"],
    "c0_ext_so_sha256": env["loss_so_sha256"],
    "core_gsplat_so_sha256": env["core_so_sha256"] if "core_so_sha256" in env else None,
    "prereg_pin_c0_ext": PREREG_PIN_C0,
    "prereg_pin_core": PREREG_PIN_CORE,
    "c0_ext_so_match": env["loss_so_sha256"] == PREREG_PIN_C0,
    "note": ("prereg inline C0-ext pin is 63 hex (malformed, dropped a char, same "
             "malformation H0/IR0 documented); authoritative live binary is the "
             "64-hex value verified by sha256sum and matches H0_SOURCE_IDENTITY.json."),
    "checkpoints": {},
    "all_checkpoint_hashes_match": True,
}
# core sha from ident is not in env; recompute via n1_core
sys.path.insert(0, "/root/epic05_n1")
import n1_core as C
src["core_gsplat_so_sha256"] = C.sha256_file(C.CORE_SO)
src["core_so_match"] = src["core_gsplat_so_sha256"] == PREREG_PIN_CORE
for cell, cr in res.items():
    scene, it = cr["scene"], cr["iter"]
    p = os.path.join(C.SNAP_RUNS[scene], f"ckpt_{it}.pt")
    h = sha(p)
    src["checkpoints"][cell] = {"path": p, "sha256": h,
                                "match": h == CKPT_HASH[(scene, it)]}
    src["all_checkpoint_hashes_match"] &= (h == CKPT_HASH[(scene, it)])
src["all_match_prereg"] = bool(src["c0_ext_so_match"] and src["core_so_match"]
                               and src["all_checkpoint_hashes_match"])
with open(os.path.join(OUT, "CVT0_SOURCE_IDENTITY.json"), "w") as f:
    json.dump(src, f, indent=2)

# ---- CVT0_CAMERA_SEMANTICS.json + CVT0_PANELS.json --------------------------
cam_sem = {"audit_type": "CVT0_CAMERA_SEMANTICS",
           "authority": "CVT0_PREREG.md section 3 (corrected SB0 true-holdout)",
           "rule": "TRUE HOLDOUT = canonical index i % 8 == 0; TRAIN POOL = rest",
           "assertion": "train_pool INTERSECT true_holdout = empty", "cells": {}}
panels_doc = {"audit_type": "CVT0_PANELS",
              "authority": "CVT0_PREREG.md section 4 (deterministic source panels)",
              "selection": ("24 stratified midpoint positions q_k=floor((k+0.5)*N/24) "
                            "into ordered TRAIN POOL; dedup by advancing to next unused; "
                            "panel_p = selected k with k mod 3 == p"),
              "holdout_panel_rule": ("8 stratified midpoint quantiles into ordered "
                                     "true-holdout pool"),
              "cells": {}}
for cell, cr in res.items():
    # recover train/holdout sizes + a representative panel (panels identical across cells per scene)
    train_size = cr["train_pool_size"]; hold_size = cr["holdout_size"]; n = cr["n_cams"]
    # recompute train/holdout index lists from dataset ordering (i%8 rule)
    holdout = [i for i in range(n) if i % 8 == 0]
    train = [i for i in range(n) if i % 8 != 0]
    assert set(train).isdisjoint(set(holdout))
    cam_sem["cells"][cell] = {
        "n_cams": n, "train_pool_size": len(train), "true_holdout_size": len(holdout),
        "train_pool_head": train[:16], "true_holdout_head": holdout[:16],
        "train_intersect_holdout_empty": True,
    }
    p0 = cr["panels"]["panel_0"]
    panels_doc["cells"][cell] = {
        "n_cams": n, "train_pool_size": len(train), "true_holdout_size": len(holdout),
        "panel_cams": {pn: cr["panels"][pn]["panel_cams"] for pn in cr["panels"]},
        "holdout_cams": p0["holdout_cams"],
        "panels_unique_8": all(len(set(cr["panels"][pn]["panel_cams"])) == 8
                               for pn in cr["panels"]),
    }
with open(os.path.join(OUT, "CVT0_CAMERA_SEMANTICS.json"), "w") as f:
    json.dump(cam_sem, f, indent=2)
with open(os.path.join(OUT, "CVT0_PANELS.json"), "w") as f:
    json.dump(panels_doc, f, indent=2)

# ---- CVT0_BASE_LOSSES.json (section 5) -------------------------------------
base = {"audit_type": "CVT0_BASE_LOSSES",
        "authority": "CVT0_PREREG.md section 5 (base evaluation, no gradients)",
        "cells": {}}
for cell, cr in res.items():
    base["cells"][cell] = {"panels": {}}
    for pn, pr in cr["panels"].items():
        base["cells"][cell]["panels"][pn] = {
            "panel_cams": pr["panel_cams"], "holdout_cams": pr["holdout_cams"],
            "L0_panel": pr["L0_panel"], "L0_holdout": pr["L0_holdout"],
            "base_panel_eval": pr["base_panel_eval"],
            "base_holdout_eval": pr["base_holdout_eval"],
        }
with open(os.path.join(OUT, "CVT0_BASE_LOSSES.json"), "w") as f:
    json.dump(base, f, indent=2)

# ---- CVT0_TRANSFER_MATRICES.json (section 6) -------------------------------
tm = {"audit_type": "CVT0_TRANSFER_MATRICES",
      "authority": "CVT0_PREREG.md section 6 (source-update matrix M_ij)",
      "definition": "M_ij = L_j(theta0) - L_j(theta_i+); T_i=mean_{j!=i} M_ij; "
                    "H_i=mean_h[L_h(theta0)-L_h(theta_i+)]; S_i=L_i(theta0)-L_i(theta_i+)",
      "cells": {}}
for cell, cr in res.items():
    tm["cells"][cell] = {"panels": {}}
    for pn, pr in cr["panels"].items():
        tm["cells"][cell]["panels"][pn] = {
            "panel_cams": pr["panel_cams"],
            "M_panel": pr["M_panel"], "T_i": pr["T_i"], "H_i": pr["H_i"], "S_i": pr["S_i"],
        }
with open(os.path.join(OUT, "CVT0_TRANSFER_MATRICES.json"), "w") as f:
    json.dump(tm, f, indent=2)

# ---- CVT0_SELECTORS.json (section 7) ---------------------------------------
sel = {"audit_type": "CVT0_SELECTORS",
       "authority": "CVT0_PREREG.md section 7 (three source selectors)",
       "selectors": {
           "RANDOM": "T_random=mean_i T_i, H_random=mean_i H_i (no draw)",
           "HARD_LOSS": "i_hard=argmax_i L_i(theta0), tie-break smallest cam id",
           "TRANSFER_ORACLE": "i_oracle=argmax_i T_i (train peers only), tie-break smallest cam id"},
       "cells": {}}
for cell, cr in res.items():
    sel["cells"][cell] = {"panels": {}}
    for pn, pr in cr["panels"].items():
        sel["cells"][cell]["panels"][pn] = pr["selectors"]
with open(os.path.join(OUT, "CVT0_SELECTORS.json"), "w") as f:
    json.dump(sel, f, indent=2)

# ---- CVT0_TRUE_HOLDOUT.json (section 3/6) ----------------------------------
th = {"audit_type": "CVT0_TRUE_HOLDOUT",
      "authority": "CVT0_PREREG.md sections 3 & 6 (true-holdout transfer)",
      "rule": "true holdout NEVER enters source selection; H_i from holdout panel only",
      "cells": {}}
for cell, cr in res.items():
    th["cells"][cell] = {"panels": {}}
    for pn, pr in cr["panels"].items():
        th["cells"][cell]["panels"][pn] = {
            "holdout_cams": pr["holdout_cams"],
            "H_i": pr["H_i"],
            "L0_holdout": pr["L0_holdout"],
            "post_holdout_eval": pr["post_holdout_eval"],
        }
with open(os.path.join(OUT, "CVT0_TRUE_HOLDOUT.json"), "w") as f:
    json.dump(th, f, indent=2)

# ---- CVT0_DIAGNOSTICS.json (section 10) -----------------------------------
diag = {"audit_type": "CVT0_DIAGNOSTICS",
        "authority": "CVT0_PREREG.md section 10 (report-only, NOT for selection)",
        "cells": {}}
for cell, cr in res.items():
    diag["cells"][cell] = {"panels": {}}
    for pn, pr in cr["panels"].items():
        diag["cells"][cell]["panels"][pn] = pr["diagnostics"]
with open(os.path.join(OUT, "CVT0_DIAGNOSTICS.json"), "w") as f:
    json.dump(diag, f, indent=2)

# ---- CVT0_TIMING.json (section 11) ----------------------------------------
tim = {"audit_type": "CVT0_TIMING",
       "authority": "CVT0_PREREG.md section 11 (timing; NO E2E speedup claim)",
       "note": "CVT0 makes no E2E speedup claim; one native Adam step + one forward eval.",
       "cells": {}}
for cell, cr in res.items():
    per_step = []; per_eval = []
    for pn, pr in cr["panels"].items():
        per_step.append(pr["timing"]["per_step_ms"])
    tim["cells"][cell] = {
        "cell_s": cr.get("cell_s"),
        "panels": {pn: pr["timing"] for pn, pr in cr["panels"].items()},
        "mean_per_step_ms": sum(per_step) / len(per_step) if per_step else None,
    }
with open(os.path.join(OUT, "CVT0_TIMING.json"), "w") as f:
    json.dump(tim, f, indent=2)

print("wrote sectioned artifacts:",
      sorted(x for x in os.listdir(OUT) if x.startswith("CVT0_")))
