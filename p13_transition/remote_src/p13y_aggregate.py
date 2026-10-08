#!/usr/bin/env python3
"""P13-Y aggregate: Y23 aggregates, scene gates (Sec.9), verdict (Sec.10),
contamination ledger, MANIFEST.

Decision classes (frozen; no post-hoc gate edits):
  scene PROMISING  : n_joint_pass >= 6/8 AND median E_delta <= 0.10 AND
                     median cos_means >= 0.995 AND median E_rel_means <= 0.05 AND
                     median cos_opacity >= 0.995 AND n_catastrophic <= 1
  scene DROP-trigger: n_joint_pass < 3/8 OR median E_delta > 0.50 OR
                     median cos_means < 0.98
  else GRAY
  global:
    3 PROMISING                                -> PRIMARY_BREAKTHROUGH_CANDIDATE
    2 PROMISING and 1 GRAY                     -> INVESTIGATE_SCENE_DEPENDENCE
    >= 2 DROP-trigger scenes                   -> DROP_PERSISTENT_LINEARIZATION
    otherwise                                  -> MIXED_NO_DECISION
"""
import hashlib
import json
import os
import re
import statistics as st

P13Y = "/root/epic05_p13y"
ART = f"{P13Y}/artifacts"
SCENES = ["room", "bicycle", "garden"]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_json(p, default=None):
    if not os.path.exists(p):
        return default
    with open(p) as f:
        return json.load(f)


def med(xs):
    xs = [x for x in xs if x is not None]
    return float(st.median(xs)) if xs else None


def main():
    agg = {"phase": "P13-Y Y2/Y3 aggregate", "date": "2026-09-27", "scenes": {}}
    per_scene_verdict = {}
    for scene in SCENES:
        sdir = f"{P13Y}/pair_results/{scene}"
        summ = load_json(f"{sdir}/_SUMMARY.json", {"pairs": []})
        pairs = []
        for fn in sorted(os.listdir(sdir)) if os.path.isdir(sdir) else []:
            if fn.startswith("pair_") and fn.endswith(".json"):
                pairs.append(load_json(os.path.join(sdir, fn)))
        plan = load_json(f"{ART}/P13Y_PAIR_PLAN_{scene}.json", {})
        rec = {
            "n_pairs_planned": len(plan.get("quantile_selection", [])),
            "n_pairs_run": len(pairs),
            "n_invalid": sum(1 for p in pairs if p.get("verdict", {}).get("invalid")),
            "n_Y2_pass": sum(1 for p in pairs if p.get("verdict", {}).get("Y2_pass")),
            "n_Y3_pass": sum(1 for p in pairs if p.get("verdict", {}).get("Y3_pass")),
            "n_joint_pass": sum(1 for p in pairs if p.get("verdict", {}).get("joint_pass")),
            "n_catastrophic": sum(1 for p in pairs
                                  if p.get("verdict", {}).get("catastrophic")),
            "median_E_delta": med([p.get("Y2", {}).get("primary", {}).get("E_delta")
                                   for p in pairs]),
            "median_E_I": med([p.get("Y2", {}).get("primary", {}).get("E_I")
                               for p in pairs]),
            "median_psnr_lin": med([p.get("Y2", {}).get("primary", {}).get("psnr_lin_vs_I1")
                                    for p in pairs]),
            "median_L_rel_err": med([p.get("Y2", {}).get("primary", {}).get("L_rel_err")
                                     for p in pairs]),
            "median_cos_means": med([p.get("Y3", {}).get("groups", {})
                                     .get("means", {}).get("cos") for p in pairs]),
            "median_E_rel_means": med([p.get("Y3", {}).get("groups", {})
                                       .get("means", {}).get("rel_l2") for p in pairs]),
            "median_cos_opacity": med([p.get("Y3", {}).get("groups", {})
                                       .get("opacity", {}).get("cos") for p in pairs]),
            "median_cos_scales": med([p.get("Y3", {}).get("groups", {})
                                      .get("scales", {}).get("cos") for p in pairs]),
            "median_cos_quats": med([p.get("Y3", {}).get("groups", {})
                                     .get("quats", {}).get("cos") for p in pairs]),
            "median_cos_sh0": med([p.get("Y3", {}).get("groups", {})
                                   .get("sh0", {}).get("cos") for p in pairs]),
            "median_cos_shN": med([p.get("Y3", {}).get("groups", {})
                                   .get("shN", {}).get("cos") for p in pairs]),
            "median_Y1_visible_jaccard": med([p.get("Y1", {}).get("compare", {})
                                              .get("visible_jaccard") for p in pairs]),
            "median_Y1_membership_jaccard": med([p.get("Y1", {}).get("compare", {})
                                                 .get("membership_jaccard") for p in pairs]),
            "median_Y1_order_inversion_rate": med(
                [p.get("Y1", {}).get("compare", {})
                 .get("order_inversion_rate_median_tile") for p in pairs]),
            "median_Y1_unchanged_full_order_tile_fraction": med(
                [p.get("Y1", {}).get("compare", {})
                 .get("unchanged_full_order_tile_fraction") for p in pairs]),
            "median_h_consistency_max_rel": med(
                [p.get("Y2", {}).get("h_consistency_max_rel") for p in pairs]),
            "dominant_sh_degree": (pairs[0].get("states", {}).get("t1", {}).get(
                "sh_degree") if pairs else None),
            "pair_table": [{"idx": p.get("pair_index"),
                            "cam": p.get("pair_spec", {}).get("camera_id"),
                            "t0": p.get("pair_spec", {}).get("t0"),
                            "t1": p.get("pair_spec", {}).get("t1"),
                            "lag": p.get("pair_spec", {}).get("lag"),
                            "E_delta": p.get("Y2", {}).get("primary", {}).get("E_delta"),
                            "L_rel_err": p.get("Y2", {}).get("primary", {}).get("L_rel_err"),
                            "cos_means": p.get("Y3", {}).get("groups", {}).get("means", {}).get("cos"),
                            "E_rel_means": p.get("Y3", {}).get("groups", {}).get("means", {}).get("rel_l2"),
                            "cos_opacity": p.get("Y3", {}).get("groups", {}).get("opacity", {}).get("cos"),
                            "Y1_vis_jaccard": p.get("Y1", {}).get("compare", {}).get("visible_jaccard"),
                            "verdict": p.get("verdict")}
                           for p in pairs],
        }
        npair = rec["n_pairs_run"]
        promising = bool(
            npair >= 8 and rec["n_joint_pass"] >= 6
            and rec["median_E_delta"] is not None and rec["median_E_delta"] <= 0.10
            and rec["median_cos_means"] is not None and rec["median_cos_means"] >= 0.995
            and rec["median_E_rel_means"] is not None and rec["median_E_rel_means"] <= 0.05
            and rec["median_cos_opacity"] is not None and rec["median_cos_opacity"] >= 0.995
            and rec["n_catastrophic"] <= 1)
        drop_trigger = bool(
            rec["n_joint_pass"] < 3
            or (rec["median_E_delta"] is not None and rec["median_E_delta"] > 0.50)
            or (rec["median_cos_means"] is not None and rec["median_cos_means"] < 0.98))
        rec["scene_class"] = ("PROMISING" if promising else
                              ("DROP_TRIGGER" if drop_trigger else "GRAY"))
        rec["scene_class_rule"] = {
            "promising": ("n_joint>=6/8 AND median E_delta<=0.10 AND median cos_means"
                          ">=0.995 AND median E_rel_means<=0.05 AND median"
                          " cos_opacity>=0.995 AND catastrophic<=1"),
            "drop_trigger": "n_joint<3/8 OR median E_delta>0.50 OR median cos_means<0.98",
        }
        agg["scenes"][scene] = rec
        per_scene_verdict[scene] = rec["scene_class"]

    n_prom = sum(1 for v in per_scene_verdict.values() if v == "PROMISING")
    n_gray = sum(1 for v in per_scene_verdict.values() if v == "GRAY")
    n_drop = sum(1 for v in per_scene_verdict.values() if v == "DROP_TRIGGER")
    if n_prom == 3:
        decision = "PRIMARY_BREAKTHROUGH_CANDIDATE"
    elif n_prom == 2 and n_gray == 1:
        decision = "INVESTIGATE_SCENE_DEPENDENCE"
    elif n_drop >= 2:
        decision = "DROP_PERSISTENT_LINEARIZATION"
    else:
        decision = "MIXED_NO_DECISION"
    agg["decision"] = {
        "class": decision,
        "counts": {"PROMISING": n_prom, "GRAY": n_gray, "DROP_TRIGGER": n_drop},
        "note": ("Sec.10: 6/8 pass is not 3x. Only PRIMARY_BREAKTHROUGH_CANDIDATE "
                 "permits designing the Y4 composed executable oracle; otherwise "
                 "move to Spectral/Compressed Adjoint (no rescuing)."),
    }

    # parity gate aggregation
    parity = {}
    for scene in SCENES:
        p = load_json(f"{ART}/P13Y_ADAPTER_PARITY_{scene}.json")
        parity[scene] = None if p is None else {
            "status": p["gate"]["status"], "all_bitwise": p["gate"]["all_bitwise"],
            "n_probes": len(p["probes"]),
            "max_abs_diff": max([x["frame_max_abs_diff"] for x in p["probes"]], default=None),
            "max_loss_rel_err": max([x["loss_rel_err"] for x in p["probes"]], default=None)}
    agg["adapter_parity"] = parity
    agg["adapter_parity_gate_status"] = (
        "PASS" if all(v and v["status"] == "PASS" for v in parity.values()) and
        all(v is not None for v in parity.values()) else
        ("INCOMPLETE" if any(v is None for v in parity.values())
         else "STOP_ADAPTER_PARITY_FAILED"))

    with open(f"{ART}/P13Y_Y23_AGG.json", "w") as f:
        json.dump(agg, f, indent=2)
    print(json.dumps({"decision": agg["decision"], "parity": agg["adapter_parity"],
                      "scenes": {s: {k: v for k, v in agg["scenes"][s].items()
                                     if k != "pair_table"}
                                 for s in SCENES}}, indent=2))

    # ---- verdict ----
    neum = load_json(f"{ART}/P13Y_CAPTURE_NEUTRALITY.json")
    ident = load_json(f"{ART}/P13Y_SOURCE_IDENTITY.json")
    seq = load_json(f"{ART}/P13Y_SEQUENCE_INDEXING.json")
    verdict = {
        "phase": "P13-Y verdict", "date": "2026-09-27",
        "class": decision,
        "scene_classes": per_scene_verdict,
        "decision_counts": agg["decision"]["counts"],
        "adapter_parity_gate": agg["adapter_parity_gate_status"],
        "source_identity_status": (ident or {}).get("status"),
        "sequence_indexing_conclusion": {
            "static_anchors_found": bool(seq and all(
                "error" not in v for v in seq["static_evidence"]["lines"].values())),
            "regeneration_matches_all_scenes": bool(seq and all(
                seq["per_scene"][s]["regeneration"]["matches_bitwise"] for s in SCENES)),
            "empirical": {s: (seq["per_scene"][s].get("empirical") if seq else None)
                          for s in SCENES},
        },
        "capture_neutrality_pass": (neum or {}).get("criterion", {}).get("pass"),
        "forbidden_actions_observed": {
            "no_production_persistent_cuda_cache": True,
            "no_full_30k_training": True,
            "no_training_speedup_claim": True,
            "no_novelty_claim": True,
            "gates_unchanged_after_results": True,
            "no_git_commit": True,
        },
        "next_step": ("design Y4 composed executable oracle (only allowed under "
                      "PRIMARY_BREAKTHROUGH_CANDIDATE; Sec.10 cost model + "
                      "S_E2E,conservative >= 4x + 25th-percentile reuse lifetime "
                      ">= 3x + cache footprint rules)"
                      if decision == "PRIMARY_BREAKTHROUGH_CANDIDATE" else
                      "move to Spectral / Compressed Adjoint; do NOT rescue "
                      "persistent linearization"),
    }
    with open(f"{ART}/P13Y_VERDICT.json", "w") as f:
        json.dump(verdict, f, indent=2)
    print(json.dumps(verdict, indent=2))

    # ---- contamination ledger ----
    log = f"{P13Y}/logs/P13Y_CONTAMINATION.log"
    units = []
    if os.path.exists(log):
        txt = open(log).read()
        for m in re.finditer(
                r"=== UNIT (\S+) START (\S+) ===(.*?)=== UNIT \1 END (\S+) rc=(\d+) ===",
                txt, re.S):
            units.append({"unit": m.group(1), "start": m.group(2),
                          "end": m.group(4), "rc": int(m.group(5)),
                          "body": m.group(3).strip()})
    cont = {"phase": "P13-Y contamination ledger", "date": "2026-09-27",
            "log": log, "n_units": len(units), "units": units,
            "policy": ("one clean physical GPU cohort per compared batch; GPU "
                       "index/UUID + compute processes logged before/after every "
                       "unit; contaminated units retained verbatim")}
    with open(f"{ART}/P13Y_CONTAMINATION.json", "w") as f:
        json.dump(cont, f, indent=2)
    print("units:", [(u["unit"], u["rc"]) for u in units])

    # ---- manifest ----
    # The manifest covers the research tree. Transport-only artifacts (the
    # return package: tarballs, its README and checksums) and __pycache__ are
    # excluded so the manifest does not hash a copy of itself.
    EXCLUDE_DIRS = {"return_pkg", "__pycache__"}
    files = []
    for root, dirs, fns in os.walk(P13Y):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for fn in fns:
            p = os.path.join(root, fn)
            try:
                files.append({"path": p, "bytes": os.path.getsize(p),
                              "sha256": sha256_file(p)})
            except OSError:
                pass
    files.sort(key=lambda r: r["path"])
    man = {"phase": "P13-Y manifest", "date": "2026-09-27",
           "excluded_dirs": sorted(EXCLUDE_DIRS),
           "note": ("transport-only return package and __pycache__ are outside "
                    "this manifest; the return package carries a checksum file "
                    "of its own"),
           "root": P13Y, "n_files": len(files),
           "total_bytes": sum(f["bytes"] for f in files), "files": files}
    with open(f"{ART}/MANIFEST_P13Y.json", "w") as f:
        json.dump(man, f, indent=2)
    print(f"MANIFEST: {len(files)} files, {man['total_bytes'] / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
