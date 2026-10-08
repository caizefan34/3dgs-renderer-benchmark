#!/usr/bin/env python3
"""Local dry-run of the P13-Y selection/aggregate/report logic on synthetic data.

Not shipped to the remote host. Verifies:
  A. select_pairs(): 8 distinct pairs, freeze order, chosen-only endpoints
  B. aggregate: scene/global classes under 4 synthetic scenarios
  C. report: P13Y_REPORT.md renders from fabricated artifacts
"""
import importlib.util
import json
import os
import shutil
import sys

SRC = r"C:\Users\36570\3dgs-renderer-benchmark\p13_transition\remote_src"
ROOT = r"C:\Users\36570\3dgs-renderer-benchmark\p13_transition\_dryrun_p13y"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def scenario(tmp, scene_classes, tag):
    P13Y = os.path.join(tmp, tag)
    ART = os.path.join(P13Y, "artifacts")
    for d in (ART, f"{P13Y}/logs", f"{P13Y}/pair_results"):
        os.makedirs(d, exist_ok=True)
    open(f"{P13Y}/logs/P13Y_CONTAMINATION.log", "w").write(
        "=== UNIT dry_start START 2026-09-27T00:00:00Z ===\nbody\n"
        "=== UNIT dry_start END 2026-09-27T00:00:01Z rc=0 ===\n")
    scenes = ["room", "bicycle", "garden"]
    SCEN = {"PROMISING": (8, 0.03), "GRAY": (5, 0.20), "DROP_TRIGGER": (1, 0.90)}
    for s, cls in zip(scenes, scene_classes):
        njoint, e_delta = SCEN[cls]
        sel = [{"quantile": q, "lag_target": 100 + i, "pair": {
            "camera_id": i, "t0": 15000 + 100 * i, "t1": 15010 + 100 * i,
            "lag": 100 + i}} for i, q in enumerate([0.05, 0.15, 0.30, 0.45,
                                                    0.60, 0.75, 0.90, 0.98])]
        json.dump({"scene": s, "frozen": True, "date": "2026-09-27",
                   "quantile_selection": sel, "sequence_sha256": "ab" * 32},
                  open(f"{ART}/P13Y_PAIR_PLAN_{s}.json", "w"))
        json.dump({"scene": s, "window": [15000, 17000], "n_iterations": 30000,
                   "n_cameras": 311, "n_adjacent_revisits_in_window": len(sel),
                   "n_adjacent_revisits_full": 5000,
                   "lag_quantile_table_in_window": {
                       "min": 99.0, "max": 200.0, "quantiles": {"0.5": 120.0}}},
                  open(f"{ART}/P13Y_REVISIT_DISTRIBUTION_{s}.json", "w"))
        sdir = f"{P13Y}/pair_results/{s}"
        os.makedirs(sdir, exist_ok=True)
        pairs = []
        for i in range(8):
            joint = i < njoint
            rec = {
                "pair_index": i, "pair_spec": {"camera_id": i, "t0": 15000 + 100 * i,
                                               "t1": 15010 + 100 * i, "lag": 100 + i},
                "states": {"t1": {"sh_degree": 3}},
                "Y1": {"compare": {"visible_jaccard": 0.99,
                                   "membership_jaccard": 0.98,
                                   "order_inversion_rate_median_tile": 0.01,
                                   "unchanged_full_order_tile_fraction": 0.9}},
                "Y2": {"h_consistency_max_rel": 0.01,
                       "primary": {"E_delta": e_delta if joint else e_delta,
                                   "E_I": 0.02, "psnr_lin_vs_I1": 40.0,
                                   "L_rel_err": 0.001}},
                "Y3": {"groups": {g: {"cos": 0.999 if cls != "DROP_TRIGGER" else 0.95,
                                      "rel_l2": 0.01} for g in
                                  ("means", "opacity", "scales", "quats", "sh0", "shN")}},
                "verdict": {"Y2_pass": joint, "Y3_pass": joint, "joint_pass": joint,
                            "catastrophic": False},
            }
            fname = f"pair_{i:02d}_q{i:02d}_cam{i}_t{15000 + 100 * i}_{15010 + 100 * i}.json"
            json.dump(rec, open(f"{sdir}/{fname}", "w"))
            pairs.append({"pair_index": i, "file": fname, "verdict": rec["verdict"]})
        json.dump({"scene": s, "pairs": pairs}, open(f"{sdir}/_SUMMARY.json", "w"))
        json.dump({"scene": s, "gate": {"status": "PASS", "all_bitwise": True,
                                       "all_pass": True},
                   "probes": [{"frame_max_abs_diff": 0.0, "loss_rel_err": 0.0}]},
                  open(f"{ART}/P13Y_ADAPTER_PARITY_{s}.json", "w"))
    json.dump({"phase": "P13-Y", "status": "IDENTITY_OK", "abort": False,
               "checks": {"p12_trainer": {"sha256": "67" * 32, "expected_sha256": "67" * 32,
                                          "match": True},
                          "c0_extension_so": {"sha256": "d2" * 32, "expected_sha256": "d2" * 32,
                                              "match": True},
                          "c0_core_so": {"sha256": "93" * 32, "expected_sha256": "93" * 32,
                                         "match": True},
                          "fused_ssim_cuda_so": {"sha256": "dd" * 32,
                                                 "expected_sha256": "dd" * 32, "match": True},
                          "higs_render_source": {"path": "/x/gaussian_inference.py",
                                                 "sha256": "d5" * 32,
                                                 "documented_pin": "512d366b",
                                                 "note": "dead .git link"},
                          "capture_trainer": {"sha256": "ca" * 32, "p12_base_sha256": "67" * 32,
                                              "additive_diff_sha256": "di" * 32}}},
              open(f"{ART}/P13Y_SOURCE_IDENTITY.json", "w"))
    json.dump({"static_evidence": {"lines": {"anchor1": {"line_numbers": [1]}}},
               "per_scene": {s: {"npy": f"/x/{s}.npy", "dtype": "int32",
                                 "shape": [30000], "regeneration": {"matches_bitwise": True},
                                 "every_camera_once_per_cycle": True,
                                 "empirical": {"conclusion": "ok"}} for s in scenes}},
              open(f"{ART}/P13Y_SEQUENCE_INDEXING.json", "w"))
    json.dump({"runs": {"off1": {"dir": "/off1"}}, "criterion": {"pass": True, "rule": "r",
               "checks": {"a": True}}, "camera_sequence_identical_all_three": True,
               "final_params_identical_off1_off2": True,
               "final_params_identical_on_off1": True,
               "noise_floor_off1_vs_off2": {"n_loss_differing": 3},
               "capture_effect_on_vs_off1": {"n_loss_differing": 2},
               "full_replay_vs_p12_original": {"n_iter_rows_compared": 171, "n_loss_differing": 4,
                                               "max_abs_loss_diff": 1e-6,
                                               "first_differing_iter": 14,
                                               "n_gs_equal_all_rows": True,
                                               "camera_sequence_sha256_match": True}},
              open(f"{ART}/P13Y_CAPTURE_NEUTRALITY.json", "w"))
    return P13Y, ART


def main():
    tmp = os.path.join(ROOT, "tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)

    si = load("p13y_sequence_indexing", os.path.join(SRC, "p13y_sequence_indexing.py"))
    seq = si.regenerate(8, 20000)
    assert (seq == si.regenerate(8, 20000)).all()
    pairs = si.adjacent_revisits(seq)
    inw, chosen = si.select_pairs(pairs, window=(15000, 17000))
    keys = [(c["pair"]["camera_id"], c["pair"]["t0"], c["pair"]["t1"])
            for c in chosen if c.get("pair")]
    assert len(chosen) == 8 and len(set(keys)) == 8, keys
    ep = sorted({c["pair"]["t0"] for c in chosen} | {c["pair"]["t1"] for c in chosen})
    assert all(15000 <= t <= 17000 for t in ep), ep
    assert len(inw) >= 8
    print(f"A OK: {len(pairs)} full revisits, {len(inw)} in-window, "
          f"lags={[c['pair']['lag'] for c in chosen]}, endpoints={ep}")

    agg = load("p13y_aggregate", os.path.join(SRC, "p13y_aggregate.py"))
    rep = load("p13y_report", os.path.join(SRC, "p13y_report.py"))
    for cls3, expect in ([["PROMISING"] * 3, "PRIMARY_BREAKTHROUGH_CANDIDATE"],
                         [["PROMISING", "PROMISING", "GRAY"],
                          "INVESTIGATE_SCENE_DEPENDENCE"],
                         [["DROP_TRIGGER", "DROP_TRIGGER", "GRAY"],
                          "DROP_PERSISTENT_LINEARIZATION"],
                         [["GRAY", "GRAY", "GRAY"], "MIXED_NO_DECISION"]):
        tag = expect.lower()
        P13Y, ART = scenario(tmp, cls3, tag)
        agg.P13Y, agg.ART = P13Y, ART
        rep.P13Y, rep.ART = P13Y, ART
        agg.main()
        v = json.load(open(f"{ART}/P13Y_VERDICT.json"))
        assert v["class"] == expect, (expect, v["class"])
        rep.main()
        md = open(f"{P13Y}/P13Y_REPORT.md").read()
        assert expect in md and "## OBSERVED" in md and "## CONCLUSION" in md
        print(f"B/C OK: {expect} -> verdict + report render")

    print("DRYRUN PASS")


if __name__ == "__main__":
    main()
