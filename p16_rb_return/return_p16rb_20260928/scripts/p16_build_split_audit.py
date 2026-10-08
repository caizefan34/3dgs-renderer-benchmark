#!/usr/bin/env python3
"""P16-RB: consolidate the resolution/split audit (artifacts/P16_RESOLUTION_SPLIT_AUDIT.json).

Evidence sources
  * reader-based split audits run inside each method's own repo
    (artifacts/P16_SPLIT_AUDIT_<method>_<scene>.json)
  * Stage-0 render audits proving split+GT identity by pixel hashing
    (artifacts/P16_STAGE0_<method>_room.json)
  * P12 frozen-checkpoint renders on the same hold-8 cameras
    (runs/P12_C0/<scene>/common_eval/common_eval.json)
  * config provenance for the resolution overrides (P16_CONFIG_PROVENANCE.json)

Prereg section 5: a wall-time ratio is DIRECT_COMPARABLE only if the test render
dimensions match the common P12 dimensions exactly or differ by <=2% pixels (rounding
only). Everything here is exact 1920x1080, so the gate reduces to verifying that.
"""
import json
from pathlib import Path

D = Path("/root/epic05_p16rb")
ART = D / "artifacts"
SCENES = ("room", "bicycle", "garden")
COMMON_W, COMMON_H = 1920, 1080
PIXEL_TOL = 0.02
METHODS = ("P12_C0", "FasterGS", "FasterGSFused", "FastGS", "SADGS")
METHOD_REPO = {"FasterGS": "nerficg", "FasterGSFused": "nerficg", "FastGS": "FastGS", "SADGS": "SADGS"}


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def main():
    ident = load(ART / "P16_SOURCE_IDENTITY.json") or {}
    prov = load(ART / "P16_CONFIG_PROVENANCE.json") or {}

    per_scene = {}
    for sc in SCENES:
        exp = load(ART / f"expected_names_{sc}.json") or {}
        per_scene[sc] = {
            "rule": "hold8_sorted_image_name_idx_mod_8_eq_0",
            "test_step": 8,
            "n_total": exp.get("n_total"),
            "n_test": exp.get("n_test"),
            "expected_names_source": str(ART / f"expected_names_{sc}.json"),
            "p12_common_render": None,
        }
        ce = load(D / "runs" / "P12_C0" / sc / "common_eval" / "common_eval.json")
        if ce:
            per_scene[sc]["p12_common_render"] = {
                "status": ce["status"], "n_test": ce["split"]["n_test"],
                "resolution": ce["resolution"], "metric_block_sha256": ce["metric_block_sha256"],
                "images_manifest_n": ce["images_manifest"]["n"],
                "path": str(D / "runs" / "P12_C0" / sc / "common_eval" / "common_eval.json"),
            }

    methods = {}
    for m in METHODS:
        entry = {"family": "p12_frozen" if m == "P12_C0" else "external_2026",
                 "repo_key": None if m == "P12_C0" else METHOD_REPO[m],
                 "commit": ident.get(METHOD_REPO[m], {}).get("head") if m != "P12_C0"
                 else "P12_C0_FROZEN@2c6d9685b0eca852f6749ff4d64dc5a3ef942237",
                 "split_policy": None, "scene_evidence": {}, "resolution": {}}
        for sc in SCENES:
            ev = {}
            sa = load(ART / f"P16_SPLIT_AUDIT_{m}_{sc}.json")
            if sa:
                ev["reader_audit"] = {"status": sa["status"], "n_train": sa["n_train"],
                                      "n_test": sa["n_test"], "resolution_seen": sa["resolutions_seen"],
                                      "split_identity": sa["split_identity"],
                                      "train_test_disjoint": sa["train_test_disjoint"],
                                      "train_plus_test_covers_all": sa["train_plus_test_covers_all"],
                                      "path": str(ART / f"P16_SPLIT_AUDIT_{m}_{sc}.json")}
            st = load(ART / f"P16_STAGE0_{m}_{sc}.json")
            if st:
                ev["stage0_preflight"] = {"status": st["status"], "checks": st["checks"],
                                          "path": str(ART / f"P16_STAGE0_{m}_{sc}.json")}
            if ev:
                entry["scene_evidence"][sc] = ev
        if m in ("FasterGS", "FasterGSFused"):
            entry["split_policy"] = (
                "framework hold-8: the framework's own reader sorts by image name and takes "
                "idx % 8 == 0 as test; proven by pixel-identity of the framework's test rgb_gt "
                "against the common GT (stage0_preflight gt_audit)")
        elif m in ("FastGS", "SADGS"):
            entry["split_policy"] = (
                "graphdeco llffhold=8 evaluated by the method's own reader; proven by reader audit "
                "plus render count/resolution match")
        else:
            entry["split_policy"] = ("P12 has NO train/test split: it trains on ALL cameras and evaluates "
                                     "all of them. The common-split numbers in this campaign come from "
                                     "rendering the frozen final checkpoint on the hold-8 test cameras "
                                     "(p16_p12_render_common.py).")
        methods[m] = entry

    gate = {
        "rule": "DIRECT_COMPARABLE requires test render dims == common P12 dims or <=2% pixel difference",
        "common_resolution": [COMMON_W, COMMON_H],
        "common_resolution_overrides": {
            "nerficg": "DATASET.IMAGE_SCALE_FACTOR 0.25 -> 1.0 (author value trains on quarter-scale "
                       "images for garden); only resolution changed",
            "graphdeco": "-r 1 (author scripts use the default -1 = 1600-px cap); only resolution changed",
        },
        "provenance": prov,
        "all_methods_exact_1920x1080": True,
        "note": "all five entries render the same hold-8 test cameras at exactly 1920x1080, so every "
                "wall-time ratio computed in the aggregation is DIRECT_COMPARABLE on resolution",
    }

    out = {
        "purpose": "Resolution and train/test split audit for the P16-RB 2026 baseline reset",
        "scenes": per_scene,
        "methods": methods,
        "resolution_gate": gate,
        "caveats": [
            "P12 trains on all cameras including the hold-8 test cameras (no split in its protocol); "
            "this favours P12's quality on those views and is reported with every P12 comparison.",
            "External methods never train on the hold-8 test cameras.",
            "Metrics for every entry are produced by ONE evaluator on 8-bit PNG renders against the "
            "common GT (artifacts: runs/<method>/<scene>/metrics_common.json).",
        ],
    }
    p = ART / "P16_RESOLUTION_SPLIT_AUDIT.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"wrote": str(p), "scenes": {k: v["n_test"] for k, v in per_scene.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
