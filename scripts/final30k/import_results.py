#!/usr/bin/env python3
"""import_results.py — Map FINAL-30K run outputs into the frozen harness artifact tree.

The trainer already writes harness-format files into each run dir; this script
  1. copies them to artifacts/final-30k/<CANDIDATE_ID>/<scene>/,
  2. annotates final_status.json with the scheduler's contamination/timing state,
  3. writes provenance.json from the trainer's results.json.

Source layout (mx):
  /mnt/storage_pool/liaoyuanjun/final30k_runs/<arm>_<scene>/
    results.json  final_status.json  timing.json  quality.json
    training_results.json  training_curve.csv
  /mnt/storage_pool/liaoyuanjun/final30k_runs/scheduler_state.json  (contamination map)

Target layout: artifacts/final-30k/<CANDIDATE_ID>/<scene>/ (frozen schema).
Candidate mapping: arm b1a -> B1A_ACCUTILE, arm c0 -> C0_V3_FINAL30K.

Usage: python import_results.py <staging_dir> <arm> <scene>
       staging_dir is a local copy (or mount) of final30k_runs.
"""
import json
import os
import shutil
import sys

ARM_TO_CANDIDATE = {"b1a": "B1A_ACCUTILE", "c0": "C0_V3_FINAL30K"}
HARNESS_FILES = ["final_status.json", "timing.json", "quality.json",
                 "memory.json", "training_results.json", "training_curve.csv"]


def main():
    if len(sys.argv) < 4:
        print("usage: import_results.py <staging_dir> <arm> <scene>", file=sys.stderr)
        sys.exit(2)
    staging, arm, scene = sys.argv[1], sys.argv[2], sys.argv[3]
    cand = ARM_TO_CANDIDATE.get(arm, arm)
    src = os.path.join(staging, f"{arm}_{scene}")
    if not os.path.isdir(src):
        print(f"ERR: no run dir {src}", file=sys.stderr)
        sys.exit(1)
    results = json.load(open(os.path.join(src, "results.json")))

    # scheduler contamination map
    contam_map = {}
    contam_state = {}
    sched = os.path.join(staging, "scheduler_state.json")
    if os.path.exists(sched):
        contam_state = json.load(open(sched))
        contam_map = contam_state.get("contaminated", {})
    key = f"{arm}_{scene}"

    tgt = os.path.join("artifacts", "final-30k", cand, scene)
    os.makedirs(tgt, exist_ok=True)

    # copy harness files
    for fn in HARNESS_FILES:
        sp = os.path.join(src, fn)
        if os.path.exists(sp):
            shutil.copy(sp, os.path.join(tgt, fn))

    # training_curve.csv — ALWAYS regenerated from results.json eval_rows so
    # every run (old or new trainer version) gets the frozen schema with the
    # lpips column: step,wall_time,psnr,ssim,lpips,l1,n_gaussians,tag
    import csv as _csv
    with open(os.path.join(tgt, "training_curve.csv"), "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["step", "wall_time", "psnr", "ssim", "lpips", "l1", "n_gaussians", "tag"])
        for r in sorted(results.get("eval_rows", []), key=lambda x: x["step"]):
            w.writerow([r["step"], r.get("wall_time_s", ""), r["psnr"], r["ssim"],
                        r.get("lpips", "null"), r.get("l1", ""), r.get("n_gaussians", ""),
                        r["tag"]])

    # annotate final_status.json with scheduler contamination
    fs_path = os.path.join(tgt, "final_status.json")
    fs = json.load(open(fs_path)) if os.path.exists(fs_path) else {"status": "MISSING"}
    fs["contaminated"] = bool(contam_map.get(key, False))
    fs["arm"] = arm
    fs["scene"] = scene
    fs["candidate"] = cand
    if fs.get("contaminated"):
        fs["timing_grade_effective"] = "FUNCTIONAL_ONLY_CONTAMINATED"
    with open(fs_path, "w") as f:
        json.dump(fs, f, indent=2)

    # provenance.json from results.json
    prov = {
        "arm": arm, "scene": scene, "candidate": cand,
        "iterations": results.get("iterations"),
        "seed": results.get("seed"),
        "renderer": results.get("renderer"),
        "binary_identity": results.get("binary_identity"),
        "timing_grade_declared": results.get("timing_grade"),
        "timing_note": results.get("timing_note"),
        "contaminated": fs["contaminated"],
        "contaminated_attempts": contam_state.get("contaminated_attempts", []),
        "torch": results.get("torch"),
        "n_cameras": results.get("n_cameras"),
        "scene_extent": results.get("scene_extent"),
        "source_run_dir": src,
        "trainer_sha256_note": "scripts/final30k/final30k_trainer.py (unified reference_v1 recipe)",
    }
    with open(os.path.join(tgt, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)

    print(f"imported {arm}/{scene} -> {tgt} (contaminated={fs['contaminated']})")


if __name__ == "__main__":
    main()
