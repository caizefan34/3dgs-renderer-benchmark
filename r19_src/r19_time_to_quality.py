#!/usr/bin/env python3
"""R19 P4 time-to-quality: for each scene, target = std_ll final PSNR - 0.1 dB.
For each backend, find the earliest eval step whose PSNR >= target AND the wall
time to reach it. NOT_REACHED if never attained. Also report fixed-5K quality
difference. Reads raw/p2_<be>_<scene>/train_results.json eval_rows."""
from __future__ import annotations
import json, os, csv
from pathlib import Path

W = Path(os.environ.get("R19_OUT",
    "/mnt/storage_pool/liaoyuanjun/TRAINABLE_HIGS_FULLTRAIN_R19_MX"))
RAW = W / "raw"
SCENES = ["bicycle", "garden", "room"]
MARGIN_DB = 0.1


def eval_rows(tag, scene):
    p = RAW / f"{tag}_{scene}" / "eval.csv"
    if not p.is_file():
        return []
    rows = []
    with open(p) as f:
        for r in csv.DictReader(f):
            rows.append({"step": int(r["step"]), "psnr": float(r["psnr"]),
                         "wall_time_s": float(r["wall_time_s"]) if r["wall_time_s"] else None})
    return rows


def final_results(tag, scene):
    p = RAW / f"{tag}_{scene}" / "train_results.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def main():
    rows = []
    for scene in SCENES:
        std_final = final_results("p2_std_ll", scene)
        higs_final = final_results("p2_higs", scene)
        if not std_final or not higs_final:
            continue
        std_psnr_final = std_final["final_eval"]["psnr"]
        higs_psnr_final = higs_final["final_eval"]["psnr"]
        target = std_psnr_final - MARGIN_DB
        std_rows = eval_rows("p2_std_ll", scene)
        higs_rows = eval_rows("p2_higs", scene)

        def time_to(rows, tgt):
            for r in rows:
                if r["psnr"] >= tgt and r["wall_time_s"] is not None:
                    return r["step"], r["wall_time_s"], r["psnr"]
            return None

        std_ttq = time_to(std_rows, target)
        higs_ttq = time_to(higs_rows, target)
        rows.append({
            "scene": scene,
            "std_final_psnr": round(std_psnr_final, 4),
            "higs_final_psnr": round(higs_psnr_final, 4),
            "target_psnr": round(target, 4),
            "std_time_to_target_s": std_ttq[1] if std_ttq else "NOT_REACHED",
            "std_step_to_target": std_ttq[0] if std_ttq else "NOT_REACHED",
            "higs_time_to_target_s": higs_ttq[1] if higs_ttq else "NOT_REACHED",
            "higs_step_to_target": higs_ttq[0] if higs_ttq else "NOT_REACHED",
            "fixed_5k_psnr_diff_higs_minus_std": round(higs_psnr_final - std_psnr_final, 4),
            "fixed_5k_time_diff_s": round(higs_final["total_time_s"] - std_final["total_time_s"], 2),
        })
    with open(W / "R19_TIME_TO_QUALITY.csv", "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
