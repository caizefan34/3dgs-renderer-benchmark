#!/usr/bin/env python3
"""Read-only per-image RGB-bias oracle on frozen room render PNGs."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path("/root")
SAD = ROOT / "sadgs_room_author_5k_l2zero_20260927/model/test/ours_5000"
FAST = ROOT / "codex_fastgs_audit_20260927/output/room_big_timed_full30k/test/ours_30000"
OUT = ROOT / "sadgs_room_author_5k_l2zero_20260927/diagnostic/color_bias_oracle/result.json"


def rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0


def psnr(mse):
    return -10.0 * math.log10(mse)


def sha(path):
    return hashlib.sha256(path.read_bytes()).digest()


def main():
    names = sorted(path.name for path in (SAD / "renders").glob("*.png"))
    expected = [f"{i:05d}.png" for i in range(39)]
    if names != expected:
        raise RuntimeError("SAD render names differ from frozen 39-view set")
    for system in (SAD, FAST):
        for subdir in ("renders", "gt"):
            actual = sorted(path.name for path in (system / subdir).glob("*.png"))
            if actual != expected:
                raise RuntimeError(f"{system}/{subdir} has wrong names")

    records = []
    for name in names:
        gt_paths = [system / "gt" / name for system in (SAD, FAST)]
        if sha(gt_paths[0]) != sha(gt_paths[1]):
            raise RuntimeError(f"GT PNG mismatch for {name}")
        gt = rgb(gt_paths[0])
        row = {"view": name}
        for label, system in (("sad", SAD), ("fastgs", FAST)):
            render = rgb(system / "renders" / name)
            if render.shape != gt.shape:
                raise RuntimeError(f"render/GT shape mismatch for {label}/{name}")
            error = gt - render
            baseline_mse = float(np.mean(error * error))
            bias = np.mean(error, axis=(0, 1))
            oracle = np.clip(render + bias, 0.0, 1.0)
            oracle_mse = float(np.mean((gt - oracle) ** 2))
            row[label] = {
                "mse": baseline_mse,
                "psnr": psnr(baseline_mse),
                "oracle_rgb_addition": [float(value) for value in bias],
                "oracle_mse": oracle_mse,
                "oracle_psnr": psnr(oracle_mse),
                "oracle_psnr_gain": psnr(oracle_mse) - psnr(baseline_mse),
                "oracle_mse_fraction_removed": 1.0 - oracle_mse / baseline_mse,
            }
        records.append(row)

    result = {
        "status": "passed",
        "scope": "fixed frozen 39 room held-out PNGs; separate per-image/per-method GT-fitted additive RGB bias oracle",
        "warning": "oracle uses test GT and is not a deployable correction or official score",
        "sad_path": str(SAD),
        "fastgs_path": str(FAST),
        "gt_png_bytes_identical": True,
        "views": len(records),
        "means": {
            key: {
                metric: float(np.mean([row[key][metric] for row in records]))
                for metric in ("psnr", "oracle_psnr", "oracle_psnr_gain", "oracle_mse_fraction_removed")
            }
            for key in ("sad", "fastgs")
        },
        "records": records,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "views": result["views"], "means": result["means"]}, indent=2))


if __name__ == "__main__":
    main()
