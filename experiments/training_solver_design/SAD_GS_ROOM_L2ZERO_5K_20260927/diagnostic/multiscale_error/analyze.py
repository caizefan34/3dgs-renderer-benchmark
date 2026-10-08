#!/usr/bin/env python3
"""Exact block-projection MSE decomposition for frozen room PNG residuals."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path("/root")
SAD = ROOT / "sadgs_room_author_5k_l2zero_20260927/model/test/ours_5000"
FAST = ROOT / "codex_fastgs_audit_20260927/output/room_big_timed_full30k/test/ours_30000"
OUT = ROOT / "sadgs_room_author_5k_l2zero_20260927/diagnostic/multiscale_error/result.json"


def rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0


def hash_bytes(path):
    return hashlib.sha256(path.read_bytes()).digest()


def lowpass_energy(residual, block):
    height, width, channels = residual.shape
    y_starts = np.arange(0, height, block)
    x_starts = np.arange(0, width, block)
    sums = np.add.reduceat(np.add.reduceat(residual, y_starts, axis=0), x_starts, axis=1)
    y_counts = np.minimum(block, height - y_starts)
    x_counts = np.minimum(block, width - x_starts)
    weights = y_counts[:, None] * x_counts[None, :]
    means = sums / weights[:, :, None]
    return float(np.sum(weights[:, :, None] * means * means) / (height * width * channels))


def energies(render, gt):
    error = render - gt
    total = float(np.mean(error * error))
    low4, low16, low64 = (lowpass_energy(error, block) for block in (4, 16, 64))
    bands = {
        "sub4px": total - low4,
        "4to16px": low4 - low16,
        "16to64px": low16 - low64,
        "over64px": low64,
    }
    if abs(sum(bands.values()) - total) > 1e-12 or min(bands.values()) < -1e-12:
        raise RuntimeError("block-projection energy did not telescope")
    return {"mse": total, "bands": bands}


def aggregate(rows):
    labels = ("sub4px", "4to16px", "16to64px", "over64px")
    mean_mse = {method: float(np.mean([row[method]["mse"] for row in rows])) for method in ("sad", "fastgs")}
    mean_band = {
        method: {label: float(np.mean([row[method]["bands"][label] for row in rows])) for label in labels}
        for method in ("sad", "fastgs")
    }
    delta = {label: mean_band["sad"][label] - mean_band["fastgs"][label] for label in labels}
    return {"views": len(rows), "mean_mse": mean_mse, "sad_minus_fastgs_mse": mean_mse["sad"] - mean_mse["fastgs"], "mean_band_mse": mean_band, "sad_minus_fastgs_band_mse": delta}


def main():
    names = [f"{index:05d}.png" for index in range(39)]
    for system in (SAD, FAST):
        for subdir in ("renders", "gt"):
            if sorted(path.name for path in (system / subdir).glob("*.png")) != names:
                raise RuntimeError(f"incomplete 39-view set: {system}/{subdir}")
    records = []
    for name in names:
        sad_gt, fast_gt = SAD / "gt" / name, FAST / "gt" / name
        if hash_bytes(sad_gt) != hash_bytes(fast_gt):
            raise RuntimeError(f"GT byte mismatch for {name}")
        gt = rgb(sad_gt)
        record = {"view": name}
        for method, system in (("sad", SAD), ("fastgs", FAST)):
            image = rgb(system / "renders" / name)
            if image.shape != gt.shape:
                raise RuntimeError(f"shape mismatch for {method}/{name}")
            record[method] = energies(image, gt)
        records.append(record)
    result = {
        "status": "passed",
        "method": "orthogonal projection of RGB residual onto piecewise-constant nonoverlapping blocks with exact edge weights",
        "bands": "sub4px=total-low4; 4to16px=low4-low16; 16to64px=low16-low64; over64px=low64",
        "caveat": "MSE energies are not mean PSNR, and excluding the outlier is descriptive only; all official metrics retain 39 views",
        "gt_png_bytes_identical": True,
        "all39": aggregate(records),
        "descriptive_without_00021": aggregate([row for row in records if row["view"] != "00021.png"]),
        "records": records,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "all39": result["all39"], "descriptive_without_00021": result["descriptive_without_00021"]}, indent=2))


if __name__ == "__main__":
    main()
