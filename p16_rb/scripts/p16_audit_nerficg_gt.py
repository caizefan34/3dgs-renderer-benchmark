#!/usr/bin/env python3
"""P16-RB: prove the NeRFICG test split and GT identity by pixel matching.

The framework renders the test set as <outdir>/test_<iter>/{rgb, rgb_gt}/*.png with
index-based file names, so the split cannot be verified from file names alone.
Instead: hash the pixels of every GT image the framework rendered and match them
against the pixel hashes of the expected hold-8 test images from the common dataset.
A full, unique match proves (a) split identity, (b) GT identity, (c) no resampling drift.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from struct import unpack

import numpy as np
from PIL import Image

D = Path("/root/epic05_p16rb")
COMMON = D / "common_data" / "mipnerf360"
TEST_STEP = 8


def hold8_names(scene: str):
    names = []
    with open(COMMON / scene / "sparse" / "0" / "images.bin", "rb") as f:
        n = unpack("<Q", f.read(8))[0]
        for _ in range(n):
            f.read(4 + 32 + 24 + 4)
            nm = b""
            while True:
                ch = f.read(1)
                if ch == b"\x00":
                    break
                nm += ch
            names.append(nm.decode())
            npts = unpack("<Q", f.read(8))[0]
            f.read(24 * npts)
    order = sorted(range(len(names)), key=lambda i: names[i])
    return [names[i] for pos, i in enumerate(order) if pos % TEST_STEP == 0]


def px_hash(p: Path) -> str:
    with Image.open(p) as im:
        a = np.asarray(im.convert("RGB"))
    return hashlib.sha256(a.tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--test-dir", default=None)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    if args.test_dir:
        tdir = outdir / args.test_dir
    else:
        cands = sorted(outdir.glob("test_*"), key=lambda p: p.stat().st_mtime)
        if not cands:
            raise SystemExit(f"no test_* directory under {outdir}")
        tdir = cands[-1]

    exp_names = hold8_names(args.scene)
    exp = {nm: px_hash(COMMON / args.scene / "images" / (Path(nm).stem + ".png")) for nm in exp_names}

    gt_files = sorted((tdir / "rgb_gt").glob("*.png"))
    rgb_files = sorted((tdir / "rgb").glob("*.png"))
    gt_hashes = {p.name: px_hash(p) for p in gt_files}
    rgb_res = None
    if rgb_files:
        with Image.open(rgb_files[0]) as im:
            rgb_res = [im.size[0], im.size[1]]

    # match framework GT pixels to expected test images
    exp_by_hash = {}
    for nm, h in exp.items():
        exp_by_hash.setdefault(h, []).append(nm)
    matched = {}
    unmatched = []
    for fname, h in gt_hashes.items():
        if h in exp_by_hash:
            matched[fname] = exp_by_hash[h][0]
        else:
            unmatched.append(fname)
    matched_names = set(matched.values())

    report = {
        "outdir": str(outdir), "test_dir": tdir.name, "scene": args.scene,
        "n_expected_test": len(exp_names),
        "n_framework_gt": len(gt_files),
        "n_framework_rgb": len(rgb_files),
        "n_gt_pixel_matched": len(matched),
        "n_gt_unique_expected_matched": len(matched_names),
        "missing_expected": sorted(set(exp_names) - matched_names)[:5],
        "unmatched_gt_files": unmatched[:5],
        "split_identity_by_pixels": len(matched_names) == len(exp_names) == len(gt_files),
        "render_resolution": rgb_res,
        "resolution_ok": rgb_res == [1920, 1080],
        "gt_resolution_ok": all(Image.open(p).size == (1920, 1080) for p in gt_files[:3]) if gt_files else False,
    }
    report["status"] = "PASS" if (report["split_identity_by_pixels"] and report["resolution_ok"]) else "FAIL"
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
