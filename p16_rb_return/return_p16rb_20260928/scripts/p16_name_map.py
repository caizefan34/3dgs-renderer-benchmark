#!/usr/bin/env python3
"""P16-RB: map index-named framework test renders to the common test-image names.

Both lineages write test renders as 00000.png (paired by index with the GT copy in
rgb_gt/ or gt/). The common split is defined by image NAMES (hold-8 on sorted names), so
this builds render_file -> common_name by hashing GT pixels and matching them against the
common GT images. The map is asserted bijective; it is what lets the ONE common metric
evaluator pair every render with the correct GT regardless of the framework's naming.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from struct import unpack

import numpy as np
from PIL import Image

COMMON = Path("/root/epic05_p16rb/common_data/mipnerf360")
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
    ap.add_argument("--method", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--rgb-dir", required=True)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rgb_dir, gt_dir = Path(args.rgb_dir), Path(args.gt_dir)
    exp_names = hold8_names(args.scene)
    exp = {nm: px_hash(COMMON / args.scene / "images" / (Path(nm).stem + ".png")) for nm in exp_names}
    exp_by_hash = {}
    for nm, h in exp.items():
        exp_by_hash.setdefault(h, []).append(nm)

    rgb_files = sorted(rgb_dir.glob("*.png"))
    gt_files = sorted(gt_dir.glob("*.png"))
    rgb_stems = [p.stem for p in rgb_files]
    gt_stems = [p.stem for p in gt_files]

    problems = []
    if len(rgb_files) != len(exp_names):
        problems.append(f"render count {len(rgb_files)} != expected {len(exp_names)}")
    if rgb_stems != gt_stems:
        problems.append("render/GT stem pairing mismatch (index pairing impossible)")

    name_of = {}
    unmatched = []
    for p in gt_files:
        h = px_hash(p)
        if h in exp_by_hash:
            name_of[p.stem] = exp_by_hash[h][0]
        else:
            unmatched.append(p.name)

    matched_names = set(name_of.values())
    if len(name_of) != len(gt_files) or len(matched_names) != len(exp_names):
        problems.append(f"gt pixel match not bijective: matched_files={len(name_of)} "
                        f"unique_names={len(matched_names)} expected={len(exp_names)}")
    if unmatched:
        problems.append(f"{len(unmatched)} GT files without a unique expected match")

    # values are extension-less stems, the convention used by the common evaluator
    name_map = {f"{stem}.png": Path(name_of[stem]).stem for stem in gt_stems if stem in name_of}
    out = {
        "method": args.method, "scene": args.scene,
        "rgb_dir": str(rgb_dir), "gt_dir": str(gt_dir),
        "rule": "gt_pixel_sha256_match_against_hold8_common_images",
        "n_renders": len(rgb_files), "n_expected": len(exp_names),
        "gt_pixel_sha256": {p.name: px_hash(p) for p in gt_files},
        "expected_gt_sha256": exp,
        "name_map": name_map,
        "problems": problems,
        "status": "PASS" if not problems else "FAIL",
    }
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"method": args.method, "scene": args.scene, "n_renders": len(rgb_files),
                      "n_mapped": len(name_map), "problems": problems, "status": out["status"]},
                     indent=2))
    return 0 if out["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
