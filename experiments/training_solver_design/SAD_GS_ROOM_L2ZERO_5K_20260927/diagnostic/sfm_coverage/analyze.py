#!/usr/bin/env python3
"""Read-only COLMAP track coverage on the frozen eight room training views."""
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path("/root")
MODEL = ROOT / "3dgs-renderer-benchmark/data/datasets/mipnerf360/room/sparse/0"
LOADER = ROOT / "codex_fastgs_audit_20260927/scene/colmap_loader.py"
OUT = ROOT / "sadgs_room_author_5k_l2zero_20260927/diagnostic/sfm_coverage/result.json"
NAMES = ("DSCF4668.JPG", "DSCF4712.JPG", "DSCF4756.JPG", "DSCF4800.JPG", "DSCF4845.JPG", "DSCF4889.JPG", "DSCF4933.JPG", "DSCF4977.JPG")
WIDTH, HEIGHT = 1600, 1066
BLOCK = 64


def main():
    spec = importlib.util.spec_from_file_location("colmap_loader", LOADER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    images = {image.name: image for image in module.read_extrinsics_binary(str(MODEL / "images.bin")).values()}
    cameras = module.read_intrinsics_binary(str(MODEL / "cameras.bin"))
    records = []
    n_y, n_x = (HEIGHT + BLOCK - 1) // BLOCK, (WIDTH + BLOCK - 1) // BLOCK
    for name in NAMES:
        image = images[name]
        camera = cameras[image.camera_id]
        valid = image.point3D_ids >= 0
        xy = image.xys[valid]
        x = xy[:, 0] * WIDTH / camera.width
        y = xy[:, 1] * HEIGHT / camera.height
        inside = (x >= 0) & (x < WIDTH) & (y >= 0) & (y < HEIGHT)
        x, y = x[inside], y[inside]
        gx = np.floor(x / BLOCK).astype(np.int32)
        gy = np.floor(y / BLOCK).astype(np.int32)
        counts = np.bincount(gy * n_x + gx, minlength=n_x * n_y).reshape(n_y, n_x)
        records.append({
            "name": name,
            "camera_original_size": [camera.width, camera.height],
            "valid_colmap_tracks_in_frame": int(len(x)),
            "blocks_64px_total": int(counts.size),
            "blocks_64px_with_at_least_1_track": int((counts >= 1).sum()),
            "blocks_64px_with_at_least_3_tracks": int((counts >= 3).sum()),
            "blocks_64px_with_at_least_10_tracks": int((counts >= 10).sum()),
            "max_tracks_per_64px_block": int(counts.max()),
        })
    result = {
        "status": "passed",
        "scope": "the same eight preselected training cameras as train_views diagnostic; COLMAP track coverage only",
        "note": "tracks are sparse and derived from input images; occupancy is not independent dense depth truth or accepted-work coverage",
        "resolution": [WIDTH, HEIGHT],
        "block": BLOCK,
        "records": records,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
