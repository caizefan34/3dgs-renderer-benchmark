#!/usr/bin/env python3
"""Forensic audit of the OLD corrupted garden point layout.

This conversion is NOT a dataset repair: the old 1.8M-point gray cloud and
stripped images.bin were synthetic/corrupt. Valid P16 baselines use the
official 138,766-point sparse reconstruction documented in
P16_GARDEN_POINT_ORDER_AUDIT_20260928.md. This script never changes its input.
"""

import argparse
import hashlib
import json
import math
import shutil
import struct
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_exact(f, n):
    value = f.read(n)
    if len(value) != n:
        raise ValueError("truncated points3D.bin")
    return value


def convert(source, out_dir):
    target = None
    ply = None
    if out_dir:
        sparse = out_dir / "sparse" / "0"
        sparse.mkdir(parents=True, exist_ok=False)
        for name in ("cameras.bin", "images.bin"):
            shutil.copy2(source.parent / name, sparse / name)
        target = (sparse / "points3D.bin").open("wb")
        ply = (sparse / "points3D.ply").open("wb")
    try:
        with source.open("rb") as f:
            count = struct.unpack("<Q", read_exact(f, 8))[0]
            if target:
                target.write(struct.pack("<Q", count))
                header = ("ply\nformat binary_little_endian 1.0\n"
                          f"element vertex {count}\n"
                          "property float x\nproperty float y\nproperty float z\n"
                          "property float nx\nproperty float ny\nproperty float nz\n"
                          "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                          "end_header\n")
                ply.write(header.encode("ascii"))
            prev_id = 0
            nonmonotone = 0
            nonfinite = 0
            nonzero_tracks = 0
            xyz_min = [math.inf] * 3
            xyz_max = [-math.inf] * 3
            first = None
            for index in range(count):
                xyz_bytes = read_exact(f, 24)
                id_bytes = read_exact(f, 8)
                rgb = read_exact(f, 3)
                err_bytes = read_exact(f, 8)
                track_len_bytes = read_exact(f, 8)
                point_id = struct.unpack("<Q", id_bytes)[0]
                xyz = struct.unpack("<3d", xyz_bytes)
                error = struct.unpack("<d", err_bytes)[0]
                track_len = struct.unpack("<Q", track_len_bytes)[0]
                if track_len > 100000:
                    raise ValueError(f"implausible track length {track_len} at {index}")
                track = read_exact(f, 8 * track_len)
                if index and point_id <= prev_id:
                    nonmonotone += 1
                prev_id = point_id
                if not all(map(math.isfinite, (*xyz, error))):
                    nonfinite += 1
                nonzero_tracks += track_len > 0
                for axis in range(3):
                    xyz_min[axis] = min(xyz_min[axis], xyz[axis])
                    xyz_max[axis] = max(xyz_max[axis], xyz[axis])
                if first is None:
                    first = {"point_id": point_id, "xyz": xyz, "rgb": list(rgb),
                             "error": error, "track_len": track_len}
                if target:
                    target.write(id_bytes + xyz_bytes + rgb + err_bytes + track_len_bytes + track)
                    ply.write(struct.pack("<6f3B", *xyz, 0.0, 0.0, 0.0, *rgb))
            if f.read(1):
                raise ValueError("trailing data in points3D.bin")
    finally:
        if target:
            target.close()
            ply.close()
    result = {"source": str(source), "source_sha256": sha256(source),
              "point_count": count, "first": first, "last_point_id": prev_id,
              "nonmonotone_ids": nonmonotone, "nonfinite_records": nonfinite,
              "records_with_tracks": nonzero_tracks, "xyz_min": xyz_min, "xyz_max": xyz_max}
    if out_dir:
        result["staged_bin_sha256"] = sha256(out_dir / "sparse/0/points3D.bin")
        result["staged_ply_sha256"] = sha256(out_dir / "sparse/0/points3D.ply")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="original nonstandard points3D.bin")
    parser.add_argument("--out-dir", type=Path, help="new isolated scene directory")
    args = parser.parse_args()
    convert(args.source, args.out_dir)
