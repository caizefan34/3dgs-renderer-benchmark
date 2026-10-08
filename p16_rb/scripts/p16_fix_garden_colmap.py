#!/usr/bin/env python3
"""P16-RB data fix: rename duplicate point3D ids in garden points3D.bin.

The P12-era source model carries duplicate point3D ids; strict pycolmap
('Check failed: points3D_.emplace(point3D_id, ...).second') refuses to load it.
This rewrite keeps every point, in order, with identical geometry/colour/tracks
and only renames duplicate occurrences to fresh unique ids (max_id+1, +2, ...).
Tracks referencing the shared id stay valid (they resolve to the first
occurrence); the renamed copies are simply unreferenced by tracks.

Usage: python3 p16_fix_garden_colmap.py <sparse0_dir> [--apply]
Without --apply it only reports duplicates.
"""
import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_points3D(path: Path):
    items = []
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            pid = struct.unpack("<Q", f.read(8))[0]
            rest = f.read(24 + 3 + 8)  # xyz (3 double), rgb (3 uint8), error (double)
            tl = struct.unpack("<Q", f.read(8))[0]
            track = f.read(8 * tl)  # per element: image_id int32, point2D_idx int32
            items.append((pid, rest, track))
    return n, items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sparse0", help="path to sparse/0 dir")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    d = Path(args.sparse0)
    p3d = d / "points3D.bin"
    if not p3d.is_file():
        raise SystemExit(f"{p3d} not found")
    with open(p3d, "rb") as f:
        header = f.read(8)
        first = f.read(32)
    if len(header) != 8 or len(first) != 32:
        raise SystemExit("truncated points3D.bin")
    count = struct.unpack("<Q", header)[0]
    standard_id = struct.unpack("<Q", first[:8])[0]
    xyz_first_id = struct.unpack("<Q", first[24:32])[0]
    if standard_id > count and xyz_first_id == 1:
        raise SystemExit("xyz-first point records detected; this ID-renaming fix is unsafe. Use p16_audit_garden_point_order.py on the original source binary.")

    n, items = read_points3D(p3d)
    seen = set()
    dup_ids = set()
    dup_occurrences = 0
    max_id = 0
    for pid, _rest, _track in items:
        if pid > max_id:
            max_id = pid
        if pid in seen:
            dup_ids.add(pid)
            dup_occurrences += 1
        else:
            seen.add(pid)

    report = {
        "file": str(p3d),
        "n_points": n,
        "n_unique_ids": len(seen),
        "n_duplicate_ids": len(dup_ids),
        "n_duplicate_occurrences": dup_occurrences,
        "duplicate_id_min": min(dup_ids) if dup_ids else None,
        "duplicate_id_max": max(dup_ids) if dup_ids else None,
        "sha256_before": sha256_file(p3d),
    }
    print(json.dumps({k: report[k] for k in report if k != "file"}, indent=2))

    if not args.apply:
        if args.report:
            Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
        return 0

    if dup_occurrences == 0:
        print("no duplicates; nothing to do")
        if args.report:
            Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
        return 0

    tmp = p3d.with_suffix(".bin.fixtmp")
    next_id = max_id + 1
    seen = set()
    renamed = 0
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", n))
        for pid, rest, track in items:
            if pid in seen:
                pid2 = next_id
                next_id += 1
                renamed += 1
            else:
                seen.add(pid)
                pid2 = pid
            f.write(struct.pack("<Q", pid2))
            f.write(rest)
            f.write(struct.pack("<Q", len(track) // 8))
            f.write(track)

    after_sha = sha256_file(tmp)
    tmp.replace(p3d)
    report.update({
        "applied": True,
        "n_renamed": renamed,
        "new_id_first": max_id + 1,
        "new_id_last": next_id - 1,
        "sha256_after": after_sha,
    })
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"applied": True, "renamed": renamed,
                      "sha256_after": after_sha}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
