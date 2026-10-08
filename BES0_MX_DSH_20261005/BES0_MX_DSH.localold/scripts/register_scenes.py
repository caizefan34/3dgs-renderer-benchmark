#!/usr/bin/env python3
"""Register the three existing BES0 scenes into config/scene_manifest.json.

Data preparation only: reads COLMAP poses, checkpoint metadata, camera
sequences and computes SHA256 hashes. No scientific computation, no GPU,
no training. Splits are deterministic (sorted camera-id, evenly spaced,
non-overlapping 12 dev + 12 heldout per scene). Must run on MX because it
reads MX-local checkpoint/COLMAP paths.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pycolmap

REPO = Path("/home/liaoyuanjun/3dgs-renderer-benchmark")
POOL = Path("/mnt/storage_pool/liaoyuanjun")
WORK = Path(os.environ.get("BES0_WORK_ROOT", str(POOL / "bes0_mx")))

SCENES = {
    "room": {
        "checkpoint": str(POOL / "retiregs/RETIRE0_room_source/checkpoints/iter_15000.pt"),
        "sparse": str(REPO / "data/datasets/mipnerf360/room/sparse/0"),
        "images": str(REPO / "data/datasets/mipnerf360/room/images"),
        "camera_sequence": str(POOL / "retiregs/RETIRE0_room_source/camera_sequence.npy"),
    },
    "bicycle": {
        "checkpoint": str(POOL / "sota_rebase_v2_runs/B1_bicycle_30k/checkpoints/iter_30000.pt"),
        "sparse": str(REPO / "data/datasets/mipnerf360/bicycle/sparse/0"),
        "images": str(REPO / "data/datasets/mipnerf360/bicycle/images"),
        "camera_sequence": str(POOL / "sota_rebase_v2_runs/B1_bicycle_30k/camera_sequence.npy"),
    },
    "garden": {
        "checkpoint": str(POOL / "sota_rebase_v2_runs/B1_garden_30k/checkpoints/iter_30000.pt"),
        "sparse": str(REPO / "data/datasets/mipnerf360/garden/sparse/0"),
        "images": str(REPO / "data/datasets/mipnerf360/garden/images"),
        "camera_sequence": str(POOL / "sota_rebase_v2_runs/B1_garden_30k/camera_sequence.npy"),
    },
}

N_DEV = 12
N_HELDOUT = 12


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_dir_listing(path: str):
    """Deterministic hash over sorted (relname, size) of a directory."""
    p = Path(path)
    entries = []
    for f in sorted(p.rglob("*")):
        if f.is_file():
            entries.append((f.relative_to(p).as_posix(), f.stat().st_size))
    h = hashlib.sha256()
    for rel, size in entries:
        h.update(rel.encode("utf-8") + b"\0" + str(size).encode("ascii") + b"\0")
    return h.hexdigest(), len(entries)


def even_spaced(pool, k):
    pool = sorted(pool)
    n = len(pool)
    idx = sorted(set(int(round(i * (n - 1) / (k - 1))) for i in range(k))) if k > 1 else [0]
    # guard against duplicates collapsing below k
    if len(idx) < k:
        seen = set(idx)
        for j in range(n):
            if j not in seen:
                idx.append(j)
                seen.add(j)
            if len(idx) == k:
                break
        idx = sorted(idx)[:k]
    return [pool[i] for i in idx]


def main():
    import torch
    manifest = {
        "schema": "bes0.scene_manifest.v1",
        "protocol": "BES0-v1",
        "repo_head": os.popen("git -C %s rev-parse HEAD" % REPO).read().strip(),
        "split_rule": "sorted camera-id; dev and heldout evenly spaced, non-overlapping; train=rest",
        "n_dev": N_DEV, "n_heldout": N_HELDOUT,
        "scenes": {},
    }
    for name, cfg in SCENES.items():
        rec = pycolmap.Reconstruction(cfg["sparse"])
        ids = sorted(rec.images.keys())
        dev = even_spaced(ids, N_DEV)
        remaining = [i for i in ids if i not in set(dev)]
        heldout = even_spaced(remaining, N_HELDOUT)
        train = [i for i in ids if i not in set(dev) | set(heldout)]
        ckpt_meta = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=False)
        gt_hash, gt_count = sha256_dir_listing(cfg["images"])
        scene = {
            "checkpoint": cfg["checkpoint"],
            "checkpoint_sha256": sha256_file(cfg["checkpoint"]),
            "checkpoint_num_points": int(ckpt_meta["num_points"]),
            "checkpoint_active_sh_degree": int(ckpt_meta["active_sh_degree"]),
            "sparse_dir": cfg["sparse"],
            "sparse_hashes": {
                "cameras.bin": sha256_file(os.path.join(cfg["sparse"], "cameras.bin")),
                "images.bin": sha256_file(os.path.join(cfg["sparse"], "images.bin")),
                "points3D.bin": sha256_file(os.path.join(cfg["sparse"], "points3D.bin")),
            },
            "images_dir": cfg["images"],
            "images_count": gt_count,
            "images_listing_sha256": gt_hash,
            "camera_sequence": cfg["camera_sequence"],
            "camera_sequence_sha256": sha256_file(cfg["camera_sequence"]),
            "camera_id_count": len(ids),
            "camera_id_range": [ids[0], ids[-1]],
            "dev_camera_ids": dev,
            "heldout_camera_ids": heldout,
            "train_camera_count": len(train),
            "dev_heldout_disjoint": set(dev).isdisjoint(set(heldout)),
        }
        manifest["scenes"][name] = scene
    out = WORK / "BES0_MX_DSH" / "config" / "scene_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "OK", "out": str(out),
                      "scenes": {k: {"n": v["checkpoint_num_points"],
                                      "imgs": v["camera_id_count"],
                                      "dev": len(v["dev_camera_ids"]),
                                      "heldout": len(v["heldout_camera_ids"]),
                                      "train": v["train_camera_count"],
                                      "disjoint": v["dev_heldout_disjoint"]}
                                  for k, v in manifest["scenes"].items()}}, indent=2))


if __name__ == "__main__":
    main()
