#!/usr/bin/env python3
"""CPU-only exact-state match of one pre-prune PLY row in the final checkpoint."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    root = Path("/root/codex_fastgs_audit_20260927/output/room_sparse_adj_eval_p1_10k")
    ply_path = root / "point_cloud/iteration_10000/point_cloud.ply"
    checkpoint_path = root / "chkpnt10000.pth"
    index = 689298
    vertices = PlyData.read(str(ply_path))["vertex"].data
    target = {
        "xyz": np.asarray([vertices[name][index] for name in ("x", "y", "z")], dtype=np.float64),
        "scale": np.asarray([vertices[name][index] for name in ("scale_0", "scale_1", "scale_2")], dtype=np.float64),
        "rotation": np.asarray([vertices[name][index] for name in ("rot_0", "rot_1", "rot_2", "rot_3")], dtype=np.float64),
        "opacity": np.asarray([vertices["opacity"][index]], dtype=np.float64),
    }

    checkpoint_state, checkpoint_iteration = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False)
    arrays = {
        "xyz": checkpoint_state[1].detach().cpu().numpy().astype(np.float64),
        "scale": checkpoint_state[4].detach().cpu().numpy().astype(np.float64),
        "rotation": checkpoint_state[5].detach().cpu().numpy().astype(np.float64),
        "opacity": checkpoint_state[6].detach().cpu().numpy().astype(np.float64),
    }
    xyz_distance = np.linalg.norm(arrays["xyz"] - target["xyz"][None, :], axis=1)
    nearest_order = np.argsort(xyz_distance)[:8]
    max_abs_diffs = []
    for row_index in range(arrays["xyz"].shape[0]):
        if xyz_distance[row_index] <= 1e-6:
            max_abs_diffs.append({
                "checkpoint_index": int(row_index),
                "xyz_distance": float(xyz_distance[row_index]),
                "scale_max_abs_diff": float(np.max(np.abs(arrays["scale"][row_index] - target["scale"]))),
                "rotation_max_abs_diff": float(np.max(np.abs(arrays["rotation"][row_index] - target["rotation"]))),
                "opacity_abs_diff": float(np.max(np.abs(arrays["opacity"][row_index] - target["opacity"]))),
            })
    result = {
        "ply_sha256": sha256(ply_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "ply_iteration": 10000,
        "checkpoint_iteration": int(checkpoint_iteration),
        "ply_gaussian_count": int(len(vertices)),
        "checkpoint_gaussian_count": int(arrays["xyz"].shape[0]),
        "ply_candidate_index": index,
        "candidate_properties_raw": {name: value.tolist() for name, value in target.items()},
        "closest_checkpoint_rows_by_xyz": [
            {
                "checkpoint_index": int(i),
                "xyz_distance": float(xyz_distance[i]),
                "scale_max_abs_diff": float(np.max(np.abs(arrays["scale"][i] - target["scale"]))),
                "rotation_max_abs_diff": float(np.max(np.abs(arrays["rotation"][i] - target["rotation"]))),
                "opacity_abs_diff": float(np.max(np.abs(arrays["opacity"][i] - target["opacity"]))),
            }
            for i in nearest_order
        ],
        "matching_xyz_rows_within_1e-6": max_abs_diffs,
        "same_xyz_scale_rotation_opacity_match_within_1e-6": any(
            row["scale_max_abs_diff"] <= 1e-6 and
            row["rotation_max_abs_diff"] <= 1e-6 and
            row["opacity_abs_diff"] <= 1e-6
            for row in max_abs_diffs
        ),
        "interpretation": "The 10K PLY is saved before the final densify/prune event; this CPU-only exact-state comparison checks whether this specific row survives into the post-event checkpoint.",
    }
    output = Path("/root/codex_fastgs_audit_20260927/outlier_diagnostic_dscf4771_20260927/check_final_checkpoint_match.json")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
