#!/usr/bin/env python3
"""R17-P2: camera protocol identity for R3/R8 vs the real training protocol.

Determines, with source-line evidence and CPU math on the actual checkpoint:
  - which cameras R3 actually used (synthetic circular, not cameras.json)
  - which camera R8 actually used (identity viewmatrix at the origin)
  - the real training camera protocol (GTDataset + cameras.json + resize)
  - the R17 P3/P4 camera set (cam0 for B1; cam0..cam3 for the 4-in-one-call B4)
  - scene-vs-camera depth evidence (fraction of Gaussians in front of each camera)

Output: R17_CAMERA_IDENTITY.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CAMERA_JSON_CANDIDATES, CKPT, H540, OUT, W540, ensure_out_dirs,
                    load_checkpoint, load_real_cameras, camera_depth_stats,
                    resolve_cameras_json, sha256_file, write_json)

REF_DIR = Path(__file__).resolve().parent.parent / "reference"


def excerpt(path: Path, needle: str, before: int = 4, after: int = 8):
    if not path.is_file():
        return {"found": False, "file": str(path)}
    lines = path.read_text(errors="replace").splitlines()
    for i, l in enumerate(lines):
        if needle in l:
            lo, hi = max(0, i - before), min(len(lines), i + after + 1)
            return {"found": True, "file": str(path), "line_of_needle": i + 1,
                    "lines": [f"{n+1}: {lines[n]}" for n in range(lo, hi)],
                    "file_sha256": sha256_file(path)}
    return {"found": False, "file": str(path)}


def r3_synthetic_camera(i: int, n_train: int, width: int, height: int):
    """Exact copy of R3 get_bicycle_cameras() single-camera construction."""
    angle = 2 * np.pi * i / n_train
    R = torch.tensor([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)],
    ], dtype=torch.float32)
    t = torch.tensor([2.0 * np.cos(angle), 0.5, 2.0 * np.sin(angle)], dtype=torch.float32)
    viewmat = torch.eye(4)
    viewmat[:3, :3] = R
    viewmat[:3, 3] = t
    K = torch.tensor([[float(width), 0, width / 2],
                      [0, float(height), height / 2],
                      [0, 0, 1]], dtype=torch.float32)
    center = -R.T @ t  # world-frame camera center: R @ C + t = 0
    return {"K": K, "viewmatrix": viewmat, "viewmatrix_translation": t,
            "camera_center": center,
            "note": "R3 synthetic circular camera (fx=width, fy=height)"}


def r8_identity_camera(width: int, height: int):
    K = torch.tensor([[float(width), 0, width / 2],
                      [0, float(height), height / 2],
                      [0, 0, 1]], dtype=torch.float32).unsqueeze(0)
    viewmat = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    return {"K": K, "viewmatrix": viewmat,
            "note": "R8 identity viewmatrix (camera at world origin, +Z forward)"}


def mat_summary(M):
    M = np.asarray(M, dtype=np.float64)
    return {"shape": list(M.shape), "rows": [[float(x) for x in row] for row in M]}


def main():
    ensure_out_dirs()
    out = {
        "deliverable": "R17_CAMERA_IDENTITY.json",
        "status": "NOT_EXECUTED",
        "r3_synthetic": {}, "r8_identity": {}, "real_protocol": {},
        "scene_depth_evidence": {}, "source_evidence": {},
    }

    # ---------------- real training camera protocol -------------------------
    try:
        cams540, cams_path, cams_sha = load_real_cameras(W540, H540)
        real_ok = True
    except Exception as e:  # noqa: BLE001
        out["real_protocol"] = {
            "status": "NOT_EXECUTED_CAMERAS_JSON_MISSING",
            "error": repr(e),
            "candidate_paths": [c for c in CAMERA_JSON_CANDIDATES if c],
        }
        real_ok = False

    if real_ok:
        def cam_repr(c):
            return {
                "img_name": c["img_name"], "id": c["id"],
                "orig_wh": [c["orig_width"], c["orig_height"]],
                "render_wh": [c["width"], c["height"]],
                "K": mat_summary(c["K"]),
                "viewmatrix": mat_summary(c["viewmatrix"]),
                "camera_center": [float(x) for x in c["camera_center"]],
                "tanfovx": c["tanfovx"], "tanfovy": c["tanfovy"],
            }
        cams1080, _, _ = load_real_cameras(1920, 1080)
        out["real_protocol"] = {
            "status": "EXECUTED",
            "source_json": cams_path,
            "source_json_sha256": cams_sha,
            "n_cameras": len(cams540),
            "protocol": ("GTDataset: load_cameras_from_json (c2w -> inv) + "
                         "resize_cameras preserving tanfov; 540p render = "
                         "960x540; 1080p render = 1920x1080"),
            "batch_protocols": {
                "B1": "viewmats [1,4,4], Ks [1,3,3], width 960, height 540 (cam0)",
                "B4_R8_style": "ONE call, viewmats [4,4,4], Ks [4,3,3], cams 0..3",
                "B4_R3_style": "4 sequential single-camera calls, cams 0..3",
            },
            "cam0_540": cam_repr(cams540[0]),
            "cam1_540": cam_repr(cams540[1]),
            "cam2_540": cam_repr(cams540[2]),
            "cam3_540": cam_repr(cams540[3]),
            "cam0_1080": {k: cam_repr(cams1080[0])[k] for k in
                          ("img_name", "render_wh", "tanfovx", "tanfovy")},
        }
        out["r17_used_cameras"] = {
            "B1_camera": "cam0 (_DSC8679)",
            "B4_cameras": [c["img_name"] for c in cams540[:4]],
            "indices": [0, 1, 2, 3],
        }

    # ---------------- synthetic camera identification -----------------------
    out["source_evidence"]["r3_camera_generator"] = excerpt(
        REF_DIR / "run_r3_benchmark.py", "def get_bicycle_cameras")
    out["source_evidence"]["r3_fx_fy"] = excerpt(
        REF_DIR / "run_r3_benchmark.py", "'fx': width * 1.0")
    out["source_evidence"]["r8_identity"] = excerpt(
        REF_DIR / "R8_WORKING_ENTRYPOINT.py", "viewmat = torch.eye(4")
    out["source_evidence"]["r8_fx_fy"] = excerpt(
        REF_DIR / "R8_WORKING_ENTRYPOINT.py", "fx, fy = width, height")

    r3c = r3_synthetic_camera(0, 4, W540, H540)
    r3c_n1 = r3_synthetic_camera(0, 1, W540, H540)
    out["r3_synthetic"] = {
        "status": "SYNTHETIC_CIRCULAR_NOT_FROM_CAMERAS_JSON",
        "cam0_K": mat_summary(r3c["K"]),
        "cam0_viewmatrix": mat_summary(r3c["viewmatrix"]),
        "cam0_viewmatrix_translation": [float(x) for x in r3c["viewmatrix_translation"]],
        "cam0_camera_center": [float(x) for x in r3c["camera_center"]],
        "note": ("fx=width=960, fy=height=540, cx=480, cy=270; radius-2 circle "
                 "around origin, height 0.5. At angle=0 (i=0) R is the identity "
                 "matrix and viewmatrix translation t=(2.0, 0.5, 0.0) (camera "
                 "center = -R^T t = (-2.0, -0.5, 0.0)), so camera-z depth equals "
                 "world-z (same as R8 identity camera), but the projection has a "
                 "+2 x-offset."),
        "call_site_discrepancy": {
            "status": "UNRESOLVED_FROM_LOCAL_ARTIFACTS",
            "script_main_call": "get_bicycle_cameras(n_train=1, n_eval=0, ...) "
                                "=> 1 camera (angle=0), each forward = single "
                                "viewmat.unsqueeze(0) call (batch 1)",
            "script_main_call_matches_n1_cam0": bool(torch.allclose(
                r3c_n1["viewmatrix"], r3c["viewmatrix"])),
            "protocol_identity_json_claim": "cameras.n_train=4, batch=4 "
                                            "(HN-540-B4 naming)",
            "run_status_mitigation": "RUN_STATUS.json lists 'Reduce batch size "
                                     "to 1 camera' as an attempted OOM mitigation",
            "interpretation": ("R3 cell names say B4 but the on-disk script "
                               "generates and loops over a single camera; the "
                               "protocol JSON claims n_train=4. R17 benchmark "
                               "(P4) does not depend on resolving this: it uses "
                               "the real cameras.json protocol instead."),
        },
    }
    r8c = r8_identity_camera(W540, H540)
    out["r8_identity"] = {
        "status": "IDENTITY_SYNTHETIC_NOT_FROM_CAMERAS_JSON",
        "K": mat_summary(r8c["K"][0]),
        "viewmatrix": mat_summary(r8c["viewmatrix"][0]),
        "note": "camera at world origin looking +Z (depth == world z)",
    }

    # ---------------- scene depth evidence (CPU on real checkpoint) ---------
    ckpt, ck_info = load_checkpoint(CKPT, require=False)
    if ckpt is None:
        out["scene_depth_evidence"] = {"status": "NOT_EXECUTED_NO_CHECKPOINT"}
        out["status"] = "PARTIAL_NO_CHECKPOINT"
        write_json(OUT / "R17_CAMERA_IDENTITY.json", out)
        return

    means = ckpt["xyz"].float()
    if means.shape[0] > 200_000:
        idx = torch.randperm(means.shape[0])[:200_000]
        means_s = means[idx]
    else:
        means_s = means
    mean_pos = means.mean(dim=0)
    med_pos = means.median(dim=0).values
    bbox = torch.stack([torch.quantile(means_s[:, i], torch.tensor([0.01, 0.99]))
                        for i in range(3)], dim=1)
    ev = {
        "status": "EXECUTED",
        "n_used_for_stats": int(means_s.shape[0]),
        "scene_mean_xyz": [float(x) for x in mean_pos],
        "scene_median_xyz": [float(x) for x in med_pos],
        "scene_bbox_1_99pct": [[float(bbox[0, i]), float(bbox[1, i])] for i in range(3)],
        "identity_camera_r8": camera_depth_stats(means_s, torch.eye(4)),
        "r3_synthetic_cam0": camera_depth_stats(means_s,
                                                r3c["viewmatrix"].to(torch.float32)),
    }
    if real_ok:
        for i in range(4):
            ev[f"real_cam{i}"] = camera_depth_stats(
                means_s, cams540[i]["viewmatrix"].cpu().float())
    out["scene_depth_evidence"] = ev

    out["status"] = "EXECUTED"
    write_json(OUT / "R17_CAMERA_IDENTITY.json", out)
    print(json.dumps({k: out[k] for k in
                      ("status", "r3_synthetic", "r8_identity")}, indent=2, default=str)[:2500],
          flush=True)


if __name__ == "__main__":
    main()
