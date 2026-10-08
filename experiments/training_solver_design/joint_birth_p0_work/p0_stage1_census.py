import json
import math
import os
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch import nn
from scipy.spatial import cKDTree

from arguments import PipelineParams
from gaussian_renderer import render_structgs
from jb_p0_capture import _restore
from scene import GaussianModel, Scene
from utils.general_utils import build_rotation


ROOT = Path("/root/sadgs_joint_birth_p0_20260928")
DATA = Path("/root/epic05_p16rb/common_data/mipnerf360")
PIPE = Namespace(debug=False, convert_SHs_python=False, compute_cov3D_python=False)
BG = torch.zeros(3, dtype=torch.float32, device="cuda")
MULT = 0.25
MAX_PAIRS = 128
MAX_CHILDREN = 32768


def model_from_state(state, model_path, scene_name):
    model = GaussianModel(3, "hybrid")
    model._xyz = nn.Parameter(torch.empty((0, 3), device="cuda"))
    _restore(model, state)
    args = Namespace(source_path=str(DATA / scene_name), model_path=str(model_path), images="images",
                     resolution=1, white_background=False, data_device="cuda", eval=True)
    scene = Scene(args, model, shuffle=False, resolution_scales=[1.0])
    return model, scene


def fixed_a_cameras(scene):
    ordered = sorted(scene.getTrainCameras(), key=lambda cam: cam.image_name)
    selected = [round(k * (len(ordered) - 1) / 11) for k in range(12)]
    return ordered, [ordered[selected[k]] for k in range(0, 12, 2)], [ordered[selected[k]] for k in range(1, 12, 2)]


def capture_support(model, cameras, candidate_ids, eta):
    ids = candidate_ids.to(device="cuda", dtype=torch.long)
    original = {name: getattr(model, name) for name in ("_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity", "filter_3D")}
    try:
        for name, value in original.items():
            setattr(model, name, nn.Parameter(value.detach()[ids].clone(), requires_grad=False) if name.startswith("_") else value.detach()[ids].clone())
        if model.filter_3D.ndim == 2:
            model.filter_3D = original["filter_3D"].detach()[ids].clone()
        centers, covs, depths, depth_stds = [], [], [], []
        xyz = model.get_xyz.detach()
        scale = model.get_scaling.detach()
        rot = build_rotation(model.get_rotation.detach())
        cov_world = rot @ torch.diag_embed(scale.square()) @ rot.transpose(1, 2)
        for cam in cameras:
            pkg = render_structgs(cam, model, PIPE, BG, MULT)
            c = pkg["cov2D"].detach()
            centers.append(c[:, 3:5].cpu().numpy())
            covs.append(c[:, :3].cpu().numpy())
            depths.append(c[:, 5].cpu().numpy())
            z_direction = cam.world_view_transform[:3, 2]
            zvar = torch.einsum("i,nij,j->n", z_direction, cov_world, z_direction).clamp_min(0)
            depth_stds.append(torch.sqrt(zvar).cpu().numpy())
        return np.stack(centers), np.stack(covs), np.stack(depths), np.stack(depth_stds)
    finally:
        for name, value in original.items():
            setattr(model, name, value)


def main(scene_name):
    out = ROOT / scene_name
    capture = out / "capture"
    entry_blob = torch.load(capture / "split_entry.pt", map_location="cpu", weights_only=False)
    all_blob = torch.load(capture / "all_split.pt", map_location="cpu", weights_only=False)
    entry = entry_blob["model"]
    all_state = all_blob["model"]
    all_state["training_options"] = entry["training_options"]
    all_state["context"] = entry.get("context", {})
    model, scene = model_from_state(entry, out / "model", scene_name)
    ordered, cams_a, cams_b = fixed_a_cameras(scene)
    frozen_names = [cam.image_name for cam in cams_a]
    capture_meta = json.loads((capture / "capture_meta.json").read_text())
    if frozen_names != capture_meta.get("A_camera_names"):
        raise RuntimeError("fixed A camera identities changed between capture and reconstruction")

    parents = entry_blob["parents"].long()
    counts = entry_blob["child_counts"].long()
    candidates = entry_blob["structural_candidates"].long()
    if parents.numel() != counts.numel() or candidates.numel() < 1:
        raise RuntimeError("capture parent/count arrays are invalid")
    row_map = all_blob["row_map"]
    if len(row_map) != all_state["tensors"]["_xyz"].shape[0]:
        raise RuntimeError("stable row map does not match all-split row count")
    children_by_parent = {}
    for row, item in enumerate(row_map):
        if item["kind"] == "child":
            children_by_parent.setdefault(int(item["parent_entry_row"]), []).append(row)

    cand_np = candidates.numpy()
    xyz = entry["tensors"]["_xyz"][candidates].numpy()
    tree = cKDTree(xyz)
    k = min(9, len(candidates))
    neighbor_dist, neighbor_idx = tree.query(xyz, k=k, workers=-1)
    proposal_set = set()
    for i in range(len(candidates)):
        for j in np.atleast_1d(neighbor_idx[i])[1:]:
            a, b = sorted((int(cand_np[i]), int(cand_np[int(j)])))
            if a != b:
                proposal_set.add((a, b))
    proposals = sorted(proposal_set)

    centers, covs, depths, depth_stds = capture_support(model, cams_a, candidates, entry_blob["eta"])
    cand_pos = {int(parent): i for i, parent in enumerate(cand_np)}
    eta = entry_blob["eta"]
    if eta is None:
        ks = np.ones((len(candidates), 3), dtype=np.int32)
        scale = torch.exp(entry["tensors"]["_scaling"][candidates])
        ks[np.arange(len(candidates)), scale.argmax(dim=1).numpy()] = 2
    else:
        ks = torch.sqrt(torch.clamp(eta[candidates], min=1.0)).ceil().int().clamp(min=1).numpy()
    family_children = {int(p): int(c) for p, c in zip(parents.tolist(), counts.tolist())}
    pair_ids = np.asarray(proposals, dtype=np.int64)
    pair_i = np.asarray([cand_pos[int(p)] for p in pair_ids[:, 0]], dtype=np.int64)
    pair_j = np.asarray([cand_pos[int(q)] for q in pair_ids[:, 1]], dtype=np.int64)
    passed = np.zeros(len(proposals), dtype=np.int16)
    valid_views = np.zeros(len(proposals), dtype=np.int16)
    overlap_sum = np.zeros(len(proposals), dtype=np.float64)
    rank_score = np.zeros(len(proposals), dtype=np.float64)
    for v in range(len(cams_a)):
        cp, cq = centers[v, pair_i], centers[v, pair_j]
        vp, vq = covs[v, pair_i], covs[v, pair_j]
        visible = (vp[:, 0] > 0) & (vp[:, 2] > 0) & (vq[:, 0] > 0) & (vq[:, 2] > 0)
        valid_views += visible
        rp = 3.0 * np.sqrt(np.maximum(vp[:, 0], 0.0)); hp = 3.0 * np.sqrt(np.maximum(vp[:, 2], 0.0))
        rq = 3.0 * np.sqrt(np.maximum(vq[:, 0], 0.0)); hq = 3.0 * np.sqrt(np.maximum(vq[:, 2], 0.0))
        iw = np.maximum(0.0, np.minimum(cp[:, 0] + rp, cq[:, 0] + rq) - np.maximum(cp[:, 0] - rp, cq[:, 0] - rq))
        ih = np.maximum(0.0, np.minimum(cp[:, 1] + hp, cq[:, 1] + hq) - np.maximum(cp[:, 1] - hp, cq[:, 1] - hq))
        area_p = (2.0 * rp) * (2.0 * hp)
        area_q = (2.0 * rq) * (2.0 * hq)
        overlap = iw * ih
        ratio = overlap / np.maximum(np.minimum(area_p, area_q), 1e-12)
        depth_ok = np.abs(depths[v, pair_i] - depths[v, pair_j]) <= 2.0 * np.maximum(depth_stds[v, pair_i], depth_stds[v, pair_j])
        ok = visible & (ratio >= 0.25) & depth_ok
        passed += ok
        overlap_sum += np.where(ok, ratio, 0.0)
        child_area_i = area_p / np.maximum(1.0, (ks[pair_i, 0] * ks[pair_i, 1]).astype(np.float64))
        child_area_j = area_q / np.maximum(1.0, (ks[pair_j, 0] * ks[pair_j, 1]).astype(np.float64))
        rank_score += np.where(ok, (child_area_i + child_area_j) * ratio, 0.0)
    legal = np.flatnonzero(passed >= 3)
    pair_rows = []
    for idx in legal:
        p, q = map(int, pair_ids[idx])
        pair_rows.append({"parents": [p, q], "children": family_children[p] + family_children[q],
                          "passing_A_views": int(passed[idx]), "valid_A_views": int(valid_views[idx]),
                          "mean_overlap_ratio": float(overlap_sum[idx] / passed[idx]),
                          "rank_score": float(rank_score[idx]), "predicted_child_bbox_area_x_overlap": float(rank_score[idx])})

    pair_rows.sort(key=lambda r: (-r["rank_score"], r["parents"][0], r["parents"][1]))
    used = set(); chosen = []; proposed_children = 0
    for row in pair_rows:
        p, q = row["parents"]
        if p in used or q in used or proposed_children + row["children"] > MAX_CHILDREN:
            continue
        chosen.append(row); used.update((p, q)); proposed_children += row["children"]
        if len(chosen) == MAX_PAIRS:
            break
    base = {
        "scene": scene_name, "iteration": entry.get("context", {}).get("iteration"),
        "A_camera_names": frozen_names, "train_camera_count": len(ordered),
        "entry_rows": len(entry["tensors"]["_xyz"]), "native_split_parent_count": len(parents),
        "eligible_candidate_count": len(candidates), "proposed_neighbor_pair_count": len(proposals),
        "legal_pair_count": len(pair_rows), "selected_disjoint_pair_count": len(chosen),
        "selected_proposed_child_count": proposed_children, "pair_cap": MAX_PAIRS,
        "proposed_child_cap": MAX_CHILDREN, "pair_selection": chosen,
        "status": "candidate_census_complete", "accepted_work_screen": None,
    }
    (out / "stage1_census.json").write_text(json.dumps(base, indent=2) + "\n")
    if len(chosen) < 16:
        base["status"] = "STOP_FEWER_THAN_16_LEGAL_PAIRS"
        (out / "stage1_census.json").write_text(json.dumps(base, indent=2) + "\n")
        return

    _restore(model, all_state)
    selected_parents = {p for row in chosen for p in row["parents"]}
    child_rows = [r for r, item in enumerate(row_map)
                  if item["kind"] == "child" and int(item["parent_entry_row"]) in selected_parents]
    selected_rows = torch.tensor(child_rows, dtype=torch.long, device="cuda")
    per_view = []
    for cam in cams_a:
        metric_map = torch.ones(cam.image_height * cam.image_width, dtype=torch.int32, device="cuda")
        with torch.no_grad():
            pkg = render_structgs(cam, model, PIPE, BG, MULT, get_flag=True, metric_map=metric_map)
        counts_gpu = pkg["accum_metric_counts"].reshape(-1).to(torch.int64)
        whole = int(counts_gpu.sum().item())
        selected_work = int(counts_gpu[selected_rows].sum().item())
        per_view.append({"camera": cam.image_name, "whole_scene_accepted_work": whole,
                         "selected_pair_children_accepted_work": selected_work,
                         "selected_child_share": selected_work / whole if whole else 0.0})
        del pkg, counts_gpu, metric_map
        torch.cuda.empty_cache()
    whole_sum = sum(x["whole_scene_accepted_work"] for x in per_view)
    selected_sum = sum(x["selected_pair_children_accepted_work"] for x in per_view)
    base["accepted_work_screen"] = {
        "definition": "native render_structgs get_flag=True per-row accum_metric_counts with all-one metric_map",
        "per_A_view": per_view, "pooled_whole_scene_accepted_work": whole_sum,
        "pooled_selected_child_accepted_work": selected_sum,
        "pooled_selected_child_share": selected_sum / whole_sum if whole_sum else 0.0,
        "gate_fraction": 0.10, "passed": (selected_sum / whole_sum if whole_sum else 0.0) >= 0.10,
        "interpretation": "screening heuristic for direct selected-child work only; not a bound on net work after retaining parents",
    }
    base["status"] = "STAGE1_SCREEN_PASS" if base["accepted_work_screen"]["passed"] else "STOP_BELOW_10_PERCENT_CHILD_WORK"
    (out / "stage1_census.json").write_text(json.dumps(base, indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
