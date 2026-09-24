"""Read-only C17-2 ordered-membership statistics collector.

Loads existing real-scene PLY/camera inputs, invokes the installed gsplat baseline
without modifying it, and writes statistical audit results only. It does not alter
renderer behavior, kernels, sorting, or training artifacts.
"""
import gc
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from benchmark_framework.scene import load_ply
from benchmark_framework.cameras import load_cameras_from_json, resize_cameras
from gsplat.rendering import rasterization

OUT = REPO / "results" / "phase-a100" / "ordered_membership_collection.json"
SCENES = {
    "room": (1080, 1080),
    "bicycle": (1920, 1080),
    "garden": (1920, 1080),
}
Q = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 1.0]
THRESHOLDS = [1, 2, 4, 8, 16, 32, 64]


def summary(x):
    x = np.asarray(x)
    if not x.size:
        return {"count": 0}
    p = np.quantile(x, Q)
    return {
        "count": int(x.size), "min": float(p[0]), "p1": float(p[1]),
        "p5": float(p[2]), "p25": float(p[3]), "p50": float(p[4]),
        "p75": float(p[5]), "p95": float(p[6]), "p99": float(p[7]),
        "p99_9": float(p[8]), "max": float(p[9]),
        "mean": float(x.mean()), "std": float(x.std()),
    }


def audit_scene(name, wh):
    width, height = wh
    base = REPO / "data" / "official" / "mipnerf360" / name
    scene = load_ply(str(base / "point_cloud.ply"), device="cuda")
    cams = resize_cameras(load_cameras_from_json(str(base / "cameras.json"), device="cuda"), width, height)
    cam = cams[0]
    quats = torch.nn.functional.normalize(scene["rotations"], dim=-1).contiguous()
    scales = torch.exp(scene["scales"]).contiguous()
    opacities = torch.sigmoid(scene["opacity"]).contiguous()
    viewmats, Ks = cam.viewmatrix.unsqueeze(0), cam.K.unsqueeze(0)
    # Existing baseline operation; metadata is inspected after its deterministic forward.
    with torch.no_grad():
        _, _, meta = rasterization(
            means=scene["xyz"], quats=quats, scales=scales, opacities=opacities,
            colors=scene["shs"], viewmats=viewmats, Ks=Ks, width=width, height=height,
            tile_size=16, packed=True, sh_degree=scene.get("sh_degree", 3), render_mode="RGB",
        )
    torch.cuda.synchronize()
    flat = meta["flatten_ids"].cpu().numpy().astype(np.int64, copy=False)
    offsets = meta["isect_offsets"].cpu().numpy().astype(np.int64, copy=False)
    gids = meta["gaussian_ids"].cpu().numpy().astype(np.int64, copy=False)
    depths = meta["depths"].cpu().numpy().astype(np.float64, copy=False)
    isect = meta["isect_ids"].cpu().numpy().astype(np.int64, copy=False)
    th, tw = offsets.shape[-2:]
    assert offsets.shape[0] == 1
    ntiles, nisect, nnz, nmaster = th * tw, len(flat), len(gids), int(scene["num_points"])
    off = offsets[0].reshape(-1)
    starts = off
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = nisect
    assert np.all(ends >= starts) and int(starts[0]) == 0 and int(ends[-1]) == nisect
    # packed flatten ids index per-visible arrays; map them to master/global Gaussian IDs.
    members = gids[flat]
    deltas, depth_deltas, run_lengths, ranges, width_bits = [], [], [], [], []
    zero = 0
    small = {str(t): 0 for t in THRESHOLDS}
    compared = 0
    depth_nonmonotonic = 0
    depth_pairs = 0
    # Neighbor comparisons are exhaustive over all horizontal, vertical and diagonal neighbors.
    neighbor = {"horizontal": [], "vertical": [], "diagonal_down_right": [], "diagonal_down_left": []}
    shared_dir = {k: [] for k in neighbor}
    for t in range(ntiles):
        seq = members[starts[t]:ends[t]]
        dseq = depths[flat[starts[t]:ends[t]]]
        if len(seq):
            r = int(seq.max() - seq.min())
            ranges.append(r)
            width_bits.append(int(math.ceil(math.log2(r + 1))) if r else 0)
        if len(seq) > 1:
            ds = np.diff(seq)
            ad = np.abs(ds)
            deltas.append(ds)
            compared += len(ds)
            zero += int((ds == 0).sum())
            for x in THRESHOLDS:
                small[str(x)] += int((ad <= x).sum())
            dd = np.diff(dseq)
            depth_pairs += len(dd)
            depth_nonmonotonic += int((dd < 0).sum())
            # Consecutive ±1 steps make an ID-contiguous run. Store all maximal run lengths.
            good = ad == 1
            if good.any():
                cuts = np.flatnonzero(np.diff(np.r_[False, good, False]))
                run_lengths.extend((cuts[1::2] - cuts[::2] + 1).tolist())
        y, x = divmod(t, tw)
        a = set(seq.tolist())
        for label, yy, xx in (("horizontal", y, x + 1), ("vertical", y + 1, x),
                              ("diagonal_down_right", y + 1, x + 1), ("diagonal_down_left", y + 1, x - 1)):
            if yy >= th or xx < 0 or xx >= tw:
                continue
            bseq = members[starts[yy * tw + xx]:ends[yy * tw + xx]]
            b = set(bseq.tolist())
            inter = len(a & b)
            union = len(a | b)
            neighbor[label].append(inter / union if union else 1.0)
            shared_dir[label].append({"a_shared": inter / len(a) if a else 1.0, "b_shared": inter / len(b) if b else 1.0})
    delta = np.concatenate(deltas) if deltas else np.empty(0, dtype=np.int64)
    # global membership frequency has one contribution per tile membership; no within-tile duplicate expected.
    frequency = np.bincount(members, minlength=nmaster)
    nonzero_freq = frequency[frequency > 0]
    exact_tile_dups = sum((ends[t] - starts[t]) - len(set(members[starts[t]:ends[t]].tolist())) for t in range(ntiles))
    # unsigned LEB128 estimate for zigzag signed deltas; it is a coding-model estimate only.
    zigzag = (delta << 1) ^ (delta >> 63)
    vbytes = np.where(zigzag < (1 << 7), 1, np.where(zigzag < (1 << 14), 2, np.where(zigzag < (1 << 21), 3, np.where(zigzag < (1 << 28), 4, 5))))
    result = {
        "scene": name, "camera_index": 0, "input": {"width": width, "height": height, "tile_size": 16, "packed": True, "activated_scales": True},
        "flatten_ids": {"dtype": "int32 (CUDA source/API contract)", "observed_length": nisect, "observed_cpu_cast_dtype": str(flat.dtype), "valid_range_packed": [int(flat.min()), int(flat.max())], "nnz": nnz, "master_gaussian_count": nmaster, "global_id_mapping": "gaussian_ids[flatten_ids]", "bytes": nisect * 4, "per_tile_duplicates": int(exact_tile_dups)},
        "tile_offsets": {"dtype": "int32 (CUDA source/API contract)", "shape": [1, th, tw], "entries": int(offsets.size), "bytes": int(offsets.size * 4), "image_dimension": 1, "sentinel_end_entry": False, "range_definition": "start=offset[t]; end=offset[t+1], or n_isects for final tile", "all_ranges_valid": True},
        "within_tile_delta": {"statistics": summary(delta), "zero_difference_count": int(zero), "zero_difference_ratio": float(zero / compared) if compared else 0.0, "small_absolute_delta_ratio": {k: float(v / compared) if compared else 0.0 for k, v in small.items()}, "delta_pairs": int(compared), "signed_delta_leb128_estimate": {"mean_bytes_per_delta": float(vbytes.mean()) if len(vbytes) else None, "first_id_not_included": True, "interpretation": "coding-model estimate only; not an implementation result"}},
        "cross_tile_overlap": {"directions": {k: {"jaccard": summary(v), "a_shared_fraction": summary([x['a_shared'] for x in shared_dir[k]]), "b_shared_fraction": summary([x['b_shared'] for x in shared_dir[k]]), "pairs": len(v)} for k, v in neighbor.items()}},
        "duplicate_membership": {"per_gaussian_tile_membership": summary(nonzero_freq), "gaussians_with_membership": int(len(nonzero_freq)), "gaussians_without_membership": int(nmaster - len(nonzero_freq)), "total_memberships": nisect, "per_tile_duplicate_entries": int(exact_tile_dups)},
        "run_length": {"consecutive_plus_or_minus_one_run_lengths": summary(run_lengths), "run_count": len(run_lengths), "interpretation": "run length counts IDs in maximal adjacent-|delta|=1 sequences"},
        "id_range": {"per_tile_global_id_range": summary(ranges), "per_tile_bits_for_range": summary(width_bits), "global_id_bits_needed": int(math.ceil(math.log2(max(nmaster, 1)))), "theoretical_unused_bits_vs_int32": int(32 - math.ceil(math.log2(max(nmaster, 1))))},
        "depth_order_locality": {"depth_pairs": int(depth_pairs), "strict_decrease_count": int(depth_nonmonotonic), "strict_decrease_ratio": float(depth_nonmonotonic / depth_pairs) if depth_pairs else 0.0, "ground_truth": "current sorted membership order"},
        "raw_key_structure": {"isect_ids_length": int(len(isect)), "observed_cpu_cast_dtype": str(isect.dtype), "not_retained_by_rasterizer_backward": True},
    }
    del scene, cams, meta, flat, offsets, gids, depths, isect, members
    gc.collect(); torch.cuda.empty_cache()
    return result


def main():
    results = {"collection_status": "complete", "method": "deterministic real-scene baseline metadata capture; no kernel or algorithm changes", "scenes": {}}
    for name, wh in SCENES.items():
        print(f"COLLECT {name}", flush=True)
        results["scenes"][name] = audit_scene(name, wh)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"WROTE {OUT}", flush=True)

if __name__ == "__main__":
    main()
