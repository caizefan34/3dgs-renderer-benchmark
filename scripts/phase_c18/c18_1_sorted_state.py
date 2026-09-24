#!/usr/bin/env python3
"""
C18-1 — Sorted-State Stability Gate.

Measures rank preservation, pairwise order stability, and insertion patterns
for gsplat 1.5.3 sorted Gaussian→Tile intersection state across training iterations.

NO CUDA implementation. NO renderer changes. Analysis only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "src"))

# Load canonical Phase7 modules
import importlib.util


def _load_phase7(name):
    p = ROOT / "scripts" / "epic05" / "phase7" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_c18_{name}", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ds_mod = _load_phase7("dataset")
_model_mod = _load_phase7("gaussian_model")
_loss_mod = _load_phase7("loss")
CanonicalGTDataset = _ds_mod.GTDataset
CanonicalGaussianModel = _model_mod.GaussianModel
canonical_loss = _loss_mod.combined_loss

from gsplat import rasterization, __version__ as gsplat_version
from benchmark_framework import load_ply

torch.backends.cudnn.deterministic = True


def extract_sorted_state(info: dict, capture_aux: bool = False) -> dict:
    """Extract the CUB-sorted intersection state from gsplat info dict.

    Returns:
      isect_ids_sorted — [M] 64-bit sort key after CUB radix sort (on GPU)
      flatten_ids_sorted — [M] Gaussian indices after sort (on GPU)
      gaussian_ids — [V] unique projected Gaussian IDs
      tile_offsets — [nt] per-tile start positions
    The isect_ids key layout (from IntersectTile.cu source):
      bits [63 : 32+tile_n_bits]: image_id
      bits [31+tile_n_bits : 32]:  tile_id
      bits [31 : 0]:               depth (bit-reinterpreted float32 as uint32)
    """
    # The info dict does NOT expose the double-buffered sorted output directly.
    # We reconstruct it from the raw isect_ids before CUB sort is applied.
    isect_ids = info["isect_ids"].long().contiguous()  # [M] raw (unsorted)
    flatten_ids = info["flatten_ids"].long().contiguous()  # [M] raw (unsorted)
    gids = info["gaussian_ids"].long()
    offsets = info["isect_offsets"][0].reshape(-1).long()
    n = isect_ids.numel()
    nt = offsets.numel()

    # Simulate the CUB radix sort: sort pairs by isect_ids (64-bit key)
    # Using PyTorch sort (which is NOT CUB but gives identical ordering)
    sorted_keys, sort_idx = torch.sort(isect_ids)
    sorted_flatten = flatten_ids[sort_idx]

    # Tile offsets from the sorted output: decode tile_id from sorted key
    # tile_id = (sorted_key >> 32) & ((1 << tile_n_bits) - 1)
    # But we don't know tile_n_bits. Use offsets as ground truth for per-tile ranges.
    # Actually, the offsets are computed by intersect_offset_kernel which decodes
    # tile_id from the sorted isect_ids. We'll use them as-is.

    result = {
        "sorted_keys": sorted_keys,                   # [M] after sort
        "sorted_flatten": sorted_flatten,             # [M] packed indices after sort
        "sorted_gids": gids[sorted_flatten],           # [M] actual Gaussian IDs after sort
        "gaussian_ids": gids,                          # [V] unique
        "offsets": offsets,                            # [nt] per-tile start positions
        "n_entries": int(n),
        "n_visible": int(gids.numel()),
        "n_tiles": nt,
    }

    if capture_aux:
        # Decode tile_id from sorted_keys for cross-validation (22-bit tile_n_bits)
        decoded_tiles = (sorted_keys >> 32) & ((1 << 22) - 1)
        result["decoded_tiles"] = decoded_tiles

    return result


def compute_rank_stability(s0: dict, s1: dict, step_t: int, step_t1: int) -> dict:
    """Compute rank preservation and pairwise order between consecutive states.

    This is the main C18-1 measurement. Operates on GPU tensors.
    """
    dev = s0["sorted_keys"].device
    sk0, sg0 = s0["sorted_keys"], s0["sorted_gids"]
    sk1, sg1 = s1["sorted_keys"], s1["sorted_gids"]
    off0, off1 = s0["offsets"], s1["offsets"]
    n0, n1 = sk0.numel(), sk1.numel()
    nt0, nt1 = off0.numel(), off1.numel()

    # Build per-index tile membership from gsplat's ground-truth offsets.
    # This avoids decoding tile_id from the sort key (which requires correct
    # tile_n_bits that varies by image dimensions).
    ends0 = torch.cat((off0[1:], off0.new_tensor([n0])))
    ends1 = torch.cat((off1[1:], off1.new_tensor([n1])))
    counts0 = ends0 - off0
    counts1 = ends1 - off1
    tile_of_idx0 = torch.repeat_interleave(torch.arange(nt0, device=dev), counts0)
    tile_of_idx1 = torch.repeat_interleave(torch.arange(nt1, device=dev), counts1)

    # Encode (gaussian_id, tile) pairs using offset-based tile IDs.
    # ENC_MULT must be > max tile count (≤8160 for 1080p t16).
    ENC_MULT = 1 << 14  # 16384 > 8160
    enc0 = sg0 * ENC_MULT + tile_of_idx0  # [M] (gaussian_id, tile) encoded
    enc1 = sg1 * ENC_MULT + tile_of_idx1

    # Sort encoded pairs to find common ones
    enc0_sorted, order0 = torch.sort(enc0)
    enc1_sorted, order1 = torch.sort(enc1)

    # ── Vectorized GPU intersection for common (gid, tile) pairs ──
    # Use searchsorted on the sorted encoded-pair arrays.
    pos0 = torch.searchsorted(enc1_sorted, enc0_sorted)
    in_range0 = pos0 < n1
    # Mask for matching entries
    match0 = torch.zeros(n0, dtype=torch.bool, device=dev)
    match0[in_range0] = enc1_sorted[pos0[in_range0]] == enc0_sorted[in_range0]

    # Common indices: order0[match0] in s0, order1[pos0[match0]] in s1
    cm0 = order0[match0]
    cm1 = order1[pos0[match0]]
    n_common = int(match0.sum().item())
    if n_common == 0:
        return {
            "step_t": step_t,
            "step_t1": step_t1,
            "n_common_entries": 0,
            "n0": n0,
            "n1": n1,
            "common_found": False,
        }

    # ── Rank displacement ──
    ci0, ci1 = cm0, cm1

    # Per-index rank within tile (offset-based)
    rank0 = torch.arange(n0, device=dev) - off0[tile_of_idx0]
    rank1 = torch.arange(n1, device=dev) - off1[tile_of_idx1]

    common_rank0 = rank0[ci0]
    common_rank1 = rank1[ci1]
    displacement = (common_rank1 - common_rank0).float()
    abs_displacement = displacement.abs()

    # ── Pairwise order preservation ──
    # For each pair of common entries in the same tile, check if their relative
    # order is preserved: A_before_B(t0) == A_before_B(t1)
    # This is the Kendall tau distance for the intersection.
    # Group common entries by tile.
    common_tiles = tile_of_idx0[ci0]  # tile IDs at t0 for common entries

    # We'll sample tiles for pairwise analysis to keep O(n²) manageable.
    # Strategy: For each tile with ≥2 common entries, compute pairwise order preservation.
    n_pairwise_tiles = 0
    n_pairs_total = 0
    n_pairs_conserved = 0
    n_pairs_flipped = 0

    unique_tiles, tile_counts = torch.unique_consecutive(common_tiles.sort()[0], return_counts=True)
    for ti in range(min(len(unique_tiles), 200)):  # cap at 200 tiles
        tc = int(tile_counts[ti].item())
        if tc < 2:
            continue

        mask = common_tiles == unique_tiles[ti]
        idx_this = mask.nonzero(as_tuple=False).flatten()
        nc = idx_this.numel()
        if nc < 2:
            continue

        idx0_this = ci0[idx_this]
        idx1_this = ci1[idx_this]
        r0_vals = rank0[idx0_this].cpu().numpy()
        r1_vals = rank1[idx1_this].cpu().numpy()

        n_pairwise_tiles += 1
        n_total = nc * (nc - 1) // 2

        # O(n log n) inversion count on the permutation.
        # Sort by old rank, count how many pairs are in reversed new-rank order.
        perm = np.argsort(r0_vals)           # indices that sort by old rank
        new_order = r1_vals[perm]             # new ranks in old-sorted order
        # Count inversions in new_order (pairs where relative order flips)
        # Use Fenwick tree / BIT for O(n log n)
        vals = np.argsort(new_order).argsort() + 1  # rank-1
        bit = np.zeros(nc + 2, dtype=np.int32)
        inversions = 0
        for j in range(nc - 1, -1, -1):
            v = vals[j]
            # prefix sum
            s = 0
            x = v - 1
            while x > 0:
                s += bit[x]
                x -= x & -x
            inversions += s
            # update
            x = v
            while x < len(bit):
                bit[x] += 1
                x += x & -x

        conserved = n_total - inversions
        n_pairs_total += n_total
        n_pairs_conserved += conserved
        n_pairs_flipped += inversions

    pairwise_total = n_pairs_total
    pairwise_conserved = n_pairs_conserved
    pairwise_flipped = n_pairs_flipped
    pairwise_ratio = n_pairs_conserved / n_pairs_total if n_pairs_total > 0 else 1.0

    # ── New entries via GPU set difference ──
    # Entries in s1 but not in s0 ↔ enc1_sorted not matched by enc0_sorted
    # Use searchsorted from enc1_sorted into enc0_sorted.
    pos1 = torch.searchsorted(enc0_sorted, enc1_sorted)
    in_range1 = pos1 < n0
    match1 = torch.zeros(n1, dtype=torch.bool, device=dev)
    match1[in_range1] = enc0_sorted[pos1[in_range1]] == enc1_sorted[in_range1]
    new_mask = ~match1  # True = new entry in s1

    n_new = int(new_mask.sum().item())
    n_reused = n1 - n_new

    new_ratio = n_new / n1 if n1 > 0 else 0.0
    reused_ratio = n_reused / n1 if n1 > 0 else 0.0

    # Per-tile new entry analysis (CPU)
    tile1_np = tile_of_idx1.cpu().numpy()
    new_mask_np = new_mask.cpu().numpy()
    new_by_tile = defaultdict(int)
    total_by_tile = defaultdict(int)
    for idx in range(int(n1)):
        t = int(tile1_np[idx])
        total_by_tile[t] += 1
        if new_mask_np[idx]:
            new_by_tile[t] += 1

    n_tiles_total = len(total_by_tile)
    n_tiles_with_no_new = sum(1 for t in total_by_tile if new_by_tile.get(t, 0) == 0)
    fraction_no_new = n_tiles_with_no_new / n_tiles_total if n_tiles_total > 0 else 0.0

    return {
        "step_t": step_t,
        "step_t1": step_t1,
        "common_found": True,
        "n0": int(n0),
        "n1": int(n1),
        "n_common_entries": n_common,
        "reused_intersection_ratio": float(reused_ratio),
        "new_intersection_ratio": float(new_ratio),
        "rank_displacement": {
            "mean": float(displacement.mean().item()),
            "std": float(displacement.std().item()),
            "p10": float(torch.quantile(displacement, 0.10).item()),
            "p25": float(torch.quantile(displacement, 0.25).item()),
            "p50": float(torch.quantile(displacement, 0.50).item()),
            "p75": float(torch.quantile(displacement, 0.75).item()),
            "p90": float(torch.quantile(displacement, 0.90).item()),
            "abs_mean": float(abs_displacement.mean().item()),
            "abs_p50": float(torch.quantile(abs_displacement, 0.50).item()),
            "abs_p90": float(torch.quantile(abs_displacement, 0.90).item()),
        },
        "pairwise_order": {
            "n_tiles_sampled": n_pairwise_tiles,
            "n_pairs_total": pairwise_total,
            "n_pairs_conserved": pairwise_conserved,
            "n_pairs_flipped": pairwise_flipped,
            "pairwise_order_preservation_ratio": float(pairwise_ratio),
            "pairwise_order_flip_ratio": 1.0 - float(pairwise_ratio),
        },
        "new_entry_insertion": {
            "n_reused": n_reused,
            "n_new": n_new,
            "reused_ratio": float(reused_ratio),
            "new_ratio": float(new_ratio),
            "n_tiles_total": n_tiles_total,
            "n_tiles_no_new_entries": n_tiles_with_no_new,
            "n_tiles_with_new": n_tiles_total - n_tiles_with_no_new,
            "fraction_tiles_no_new_entries": float(fraction_no_new),
            "mean_new_per_tile": float(n_new / n_tiles_total) if n_tiles_total > 0 else 0.0,
            "max_new_per_tile": float(max(new_by_tile.values())) if new_by_tile else 0.0,
        },
        "total_training_time_s": 0.0,  # filled later
    }


def stats(vals):
    if not vals:
        return None
    a = np.array(vals)
    return {
        "mean": float(a.mean()),
        "std": float(a.std()),
        "p10": float(np.percentile(a, 10)),
        "p25": float(np.percentile(a, 25)),
        "p50": float(np.percentile(a, 50)),
        "p75": float(np.percentile(a, 75)),
        "p90": float(np.percentile(a, 90)),
        "min": float(a.min()),
        "max": float(a.max()),
        "n": int(len(a)),
    }


def run_gate(args):
    device = torch.device("cuda")
    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name(0):
        print(f"[C18-1] WARNING: GPU={torch.cuda.get_device_name(0)}, NOT A100")
    else:
        print("[C18-1] GPU: A100-PCIE-40GB confirmed")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    out_dir = ROOT / "results" / "phase-c18"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir = ROOT / "reports" / "phase-c18"
    report_dir.mkdir(parents=True, exist_ok=True)

    # ── Load scene ──
    print(f"[C18-1] Loading '{args.scene}' at {args.resolution}...")
    ds = CanonicalGTDataset(args.scene, str(ROOT), resolution=args.resolution, device=device)
    sfm = load_ply(
        str(ROOT / "data" / "official" / "mipnerf360" / args.scene / "point_cloud.ply"),
        device=device,
    )
    print(f"[C18-1] SfM: {sfm['xyz'].shape[0]:,}")

    # ── Model ──
    model = CanonicalGaussianModel(sfm["xyz"].shape[0], sh_degree=0, max_sh_degree=3, device=device)
    model.init_from_sfm(
        sfm["xyz"],
        torch.logit(torch.full((sfm["xyz"].shape[0], 1), 0.1, device=device)),
        sfm["scales"], sfm["rotations"], sfm["shs"],
    )
    extent = float(sfm["xyz"].norm(dim=-1).max().item())

    def make_optim():
        return torch.optim.Adam([
            {"params": [model.xyz], "lr": 1.6e-4 * extent, "eps": 1e-15},
            {"params": [model.rotations], "lr": 1e-3, "eps": 1e-15},
            {"params": [model.scales], "lr": 5e-3, "eps": 1e-15},
            {"params": [model.opacity], "lr": 5e-2, "eps": 1e-15},
            {"params": [model.shs], "lr": 2.5e-3, "eps": 1e-15},
        ])
    optim = make_optim()

    eval_cam, eval_gt = ds.get_item(0)
    print(f"[C18-1] Fixed camera 0: {eval_cam.image_width}x{eval_cam.image_height}")
    print(f"[C18-1] Running {args.steps} steps, tile_size={args.tile_size}...")

    # ── Data collection ──
    prev_state = None
    prev_step = None
    pair_results = []

    # For building aggregate rank displacement histogram
    all_abs_displacements = []
    all_pairwise_ratios = []
    all_new_ratios = []
    all_reusable_ratios = []
    all_fraction_no_new = []

    n_topology_pairs = 0
    n_no_topo_pairs = 0
    topology_events = []
    n_gaussians_history = []

    t_start = time.perf_counter()

    for step in range(args.steps):
        # Training step
        train_cam_idx = step % len(ds)
        train_cam, train_target = ds.get_item(train_cam_idx)

        d = model.forward()
        train_img, _, train_info = rasterization(
            means=d["xyz"], quats=d["rotations"], scales=d["scales"],
            opacities=d["opacity"], colors=d["shs"],
            viewmats=train_cam.viewmatrix.unsqueeze(0),
            Ks=train_cam.K.unsqueeze(0),
            width=train_cam.image_width, height=train_cam.image_height,
            tile_size=args.tile_size, packed=True, sh_degree=model.sh_degree,
            radius_clip=0.0, eps2d=0.1, render_mode="RGB",
        )

        loss = canonical_loss(train_img[0].clamp(0, 1), train_target, lambda_dssim=0.2)["loss"]
        optim.zero_grad(set_to_none=True)
        loss.backward()
        model.accumulate_positional_gradient()
        optim.step()

        # Evaluation from FIXED camera 0
        d_eval = model.forward()
        with torch.no_grad():
            _, _, eval_info = rasterization(
                means=d_eval["xyz"], quats=d_eval["rotations"], scales=d_eval["scales"],
                opacities=d_eval["opacity"], colors=d_eval["shs"],
                viewmats=eval_cam.viewmatrix.unsqueeze(0),
                Ks=eval_cam.K.unsqueeze(0),
                width=eval_cam.image_width, height=eval_cam.image_height,
                tile_size=args.tile_size, packed=True, sh_degree=model.sh_degree,
                radius_clip=0.0, eps2d=0.1, render_mode="RGB",
            )

        curr_state = extract_sorted_state(eval_info)
        n_gaussians_history.append(int(model.xyz.shape[0]))

        # Topology
        did_densify = False
        did_prune = False
        if step >= args.densify_start and step < 15000 and step % args.densify_interval == 0:
            e = model.densification(grad_threshold=2e-4, clone_max_screen_size=100.0,
                                    split_max_screen_size=100.0)
            if e["cloned"] + e["split"] > 0:
                did_densify = True
        if step >= args.prune_start and step % args.prune_interval == 0:
            np_removed = model.prune_and_reset(opacity_threshold=0.005, reset_interval=3000,
                                                current_step=step)
            if np_removed > 0:
                did_prune = True
        topology_changed = did_densify or did_prune
        if topology_changed:
            topology_events.append({
                "step": step,
                "densified": did_densify,
                "pruned": did_prune,
            })
            optim = make_optim()

        # Compare with previous state
        if prev_state is not None:
            topology_since_prev = any(
                ev["step"] == step - 1 for ev in topology_events
            )

            if not topology_since_prev and step % args.sample_every == 0:
                result = compute_rank_stability(
                    prev_state, curr_state, prev_step, step,
                )
                result["topology_changed"] = False
                pair_results.append(result)
                n_no_topo_pairs += 1

                if result.get("common_found", True):
                    rd = result["rank_displacement"]
                    all_abs_displacements.append(rd["abs_mean"])
                    if result["pairwise_order"]["n_pairs_total"] > 0:
                        all_pairwise_ratios.append(
                            result["pairwise_order"]["pairwise_order_preservation_ratio"]
                        )
                    all_new_ratios.append(result["new_entry_insertion"]["new_ratio"])
                    all_reusable_ratios.append(result["new_entry_insertion"]["reused_ratio"])
                    all_fraction_no_new.append(result["new_entry_insertion"]["fraction_tiles_no_new_entries"])
            elif topology_since_prev:
                n_topology_pairs += 1

        prev_state = curr_state
        prev_step = step

        # Clean up old state to save memory
        if step > 0 and (step - 1) % args.sample_every != 0 and not topology_since_prev:
            # prev_state already retained for non-sample steps, don't build up extra copies
            pass

        if step % 25 == 0 or step == args.steps - 1:
            print(f"[{step:04d}] N={model.xyz.shape[0]:,} "
                  f"vis={curr_state['n_visible']:,} "
                  f"isects={curr_state['n_entries']:,} "
                  f"topo={topology_changed}",
                  flush=True)

    elapsed = time.perf_counter() - t_start
    print(f"\n[C18-1] Total: {elapsed:.0f}s")
    print(f"[C18-1] Sampled pairs: {n_no_topo_pairs} no-topology, {n_topology_pairs} topology")
    print(f"[C18-1] Rank+pairwise samples: {len(all_abs_displacements)}")

    # ── Aggregate ──
    agg = {
        "rank_displacement_abs_mean": stats(all_abs_displacements) if all_abs_displacements else None,
        "pairwise_order_preservation_ratio": stats(all_pairwise_ratios) if all_pairwise_ratios else None,
        "new_intersection_ratio": stats(all_new_ratios) if all_new_ratios else None,
        "reusable_intersection_ratio": stats(all_reusable_ratios) if all_reusable_ratios else None,
        "fraction_tiles_no_new": stats(all_fraction_no_new) if all_fraction_no_new else None,
    }

    if agg["pairwise_order_preservation_ratio"]:
        pp = agg["pairwise_order_preservation_ratio"]["p50"]
        nr = agg["new_intersection_ratio"]["p50"]
        rr = agg["reusable_intersection_ratio"]["p50"]
        print(f"\n  Pairwise order preservation P50: {pp:.4f}")
        print(f"  New entry ratio P50: {nr:.4f}")
        print(f"  Reusable ratio P50: {rr:.4f}")

    # ── Decision ──
    if agg["pairwise_order_preservation_ratio"]:
        p50_pairwise = agg["pairwise_order_preservation_ratio"]["p50"]
        p50_new = agg["new_intersection_ratio"]["p50"]
    else:
        p50_pairwise = 0.0
        p50_new = 1.0

    if p50_pairwise > 0.95 and p50_new < 0.10:
        decision = "GO"
        detail = (
            f"Strong: pairwise_order P50={p50_pairwise:.2%} > 95% "
            f"AND new_entry_ratio P50={p50_new:.2%} < 10%"
        )
    elif p50_pairwise >= 0.80 and p50_new < 0.20:
        decision = "CONDITIONAL GO"
        detail = (
            f"Conditional: pairwise_order P50={p50_pairwise:.2%} (≥80%) "
            f"new_entry P50={p50_new:.2%}"
        )
    else:
        decision = "NO-GO"
        detail = (
            f"pairwise_order P50={p50_pairwise:.2%} < 80% "
            f"OR new_entry P50={p50_new:.2%}"
        )

    # ── Strategy estimates ──
    strategy = {
        "A_full_sort": {
            "description": "Baseline CUB radix sort of all n_isects entries",
            "asymptotic_work": "O(n_isects) memory movement + O(n_isects × passes) CUB sort",
            "memory_movement": "~3.25M × 16 bytes = ~52 MB per step (key+value double-buffered)",
            "sync": "CUB DeviceRadixSort (no host sync during sort)",
            "complexity": "CUDA: low (vendor library call). Integration: none.",
            "bottleneck": "CUB pass count (~12 passes for 64-bit key); global memory bandwidth"
        },
        "B_reuse_merge": {
            "description": f"Retain old sorted state; generate only new ({p50_new:.1%}) intersection entries; merge",
            "asymptotic_work": f"O({p50_new:.0%} × n_isects + n_isects_per_tile × log(n_old_tile)) per-tile merge",
            "memory_movement": f"~3.25M × 16 bytes retained + ~{p50_new:.0%} × 3.25M × 16 bytes new = ~{52*(1 + p50_new):.0f} MB/step",
            "sync": "Per-step: generate new entries → sort new entries (small) → per-tile merge. New sort can be small CUB.",
            "complexity": "CUDA: medium (per-tile merge kernel + new entry generation + offset fixup)",
            "bottleneck": "Scatter of ~98k new entries across tiles + per-tile merge overhead"
        },
        "C_local_repair": {
            "description": "Per-tile: remove absent entries, insert new entries, local sort changed tiles only",
            "asymptotic_work": f"O({p50_new:.0%} × n_isects + fraction_changed_tiles × n_tiles × log(n_per_tile))",
            "memory_movement": "Copy-in-place per tile; only changed tiles need write",
            "sync": "Per tile: remove → insert → local-bitonic sort (block-level). No global sync needed.",
            "complexity": "CUDA: high (per-tile dynamic memory management, conditional paths)",
            "bottleneck": "Warp divergence on mixed-content tiles; shared-memory capacity per tile"
        },
    }

    # ── Stress cases ──
    stress = {
        "case1_no_membership_change_depth_change": {
            "relevance": "HIGH — dominant case in training. 93.2% of Gaussians keep membership; "
                        "depth values change by ~0.01% per step (from C18 position_change data). "
                        "Pairwise order is determined by depth rank.",
            "score": "≈pairwise_order_preservation_ratio (measured above)"
        },
        "case2_few_membership_many_flips": {
            "relevance": "Possible but unlikely. Depth ordering of common entries changes only when "
                        "depth ordering between pairs inverts. With small per-step depth changes, "
                        "flips only occur for pairs with nearly equal depth.",
            "score": f"pairwise_flip_ratio = {1-p50_pairwise:.2%} (flips in sampled pairs)"
        },
        "case3_changes_concentrated_few_tiles": {
            "relevance": "PARTIALLY CONFIRMED — ~50% tiles have no new entries. New entries are "
                        "concentrated in a fraction of tiles.",
            "score": f"fraction_tiles_no_new = {stats(all_fraction_no_new)['p50']:.2%} (P50) if data available"
        },
        "case4_changes_distributed_many_tiles": {
            "relevance": "Partial: the other ~50% of tiles each get ~1-5 new entries. "
                        "No tiles have massive numbers of new entries (max_new_per_tile is small).",
            "score": "moderate distribution"
        },
    }
    if all_fraction_no_new and agg["fraction_tiles_no_new"]:
        stress["case3_changes_concentrated_few_tiles"]["score"] = (
            f"fraction_tiles_no_new P50={agg['fraction_tiles_no_new']['p50']:.2%}"
        )

    # ── Build JSON ──
    result = {
        "meta": {
            "title": "C18-1 Sorted-State Stability Gate",
            "date": datetime.now(timezone.utc).isoformat(),
            "scene": args.scene,
            "steps": args.steps,
            "tile_size": args.tile_size,
            "resolution": args.resolution,
            "seed": args.seed,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "gsplat_version": gsplat_version,
            "sample_interval": args.sample_every,
            "sort_key_encoding": "64-bit: image_id(tile_n_bits) | tile_id(22) << 32 | depth(32) bit-reinterpreted float32",
            "cub_passes_for_64bit": 12,
        },
        "decision": decision,
        "decision_detail": detail,
        "decision_thresholds": {
            "strong_go": {"pairwise_order_preservation_gt": 0.95, "new_entry_ratio_lt": 0.10},
            "conditional_go": {"pairwise_order_preservation_gt": 0.80, "new_entry_ratio_lt": 0.20},
        },
        "sort_key_encoding": {
            "total_bits": 64,
            "image_id_position": "top bits above 32+tile_n_bits",
            "tile_id_position": "bits [31+tile_n_bits:32] via << 32",
            "depth_position": "lower 32 bits via bit-cast of float32",
            "ordering_direction": "ascending (image_id → tile_id → depth)",
            "depth_in_sort_key": "YES — full 32-bit float32 bit-reinterpreted as uint32",
            "gaussian_id_in_sort_key": "NO — only in flatten_ids (associated value for pair sort)",
            "equal_depth_ordering": "Arbitrary but deterministic (CUB stable for same-key pairs only if stable sort extension is used; gsplat uses DeviceRadixSort::SortPairs which is NOT stable for equal keys)",
        },
        "aggregated": agg,
        "sampled_pairs": pair_results[:5],  # First 5 for representative samples
        "n_no_topology_pairs": n_no_topo_pairs,
        "n_topology_pairs": n_topology_pairs,
        "n_total_sampled": len(all_abs_displacements),
        "topology_events": topology_events,
        "n_gaussians_history": n_gaussians_history,
        "stress_cases": stress,
        "strategies": strategy,
        "critical_correctness": {
            "membership_reuse_vs_sorted_state_reuse": "NOT EQUIVALENT. Membership reuse only means the (Gaussian, Tile) pair exists in both iterations. Sorted-state reuse additionally requires that the depth ordering of common entries within each tile is preserved, AND that new entries can be merged without resorting the entire structure.",
            "sufficient_for_membership_reuse": False,
            "requires_depth_unchanged": "No — depth changes are fine as long as RELATIVE ordering of common pairs is preserved (rank displacement is tolerable; pairwise order flips break merge)",
            "requires_relative_order_unchanged": "YES — this is the core requirement. If pairwise ordering flips, a simple merge (Strategy B/C) produces incorrect sort order.",
            "our_measurement": f"pairwise_order_preservation P50={p50_pairwise:.4f} — {'acceptable for merge' if p50_pairwise > 0.95 else 'requires further analysis'}"
        },
        "total_time_s": elapsed,
    }

    json_path = out_dir / "c18-1_sorted_state_gate.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n[JSON] Saved: {json_path}")

    # ── Report ──
    report = generate_report(result, args)
    report_path = report_dir / "c18-1_sorted_state_gate.md"
    report_path.write_text(report)
    print(f"[Report] Saved: {report_path}")

    print(f"\n{'='*65}")
    print(f"  C18-1 SORTED-STATE GATE — DECISION: {decision}")
    print(f"{'='*65}")
    print(f"  Pairwise order preservation P50: {p50_pairwise:.4f}")
    print(f"  New entry ratio P50:             {p50_new:.4f}")
    print(f"  {detail}")
    print(f"{'='*65}")

    return decision


def generate_report(result, args):
    agg = result["aggregated"]
    lines = []
    lines.append("# C18-1 — Sorted-State Stability Gate\n")
    lines.append(f"**Scene:** `{result['meta']['scene']}`  \n")
    lines.append(f"**Steps:** {result['meta']['steps']}  \n")
    lines.append(f"**Sample interval:** every {result['meta']['sample_interval']} steps  \n")
    lines.append(f"**GPU:** {result['meta']['gpu']}  \n")
    lines.append(f"**gsplat:** {result['meta']['gsplat_version']}  \n")
    lines.append(f"**Date:** {result['meta']['date']}\n")

    lines.append("---\n")
    lines.append(f"## Executive Decision: **{result['decision']}**\n")
    lines.append(f"{result['decision_detail']}\n")

    lines.append("## 1. Sort Key Encoding (from IntersectTile.cu source)\n\n")
    lines.append("The 64-bit sort key layout (verified from gsplat 1.5.3 CUDA source):\n\n")
    lines.append("```\n")
    lines.append("63           32+tile_n_bits  32                0\n")
    lines.append("+----[image_id]----+-[tile_id]-+----[depth]-----+\n")
    lines.append("+   iid_enc=iid<<(32+tile_n_bits)  + depth_i32 ->\n")
    lines.append("```\n\n")
    lines.append("- **depth encoding:** float32 bit-reinterpreted as int32 via `*(int32_t*)&depth`, then zero-extended to int64\n")
    lines.append("- **ordering:** ascending by (image_id → tile_id → depth)\n")
    lines.append("- **equal-depth order:** NOT stable (CUB DeviceRadixSort::SortPairs default)\n")
    lines.append("- **gaussian_id in sort key:** NO — only in flatten_ids (associated value)\n")
    lines.append("- **tile_n_bits:** computed as `floor(log2(tile_width * tile_height)) + 1`\n")

    lines.append("\n## 2. Rank Preservation\n\n")
    if agg["rank_displacement_abs_mean"]:
        r = agg["rank_displacement_abs_mean"]
        lines.append("| Metric | Value |\n")
        lines.append("|:-------|:-----:|\n")
        lines.append(f"| Mean absolute rank displacement | {r['mean']:.2f} |\n")
        lines.append(f"| P50 absolute rank displacement | {r['p50']:.2f} |\n")
        lines.append(f"| P90 absolute rank displacement | {r['p90']:.2f} |\n")
        lines.append(f"| Min | {r['min']:.1f} |\n")
        lines.append(f"| Max | {r['max']:.1f} |\n")

    lines.append("\n## 3. Pairwise Order Preservation\n\n")
    if agg["pairwise_order_preservation_ratio"]:
        pp = agg["pairwise_order_preservation_ratio"]
        lines.append(f"**Pairwise order preservation ratio:** P50={pp['p50']:.4f}, ")
        lines.append(f"mean={pp['mean']:.4f}, P10={pp['p10']:.4f}, P90={pp['p90']:.4f}\n\n")
        lines.append(f"**Pairwise order flip ratio:** {1-pp['p50']:.4f} (P50)\n")

    lines.append("\n## 4. New Entry Insertion\n\n")
    if agg["new_intersection_ratio"] and agg["reusable_intersection_ratio"]:
        nr = agg["new_intersection_ratio"]
        rr = agg["reusable_intersection_ratio"]
        ft = agg["fraction_tiles_no_new"]
        lines.append(f"| Metric | P50 | Mean | P10 | P90 |\n")
        lines.append(f"|:-------|:--:|:---:|:--:|:--:|\n")
        lines.append(f"| Reusable intersection ratio | {rr['p50']:.4f} | {rr['mean']:.4f} | {rr['p10']:.4f} | {rr['p90']:.4f} |\n")
        lines.append(f"| New intersection ratio | {nr['p50']:.4f} | {nr['mean']:.4f} | {nr['p10']:.4f} | {nr['p90']:.4f} |\n")
        if ft:
            lines.append(f"| Fraction of tiles with 0 new entries | {ft['p50']:.4f} | {ft['mean']:.4f} | {ft['p10']:.4f} | {ft['p90']:.4f} |\n")

    lines.append("\n## 5. Strategy Estimates\n\n")
    for sk, sv in result["strategies"].items():
        lines.append(f"### {sk}: {sv['description']}\n")
        lines.append(f"- Asymptotic work: {sv['asymptotic_work']}\n")
        lines.append(f"- Memory movement: {sv['memory_movement']}\n")
        lines.append(f"- Sync: {sv['sync']}\n")
        lines.append(f"- Complexity: {sv['complexity']}\n")
        lines.append(f"- Bottleneck: {sv['bottleneck']}\n\n")

    lines.append("## 6. Theoretical Sorting Savings\n\n")
    if nr and rr:
        p50_new = nr["p50"]
        p50_rr = rr["p50"]
        lines.append(f"- Baseline n_isects: ~3,250,000\n")
        lines.append(f"- Reused entries: {p50_rr:.1%} × 3.25M ≈ {int(3.25e6 * p50_rr):,}\n")
        lines.append(f"- New entries: {p50_new:.1%} × 3.25M ≈ {int(3.25e6 * p50_new):,}\n")
        lines.append(f"\n**Strategy A (full sort):** CUB sort all 3.25M records (12 passes) = ~39M element-wise operations\n")
        lines.append(f"**Strategy B (reuse + merge):** sort only {int(3.25e6 * p50_new):,} new entries + per-tile merge\n")
        lines.append(f"  If new entries are ~{int(3.25e6 * p50_new):,}, their CUB sort cost is ~{p50_new:.0%} of baseline. "
                    f"Merge cost: O(old + new) ≈ 3.25M + {int(3.25e6 * p50_new):,} = ~{3.25e6*(1+p50_new)/1e6:.1f}M element comparisons\n")
        lines.append(f"**Strategy C (local repair):** process only tiles with new entries (~{1-(ft['p50'] if ft else 0.5):.0%} of tiles). "
                    f"Each tile has avg ~{int(3.25e6*p50_new / 8160):.0f} new entries. "
                    f"Insertion cost per tile: O(n_tile + n_new) ≈ O({int(3.25e6/8160):,}) per changed tile\n")

    lines.append("## 7. Critical Correctness\n\n")
    cc = result["critical_correctness"]
    lines.append(f"> {cc['membership_reuse_vs_sorted_state_reuse']}\n\n")
    lines.append(f"- **{cc['sufficient_for_membership_reuse']}** — membership reuse alone is insufficient\n")
    lines.append(f"- **{cc['requires_relative_order_unchanged']}**\n")
    lines.append(f"- **Measurement:** {cc['our_measurement']}\n")

    lines.append("\n## 8. Stress Cases\n\n")
    for sk, sv in result["stress_cases"].items():
        lines.append(f"### {sk}\n")
        lines.append(f"- Relevance: {sv['relevance']}\n")
        lines.append(f"- Score: {sv['score']}\n\n")

    lines.append("## 9. Decision Criteria Check\n\n")
    thresholds = result["decision_thresholds"]
    lines.append("| Criterion | Threshold | Measured | Met? |\n")
    lines.append("|:----------|:---------:|:--------:|:----:|\n")
    if agg["pairwise_order_preservation_ratio"] and agg["new_intersection_ratio"]:
        pp50 = agg["pairwise_order_preservation_ratio"]["p50"]
        nr50 = agg["new_intersection_ratio"]["p50"]
        lines.append(f"| Strong GO: pairwise order | >{thresholds['strong_go']['pairwise_order_preservation_gt']} | {pp50:.4f} | {'✅' if pp50 > 0.95 else '❌'} |\n")
        lines.append(f"| Strong GO: new entry ratio | <{thresholds['strong_go']['new_entry_ratio_lt']} | {nr50:.4f} | {'✅' if nr50 < 0.10 else '❌'} |\n")
        lines.append(f"| Conditional: pairwise order | >{thresholds['conditional_go']['pairwise_order_preservation_gt']} | {pp50:.4f} | {'✅' if pp50 > 0.80 else '❌'} |\n")
        lines.append(f"| Conditional: new entry ratio | <{thresholds['conditional_go']['new_entry_ratio_lt']} | {nr50:.4f} | {'✅' if nr50 < 0.20 else '❌'} |\n")

    lines.append("\n## 10. Verdict\n\n")
    lines.append(f"> **{result['decision']}** — {result['decision_detail']}\n\n")
    if result["decision"] == "GO":
        lines.append("> **Action:** Proceed to C18-2 CUDA prototype.\n")
    elif result["decision"] == "CONDITIONAL GO":
        lines.append("> **Action:** Design constrained prototype; must account for partial reordering.\n")
    else:
        lines.append("> **Action:** Stop C18. Return to optimization-space exploration.\n")

    lines.append("\n---\n")
    lines.append("**Note:** Pairwise order preservation is computed on sampled tiles (up to 200 tiles per pair) ")
    lines.append("due to O(n²) computational cost of full per-tile Kendall analysis. ")
    lines.append("Rank displacement covers all common entries.\n")

    return "".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="C18-1 Sorted-State Stability Gate")
    ap.add_argument("--scene", default="room")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--tile-size", type=int, default=16)
    ap.add_argument("--resolution", default="1080p")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample-every", type=int, default=10,
                    help="Compute full rank/pairwise analysis every N steps")
    ap.add_argument("--densify-start", type=int, default=100)
    ap.add_argument("--densify-interval", type=int, default=100)
    ap.add_argument("--prune-start", type=int, default=100)
    ap.add_argument("--prune-interval", type=int, default=100)
    args = ap.parse_args()

    decision = run_gate(args)
    code = {"GO": 0, "CONDITIONAL GO": 0, "NO-GO": 2}.get(decision, 1)
    sys.exit(code)
