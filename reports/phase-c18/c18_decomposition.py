#!/usr/bin/env python3
"""
C18 — Quantitative Decomposition Gate.

Measures consecutive-iteration Gaussian→Tile membership stability to
determine whether incremental intersection rebuild is computationally viable.

No renderer modifications. No CUDA implementation. No literature survey.

IMPORTANT GAUSSIAN ID STABILITY:
  - gaussian_ids in the info dict are INDICES into the parameter arrays (xyz, etc.).
  - When NO topology change (densification/pruning), these indices are stable and
    can be directly compared across iterations: same index = same Gaussian.
  - When pruning occurs, indices shift because elements are removed from the middle.
    Per-Gaussian comparison via index is then INVALID.
  - When densification occurs (append-only), existing indices remain stable; new
    Gaussians get new indices beyond the original range.
  - We ONLY perform per-Gaussian membership comparison on no-topology pairs.
  - For topology-change pairs, we report the rate of events and their impact
    on intersection count (which is valid even with shifting indices).
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
sys.path.insert(0, str(ROOT / "src"))

from gsplat import rasterization
from benchmark_framework import load_ply
from scripts.epic05.phase7.dataset import GTDataset
from scripts.epic05.phase7.gaussian_model import GaussianModel
from scripts.epic05.phase7.loss import combined_loss

torch.backends.cudnn.deterministic = True


def extract_visible_data(info: dict) -> dict:
    """Extract per-Gaussian visible tile data from gsplat info dict.

    Returns GPU tensors for fast comparison.
    """
    gids = info["gaussian_ids"].long()                       # [V] unique projected GIDs
    flat = info["flatten_ids"].long()                         # [M] indices into gids
    offsets = info["isect_offsets"][0].reshape(-1).long()     # [N_tiles]
    n = flat.numel()
    nt = offsets.numel()

    # Build tile_id for every intersection entry
    ends = torch.cat((offsets[1:], offsets.new_tensor([n])))
    counts = ends - offsets
    tile_ids = torch.repeat_interleave(
        torch.arange(nt, dtype=torch.long, device=offsets.device), counts
    )

    # Actual Gaussian ID for every entry: gids[flatten_ids[i]]
    entry_gids = gids[flat]  # [M]

    # Sorted unique visible Gaussian IDs
    visible_gids, _ = torch.sort(gids)

    # Build encoded (gid, tile) pairs for fast comparison
    # Use a large multiplier to pack: gid * MAX_TILES + tile_id
    # MAX_TILES must be > max possible tile count (image tiles ≤ ~8192)
    MAX_INSTANCE = 1 << 24  # 16M Gaussians max
    MAX_TILES_INSTANCE = 1 << 24
    encoded = torch.sort(entry_gids * MAX_TILES_INSTANCE + tile_ids).values

    return {
        "entry_gids": entry_gids,          # [M] Gaussian ID for every intersection entry
        "tile_ids": tile_ids,              # [M] tile ID for every entry
        "encoded": encoded,                # [M] sorted encoded (gid * 2^24 + tile_id)
        "visible_gids": visible_gids,      # [V] sorted unique visible GIDs
        "n_entries": int(n),
        "n_visible": int(gids.numel()),
        "n_tiles_active": int((counts > 0).sum().item()),
    }


def compute_intersection_overlap(d0: dict, d1: dict) -> dict:
    """Compute intersection-level overlap metrics between two iterations.

    Uses sorted encoded (gid, tile) pairs and binary search.
    Valid even when indices shift (topology change) because the (gid, tile)
    encoding simply becomes a different (gid', tile') pair — overlap drops.

    WARNING: After pruning, the SAME physical Gaussian may have a different gid.
    This means intersection_overlap UNDERSTATES true reuse for topology pairs.
    """
    enc0, enc1 = d0["encoded"], d1["encoded"]
    device = enc0.device

    # Two-pointer intersection count on sorted arrays
    i = j = 0
    common = 0
    n0, n1 = enc0.numel(), enc1.numel()
    while i < n0 and j < n1:
        if enc0[i] == enc1[j]:
            common += 1
            i += 1
            j += 1
        elif enc0[i] < enc1[j]:
            i += 1
        else:
            j += 1

    n_union = n0 + n1 - common
    intersection_overlap = common / n_union if n_union > 0 else 1.0
    reusable_ratio = common / n1 if n1 > 0 else 1.0
    new_ratio = (n1 - common) / n1 if n1 > 0 else 0.0
    removed_ratio = (n0 - common) / n0 if n0 > 0 else 0.0

    return {
        "intersection_overlap": float(intersection_overlap),
        "reusable_intersection_ratio": float(reusable_ratio),
        "new_intersection_ratio": float(new_ratio),
        "removed_intersection_ratio": float(removed_ratio),
        "n_common_entries": int(common),
    }


def compute_membership_stability_per_gaussian(d0: dict, d1: dict) -> dict:
    """Compute per-Gaussian membership stability for NO-TOPOLOGY pairs only.

    This is VALID only when Gaussian indices haven't shifted (no prune).
    For each Gaussian ID visible in both iterations, compares tile sets.
    """
    gids0, tiles0 = d0["entry_gids"], d0["tile_ids"]
    gids1, tiles1 = d1["entry_gids"], d1["tile_ids"]

    # Build sorted arrays grouped by gid
    # Pack: gid * MAX + tile_id and sort by key
    MAX_TILES = 1 << 24
    packed0, order0 = torch.sort(gids0 * MAX_TILES + tiles0)
    packed1, order1 = torch.sort(gids1 * MAX_TILES + tiles1)

    sgid0 = gids0[order0]  # sorted by (gid, tile)
    stile0 = tiles0[order0]
    sgid1 = gids1[order1]
    stile1 = tiles1[order1]

    # Move to CPU for per-Gaussian set operations
    sgid0_np = sgid0.cpu().numpy()
    stile0_np = stile0.cpu().numpy()
    sgid1_np = sgid1.cpu().numpy()
    stile1_np = stile1.cpu().numpy()

    # Build tile-set dict for each Gaussian
    def build_tile_dict(sgid, stile):
        d = {}
        g_start = 0
        n = len(sgid)
        for i in range(n + 1):
            if i == n or sgid[i] != sgid[g_start]:
                d[int(sgid[g_start])] = set(stile[g_start:i].tolist())
                g_start = i
        return d

    tset0 = build_tile_dict(sgid0_np, stile0_np)
    tset1 = build_tile_dict(sgid1_np, stile1_np)

    # Only common Gaussian IDs
    common_gids = set(tset0.keys()) & set(tset1.keys())

    unchanged = 0
    changed = 0
    tile_counts_t0 = []
    tile_counts_t1 = []
    tile_delta = []

    for g in common_gids:
        s0 = tset0[g]
        s1 = tset1[g]
        tile_counts_t0.append(len(s0))
        tile_counts_t1.append(len(s1))
        tile_delta.append(len(s1) - len(s0))
        if s0 == s1:
            unchanged += 1
        else:
            changed += 1

    total_common = len(common_gids)
    stable_ratio = unchanged / total_common if total_common > 0 else 0.0
    changed_ratio = changed / total_common if total_common > 0 else 0.0

    return {
        "n_common_gaussians": total_common,
        "stable_membership_ratio": float(stable_ratio),
        "changed_membership_ratio": float(changed_ratio),
        "mean_tiles_per_gaussian_t0": float(np.mean(tile_counts_t0)) if tile_counts_t0 else 0.0,
        "mean_tiles_per_gaussian_t1": float(np.mean(tile_counts_t1)) if tile_counts_t1 else 0.0,
        "median_tile_delta": float(np.median(tile_delta)) if tile_delta else 0.0,
    }


def stats(arr):
    """Compute descriptive stats for an array of floats."""
    if not arr:
        return None
    a = np.array(arr)
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


def run_decomposition(args):
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        # For local testing fallback
        print("[C18] WARNING: CUDA not available, using CPU (results not valid)")
        device = torch.device("cpu")
    elif "A100" not in torch.cuda.get_device_name(0):
        print(f"[C18] WARNING: GPU={torch.cuda.get_device_name(0)}, not A100")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    out = ROOT / "results" / "phase-c18"
    out.mkdir(parents=True, exist_ok=True)
    reports_dir = ROOT / "reports" / "phase-c18"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # ── Load dataset ──
    print(f"[C18] Loading scene '{args.scene}' at {args.resolution}...")
    ds = GTDataset(args.scene, str(ROOT), resolution=args.resolution, device=device)
    sfm = load_ply(
        str(ROOT / "data" / "official" / "mipnerf360" / args.scene / "point_cloud.ply"),
        device=device,
    )
    print(f"[C18] SfM points: {sfm['xyz'].shape[0]:,}")

    # ── Init model ──
    model = GaussianModel(
        sfm["xyz"].shape[0], sh_degree=0, max_sh_degree=3, device=device,
    )
    model.init_from_sfm(
        sfm["xyz"],
        torch.logit(torch.full((sfm["xyz"].shape[0], 1), 0.1, device=device)),
        sfm["scales"],
        sfm["rotations"],
        sfm["shs"],
    )
    extent = float(sfm["xyz"].norm(dim=-1).max().item())
    print(f"[C18] Extent: {extent:.2f}")

    def make_optim():
        return torch.optim.Adam([
            {"params": [model.xyz], "lr": 1.6e-4 * extent, "eps": 1e-15},
            {"params": [model.rotations], "lr": 1e-3, "eps": 1e-15},
            {"params": [model.scales], "lr": 5e-3, "eps": 1e-15},
            {"params": [model.opacity], "lr": 5e-2, "eps": 1e-15},
            {"params": [model.shs], "lr": 2.5e-3, "eps": 1e-15},
        ])

    optim = make_optim()

    # ── Fixed evaluation camera (camera 0) ──
    eval_cam, eval_gt = ds.get_item(0)
    print(f"[C18] Fixed camera 0: {eval_cam.image_width}x{eval_cam.image_height}")
    print(f"[C18] Running {args.steps} steps, tile_size={args.tile_size}...")

    # ── Data collection ──
    all_step_data = []          # raw per-step captures
    step_records = {}           # step -> extract_visible_data dict
    topology_events = []        # steps with topology change
    n_gaussians_history = []

    # For overlap tracking across different lags
    pair_metrics_no_topo = []       # for stable-to-stable pairs
    pair_metrics_topo = []          # when topology occurred at t+1
    pair_metrics_lag2 = []          # t → t+2 across no-topology span
    pair_metrics_lag4 = []          # t → t+4 across no-topology span

    t0 = time.perf_counter()

    for step in range(args.steps):
        # ── Training step ──
        train_cam_idx = step % len(ds)
        train_cam, train_target = ds.get_item(train_cam_idx)

        d = model.forward()
        train_img, _, train_info = rasterization(
            means=d["xyz"],
            quats=d["rotations"],
            scales=d["scales"],
            opacities=d["opacity"],
            colors=d["shs"],
            viewmats=train_cam.viewmatrix.unsqueeze(0),
            Ks=train_cam.K.unsqueeze(0),
            width=train_cam.image_width,
            height=train_cam.image_height,
            tile_size=args.tile_size,
            packed=True,
            sh_degree=model.sh_degree,
            radius_clip=0.0,
            eps2d=0.1,
            render_mode="RGB",
        )

        # Loss + backward
        loss_data = combined_loss(
            train_img[0].clamp(0, 1), train_target, lambda_dssim=0.2,
        )
        loss = loss_data["loss"]
        optim.zero_grad(set_to_none=True)
        loss.backward()
        model.accumulate_positional_gradient()
        optim.step()

        # ── Render from FIXED camera 0 for measurement ──
        with torch.no_grad():
            fixed_img, _, eval_info = rasterization(
                means=d["xyz"],
                quats=d["rotations"],
                scales=d["scales"],
                opacities=d["opacity"],
                colors=d["shs"],
                viewmats=eval_cam.viewmatrix.unsqueeze(0),
                Ks=eval_cam.K.unsqueeze(0),
                width=eval_cam.image_width,
                height=eval_cam.image_height,
                tile_size=args.tile_size,
                packed=True,
                sh_degree=model.sh_degree,
                radius_clip=0.0,
                eps2d=0.1,
                render_mode="RGB",
            )

        curr_data = extract_visible_data(eval_info)
        step_records[step] = curr_data
        n_gaussians_history.append(int(model.xyz.shape[0]))

        # ── Densification & Pruning ──
        did_densify = False
        did_prune = False

        if step >= args.densify_start and step < 15000 and step % args.densify_interval == 0:
            e = model.densification(
                grad_threshold=2e-4,
                clone_max_screen_size=100.0,
                split_max_screen_size=100.0,
            )
            if e["cloned"] + e["split"] > 0:
                did_densify = True

        if step >= args.prune_start and step % args.prune_interval == 0:
            n_pruned = model.prune_and_reset(
                opacity_threshold=0.005, reset_interval=3000, current_step=step,
            )
            if n_pruned > 0:
                did_prune = True

        topology_changed = did_densify or did_prune
        if topology_changed:
            topology_events.append({
                "step": step,
                "densified": did_densify,
                "pruned": did_prune,
                "n_gaussians_before": n_gaussians_history[-2] if len(n_gaussians_history) > 1 else n_gaussians_history[-1],
                "n_gaussians_after": int(model.xyz.shape[0]),
            })
            optim = make_optim()

        # ── Compute overlap metrics with previous step ──
        if step > 0:
            prev_data = step_records[step - 1]

            # Intersection-level overlap (valid even with topology change at index level,
            # though UNDERSTATES true reuse due to index shifting)
            isect_metrics = compute_intersection_overlap(prev_data, curr_data)

            # Per-Gaussian membership (VALID only for no-topology pairs)
            membership_metrics = {
                "n_common_gaussians": 0,
                "stable_membership_ratio": 0.0,
                "changed_membership_ratio": 0.0,
            }
            if not topology_changed:
                membership_metrics = compute_membership_stability_per_gaussian(
                    prev_data, curr_data,
                )

            combined = {
                "step_t": step - 1,
                "step_t1": step,
                "topology_changed": topology_changed,
                "n_gaussians_t": n_gaussians_history[-2],
                "n_gaussians_t1": n_gaussians_history[-1],
                **isect_metrics,
                **membership_metrics,
            }

            if topology_changed:
                pair_metrics_topo.append(combined)
            else:
                pair_metrics_no_topo.append(combined)

            # Lag-2
            if step >= 2 and not topology_changed:
                prev2_data = step_records.get(step - 2)
                if prev2_data:
                    m = compute_intersection_overlap(prev2_data, curr_data)
                    pair_metrics_lag2.append({
                        "step_t": step - 2,
                        "step_t1": step,
                        **m,
                    })

            # Lag-4
            if step >= 4:
                prev4_data = step_records.get(step - 4)
                # Only valid if none of the intermediate steps had topology
                # Simple: check if any of step-4..step had topology
                had_topo_intermediate = topology_changed
                for s in range(step - 3, step):
                    if s >= 0 and any(
                        te["step"] == s for te in topology_events
                    ):
                        had_topo_intermediate = True
                if prev4_data and not had_topo_intermediate:
                    m = compute_intersection_overlap(prev4_data, curr_data)
                    pair_metrics_lag4.append({
                        "step_t": step - 4,
                        "step_t1": step,
                        **m,
                    })

        if step % 25 == 0 or step == args.steps - 1:
            print(f"[{step:04d}] N={model.xyz.shape[0]:,} "
                  f"vis={curr_data['n_visible']:,} "
                  f"isects={curr_data['n_entries']:,} "
                  f"topo={topology_changed}",
                  flush=True)

    total_time = time.perf_counter() - t0
    print(f"\n[C18] Total time: {total_time:.0f}s")
    print(f"[C18] No-topology pairs: {len(pair_metrics_no_topo)}")
    print(f"[C18] Topology pairs: {len(pair_metrics_topo)}")
    print(f"[C18] Lag-2 pairs: {len(pair_metrics_lag2)}")
    print(f"[C18] Lag-4 pairs: {len(pair_metrics_lag4)}")

    # ── Aggregate ──
    def agg_pairs(pairs, label):
        if not pairs:
            return None
        reusable = [p["reusable_intersection_ratio"] for p in pairs]
        new_r = [p["new_intersection_ratio"] for p in pairs]
        removed = [p["removed_intersection_ratio"] for p in pairs]
        overlap = [p["intersection_overlap"] for p in pairs]
        stable = [p.get("stable_membership_ratio", 0.0) for p in pairs]
        changed = [p.get("changed_membership_ratio", 0.0) for p in pairs]

        result = {
            "reusable_intersection_ratio": stats(reusable),
            "new_intersection_ratio": stats(new_r),
            "removed_intersection_ratio": stats(removed),
            "intersection_overlap": stats(overlap),
            "stable_membership_ratio": stats(stable),
            "changed_membership_ratio": stats(changed),
            "n_pairs": len(pairs),
        }
        rp50 = result["reusable_intersection_ratio"]["p50"]
        cp50 = result["changed_membership_ratio"]["p50"]
        sp50 = result["stable_membership_ratio"]["p50"]
        print(f"  [{label}] reusable(P50)={rp50:.4f} "
              f"stable(P50)={sp50:.4f} "
              f"changed(P50)={cp50:.4f} "
              f"n={len(pairs)}")
        return result

    print("\n[C18] Aggregated statistics:")
    no_topo_agg = agg_pairs(pair_metrics_no_topo, "NO-TOPOLOGY")
    topo_agg = agg_pairs(pair_metrics_topo, "TOPOLOGY")
    lag2_agg = agg_pairs(pair_metrics_lag2, "LAG-2")
    lag4_agg = agg_pairs(pair_metrics_lag4, "LAG-4")

    # ── Phase analysis ──
    print("\n[C18] Phase analysis:")
    phases = {}
    for label, lo, hi in [("early", 0, 100), ("middle", 100, 400), ("late", 400, args.steps)]:
        subset = [r for r in pair_metrics_no_topo if lo <= r["step_t1"] < hi]
        phases[label] = agg_pairs(subset, f"Phase {label} ({lo}-{hi})")

    # ── Step-binned ──
    step_bins = {}
    for r in pair_metrics_no_topo:
        bin_idx = (r["step_t1"] // 100) * 100
        step_bins.setdefault(bin_idx, []).append(r)
    binned = {}
    for k, v in sorted(step_bins.items()):
        if v:
            binned[f"step_{k}_{k+99}"] = agg_pairs(v, f"BIN {k}")

    # ── Decision logic ──
    if no_topo_agg:
        p50_reusable = no_topo_agg["reusable_intersection_ratio"]["p50"]
        p50_changed = no_topo_agg["changed_membership_ratio"]["p50"]
        mean_stable = no_topo_agg["stable_membership_ratio"]["mean"]
        p50_stable = no_topo_agg["stable_membership_ratio"]["p50"]
    else:
        p50_reusable = 0.0
        p50_changed = 1.0
        mean_stable = 0.0
        p50_stable = 0.0

    # Decision thresholds from spec
    if p50_changed < 0.10 and p50_reusable > 0.70:
        decision = "GO"
        decision_detail = (
            f"Strong: changed_membership_ratio P50={p50_changed:.2%} < 10% "
            f"AND reusable_intersection_ratio P50={p50_reusable:.2%} > 70%"
        )
    elif p50_changed < 0.30 and p50_reusable > 0.40:
        decision = "CONDITIONAL GO"
        decision_detail = (
            f"Conditional: changed={p50_changed:.2%} (threshold <30%), "
            f"reusable={p50_reusable:.2%} (threshold >40%)"
        )
    elif p50_changed < 0.50 and p50_reusable > 0.30:
        decision = "HIGH RISK"
        decision_detail = (
            f"High risk: changed={p50_changed:.2%} (threshold <50%), "
            f"reusable={p50_reusable:.2%} (threshold >30%)"
        )
    else:
        decision = "NO-GO"
        decision_detail = (
            f"changed={p50_changed:.2%} >= 50% OR "
            f"reusable={p50_reusable:.2%} < 40%"
        )

    # ── Build topology impact ──
    tfrac = len(pair_metrics_topo) / (len(pair_metrics_no_topo) + len(pair_metrics_topo))
    topology_impact = {
        "n_events": len(topology_events),
        "fraction_of_steps": tfrac,
        "events_detail": topology_events,
    }
    if topo_agg:
        topology_impact["reusable_during_topo"] = topo_agg["reusable_intersection_ratio"]

    # ── Build JSON result ──
    # Representative samples: first 3 no-topology + last 1
    samples = pair_metrics_no_topo[:3] + (
        pair_metrics_no_topo[-1:] if len(pair_metrics_no_topo) > 3 else []
    )

    result = {
        "meta": {
            "title": "C18 Quantitative Decomposition Gate",
            "date": datetime.now(timezone.utc).isoformat(),
            "scene": args.scene,
            "steps": args.steps,
            "tile_size": args.tile_size,
            "resolution": args.resolution,
            "seed": args.seed,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "gsplat_version": __import__("gsplat").__version__,
        },
        "decision": decision,
        "decision_detail": decision_detail,
        "decision_thresholds": {
            "strong_go": {"changed_membership_ratio_lt": 0.10, "reusable_intersection_ratio_gt": 0.70},
            "conditional_go": {"changed_membership_ratio_range": "10-30%", "reusable_intersection_ratio_range": "40-70%"},
            "high_risk": {"changed_membership_ratio_range": "30-50%", "reusable_intersection_ratio_range": "30-40%"},
            "no_go": {"changed_membership_ratio_gte": 0.50, "or_reusable_intersection_ratio_lt": 0.40},
        },
        "aggregated_no_topology": no_topo_agg,
        "aggregated_topology_pairs": topo_agg,
        "aggregated_lag2": lag2_agg,
        "aggregated_lag4": lag4_agg,
        "phase_analysis": phases,
        "binned_by_100_steps": binned,
        "topology_impact": topology_impact,
        "n_gaussians_history": n_gaussians_history,
        "total_training_time_s": total_time,
        "representative_samples": samples,
    }

    json_path = out / "c18_decomposition_gate.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n[JSON] Saved: {json_path}")

    # ── Generate report ──
    report = generate_report(result, args)
    report_path = reports_dir / "c18_decomposition_gate.md"
    report_path.write_text(report)
    print(f"[Report] Saved: {report_path}")

    # ── Terminal banner ──
    print(f"\n{'='*65}")
    print(f"  C18 DECOMPOSITION GATE — DECISION: {decision}")
    print(f"{'='*65}")
    if no_topo_agg:
        print(f"  Reusable intersection (P50): {p50_reusable:.4f}")
        print(f"  Changed membership (P50):    {p50_changed:.4f}")
        print(f"  Stable membership (mean):    {mean_stable:.4f}")
    print(f"  {decision_detail}")
    print(f"{'='*65}")

    return decision


def generate_report(result, args):
    lines = []
    lines.append("# C18 — Quantitative Decomposition Gate\n")
    lines.append(f"**Scene:** `{result['meta']['scene']}`  ")
    lines.append(f"**Steps:** {result['meta']['steps']}  ")
    lines.append(f"**Tile size:** {result['meta']['tile_size']}  ")
    lines.append(f"**GPU:** {result['meta']['gpu']}  ")
    lines.append(f"**gsplat:** {result['meta']['gsplat_version']}  ")
    lines.append(f"**Date:** {result['meta']['date']}  ")
    lines.append(f"**Fixed camera:** camera 0 (same viewpoint every step)\n")

    lines.append("---\n")
    lines.append(f"## Executive Decision: **{result['decision']}**\n")
    lines.append(f"{result['decision_detail']}\n")

    # ── 1. Gaussian Stability ──
    lines.append("---\n")
    lines.append("## 1. Consecutive-Iteration Gaussian Stability\n")
    agg = result["aggregated_no_topology"]
    if agg:
        r = agg["reusable_intersection_ratio"]
        c = agg["changed_membership_ratio"]
        s = agg["stable_membership_ratio"]
        o = agg["intersection_overlap"]
        n = agg["new_intersection_ratio"]

        lines.append(f"### Intersection-level overlap (no-topology pairs, n={agg['n_pairs']})\n")
        lines.append(f"| Metric | P10 | P25 | P50 | P75 | P90 | Mean |\n")
        lines.append(f"|:-------|:--:|:--:|:--:|:--:|:--:|:----:|\n")
        lines.append(f"| Reusable intersection ratio | {r['p10']:.4f} | {r['p25']:.4f} | **{r['p50']:.4f}** | {r['p75']:.4f} | {r['p90']:.4f} | {r['mean']:.4f} |\n")
        lines.append(f"| Changed membership ratio | {c['p10']:.4f} | {c['p25']:.4f} | **{c['p50']:.4f}** | {c['p75']:.4f} | {c['p90']:.4f} | {c['mean']:.4f} |\n")
        lines.append(f"| Stable membership ratio | {s['p10']:.4f} | {s['p25']:.4f} | {s['p50']:.4f} | {s['p75']:.4f} | {s['p90']:.4f} | {s['mean']:.4f} |\n")
        lines.append(f"| Intersection overlap (Jaccard) | {o['p10']:.4f} | {o['p25']:.4f} | {o['p50']:.4f} | {o['p75']:.4f} | {o['p90']:.4f} | {o['mean']:.4f} |\n")
        lines.append(f"| New intersection ratio | {n['p10']:.4f} | {n['p25']:.4f} | {n['p50']:.4f} | {n['p75']:.4f} | {n['p90']:.4f} | {n['mean']:.4f} |\n")

    # ── 2. Temporal Window ──
    lines.append("\n### Temporal window (intersection overlap)\n")
    lines.append("| Lag | P50 reusable | P50 overlap | n_pairs |\n")
    lines.append("|:---:|:-----------:|:----------:|:-------:|\n")
    if agg:
        lines.append(f"| t→t+1 | {agg['reusable_intersection_ratio']['p50']:.4f} | {agg['intersection_overlap']['p50']:.4f} | {agg['n_pairs']} |\n")
    lag2 = result["aggregated_lag2"]
    lag4 = result["aggregated_lag4"]
    if lag2:
        lines.append(f"| t→t+2 | {lag2['reusable_intersection_ratio']['p50']:.4f} | {lag2['intersection_overlap']['p50']:.4f} | {lag2['n_pairs']} |\n")
    if lag4:
        lines.append(f"| t→t+4 | {lag4['reusable_intersection_ratio']['p50']:.4f} | {lag4['intersection_overlap']['p50']:.4f} | {lag4['n_pairs']} |\n")

    # ── 3. Phase Analysis ──
    lines.append("\n## 3. Phase Analysis\n")
    for phase_name in ["early", "middle", "late"]:
        pd = result["phase_analysis"].get(phase_name)
        if pd:
            rp50 = pd["reusable_intersection_ratio"]["p50"]
            cp50 = pd["changed_membership_ratio"]["p50"]
            lines.append(f"- **{phase_name}**: reusable P50={rp50:.4f}, changed P50={cp50:.4f}, n={pd['n_pairs']}\n")

    # ── 4. Topology Impact ──
    lines.append("\n## 4. Topology-Change Impact\n")
    ti = result["topology_impact"]
    lines.append(f"- **Events:** {ti['n_events']} topology changes across {result['meta']['steps']} steps\n")
    lines.append(f"- **Fraction of steps with topology:** {ti['fraction_of_steps']:.2%}\n")
    lines.append(f"- **Intersection reuse during topology-change steps:** ")
    if ti.get("reusable_during_topo"):
        tr = ti["reusable_during_topo"]
        lines.append(f"P50={tr['p50']:.4f} (vs no-topology P50={agg['reusable_intersection_ratio']['p50']:.4f})\n")
    else:
        lines.append("N/A (per-Gaussian comparison invalid due to index shifting)\n")
    lines.append("- **Note:** Per-Gaussian membership comparison is INVALID for topology-change ")
    lines.append("pairs because pruning shifts Gaussian parameter indices. ")
    lines.append("Only intersection-level (encoded pair) comparison is reported.\n")

    # ── 5. Potential Pass2 Reduction ──
    lines.append("\n## 5. Potential Pass2 Reduction\n")
    if agg:
        rp50 = agg["reusable_intersection_ratio"]["p50"]
        rp90 = agg["reusable_intersection_ratio"]["p90"]
        rp10 = agg["reusable_intersection_ratio"]["p10"]
        lines.append(f"- **Theoretical max Pass2 reduction:** {rp50:.2%} of intersection records (P50)\n")
        lines.append(f"- Range: P10={rp10:.2%} to P90={rp90:.2%}\n")

        # Estimate records (typical n_isects ~3.25M)
        est_records = 3250000
        lines.append(f"- Baseline records per step: ~{est_records:,}\n")
        lines.append(f"- Candidates for reuse: ~{int(est_records * rp50):,} records (P50)\n")
        lines.append(f"- Records requiring rebuild: ~{int(est_records * (1 - rp50)):,} records (P50)\n")
        lines.append(f"- **Caveat:** These are theoretical upper bounds. Actual speedup depends on ")
        lines.append(f"implementation overhead (invalidation scan, incremental merge, memory management).\n")

    # ── 6. Decision Criteria ──
    lines.append("\n## 6. Decision Criteria Check\n")
    thresholds = result["decision_thresholds"]
    lines.append("| Criterion | Threshold | Measured | Met? |\n")
    lines.append("|:----------|:---------:|:--------:|:----:|\n")
    if agg:
        rp50 = agg["reusable_intersection_ratio"]["p50"]
        cp50 = agg["changed_membership_ratio"]["p50"]
        sp50 = agg["stable_membership_ratio"]["p50"]

        lines.append(f"| Strong GO: changed membership | <{thresholds['strong_go']['changed_membership_ratio_lt']} | {cp50:.4f} | {'✅' if cp50 < 0.10 else '❌'} |\n")
        lines.append(f"| Strong GO: reusable intersect | >{thresholds['strong_go']['reusable_intersection_ratio_gt']} | {rp50:.4f} | {'✅' if rp50 > 0.70 else '❌'} |\n")
        lines.append(f"| Conditional: changed membership | {thresholds['conditional_go']['changed_membership_ratio_range']} | {cp50:.4f} | {'✅' if cp50 < 0.30 else '❌'} |\n")
        lines.append(f"| Conditional: reusable intersect | {thresholds['conditional_go']['reusable_intersection_ratio_range']} | {rp50:.4f} | {'✅' if rp50 > 0.40 else '❌'} |\n")
        if cp50 < 0.30 and rp50 > 0.40:
            lines.append(f"| **CONDITIONAL GO SATISFIED** | both conditions | {cp50:.4f} / {rp50:.4f} | ✅ |\n")
        elif cp50 < 0.10 and rp50 > 0.70:
            lines.append(f"| **STRONG GO SATISFIED** | both conditions | {cp50:.4f} / {rp50:.4f} | ✅ |\n")

    lines.append("\n### Verdict\n")
    lines.append(f"> **{result['decision']}** — {result['decision_detail']}\n")

    if result["decision"] == "NO-GO":
        lines.append("> **Action:** C18 is stopped. Return to optimization-space exploration.\n")
    elif result["decision"] == "GO":
        lines.append("> **Action:** Proceed to CUDA prototype + forward correctness phase.\n")
    elif result["decision"] == "CONDITIONAL GO":
        lines.append("> **Action:** Proceed to CUDA prototype phase, but design must account for ")
        lines.append("partial reuse (not all intersections are reusable). ")
        lines.append("Implementation overhead must be carefully bounded.\n")
    elif result["decision"] == "HIGH RISK":
        lines.append("> **Action:** Further analysis required before prototyping. ")
        lines.append("A strong implementation argument is needed showing that ")
        lines.append("implementation overhead < saved Pass2 cost.\n")

    lines.append("\n---\n")
    lines.append(
        "**Note:** Per-Gaussian membership stability is only measured on no-topology "
        "pairs (topology changes shift Gaussian parameter indices, making index-based "
        "comparison invalid). During densification/pruning, the intersection structure "
        "must be fully rebuilt anyway, so only no-topology reuse matters for the "
        "viability assessment.\n"
    )

    return "".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="C18 Decomposition Gate")
    ap.add_argument("--scene", default="room")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--tile-size", type=int, default=16)
    ap.add_argument("--resolution", default="1080p")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--densify-start", type=int, default=100)
    ap.add_argument("--densify-interval", type=int, default=100)
    ap.add_argument("--prune-start", type=int, default=100)
    ap.add_argument("--prune-interval", type=int, default=100)
    args = ap.parse_args()

    decision = run_decomposition(args)
    code = {"GO": 0, "CONDITIONAL GO": 0, "HIGH RISK": 1, "NO-GO": 2}.get(decision, 1)
    sys.exit(code)
