#!/usr/bin/env python3
"""
C19-1A — Dirty-Run Locality Gate.

Measures the fine structure of depth-ordering violations inside tiles
classified as "requiring repair" by C18-2.

Instead of classifying entire tiles as dirty, this analysis:
  1. Detects exact adjacent inversion edges in prev-order reused sequences
  2. Constructs contiguous dirty runs from connected inversions
  3. Measures run sizes, counts, and separation
  4. Estimates minimum repair regions
  5. Determines whether local (run-level) repair is viable

Output: per-tile data → aggregated metrics → decision (GO/CONDITIONAL/NO-GO)
"""
from __future__ import annotations
import argparse, json, math, random, sys, time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(1, str(ROOT/"src"))
import importlib.util as _iu

# Load scene.py directly (avoid full benchmark_framework chain)
_spec = _iu.spec_from_file_location("_scene_mod", ROOT / "src" / "benchmark_framework" / "scene.py")
_scene_mod = _iu.module_from_spec(_spec)
_spec.loader.exec_module(_scene_mod)
load_ply = _scene_mod.load_ply

def _lp(n):
    p = ROOT/"scripts"/"epic05"/"phase7"/f"{n}.py"
    s = _iu.spec_from_file_location(f"_c{n}", p); m = _iu.module_from_spec(s); s.loader.exec_module(m); return m
_ds = _lp("dataset"); _md = _lp("gaussian_model"); _ls = _lp("loss")
GTData = _ds.GTDataset; GModel = _md.GaussianModel; closs = _ls.combined_loss
from gsplat import rasterization, __version__ as gsv
torch.backends.cudnn.deterministic = True


# ── helpers ──

def _make_output_dirs():
    for d in [ROOT/"results"/"phase-c19", ROOT/"results"/"phase-c19"/"dirty_run", ROOT/"reports"/"phase-c19"]:
        d.mkdir(parents=True, exist_ok=True)

def extract(info, d):
    """Extract sorted intersection state from gsplat info dict."""
    i = info["isect_ids"].long().contiguous()
    f = info["flatten_ids"].long().contiguous()
    g = info["gaussian_ids"].long()
    o = info["isect_offsets"][0].reshape(-1).long()
    n = i.numel(); nt = o.numel()
    sk, si = torch.sort(i); sf = f[si]; sg = g[sf]
    ends = torch.cat((o[1:], o.new_tensor([n])))
    tile_of = torch.repeat_interleave(torch.arange(nt, device=d), ends - o)
    ENC = 1 << 14
    pair = sg * ENC + tile_of
    return {"k": sk, "sf": sf, "sg": sg, "pair": pair, "gids": g,
            "off": o, "tile_of": tile_of, "ends": ends,
            "n": n, "nv": int(g.numel()), "nt": nt}

def fmt_st(d):
    """Return compact stats dict with p50/p25/p75/p90/p95/p99/mean."""
    if not isinstance(d, (list, np.ndarray)) or len(d) == 0:
        return None
    a = np.array(d, dtype=np.float64)
    return {k: float(v) for k, v in zip(
        ["mean","std","p10","p25","p50","p75","p90","p95","p99","min","max","n"],
        [a.mean(), a.std(),
         np.percentile(a,10), np.percentile(a,25), np.percentile(a,50),
         np.percentile(a,75), np.percentile(a,90), np.percentile(a,95),
         np.percentile(a,99), a.min(), a.max(), len(a)])}


# ── Per-tile dirty-run analysis on CPU ──

def analyze_tile_dirty_runs(depths: np.ndarray, n_new: int):
    """
    Analyze a single tile's prev-order reused depths (with curr depth values).

    depths: 1D array of depths in PREV order (reused entries first, then new entries).
            Length = total_entries_in_tile.
    n_new:  number of new entries at the end of depths.

    Returns dict with all per-tile dirty-run metrics, or None if tile is clean.
    """
    n = len(depths)
    if n < 2:
        return None  # trivially sorted

    # Detect adjacent inversions in reused portion only (new entries are
    # trivially sorted among themselves since they come from the same step)
    reused_depths = depths[:n - n_new] if n_new > 0 else depths
    n_reused = len(reused_depths)

    if n_reused < 2:
        return None

    # Adjacent inversion detection
    inversions = np.where(reused_depths[:-1] > reused_depths[1:])[0]
    n_inv = len(inversions)

    if n_inv == 0:
        return None  # clean tile

    # ── Build dirty runs from inversion spans ──
    # Each inversion at position i creates a span [i, i+1] (2 entries)
    # Merge overlapping or touching spans into dirty runs
    # Spans [l1, r1] and [l2, r2] merge if l2 <= r1 (touching/overlapping)

    # Start with each inversion as a span
    spans = list(zip(inversions, inversions + 1))  # (start, end) inclusive
    spans.sort(key=lambda x: x[0])

    # Merge
    runs = []
    cur_l, cur_r = spans[0]
    for l, r in spans[1:]:
        if l <= cur_r + 1:  # touching or overlapping (allow gap of 0)
            cur_r = max(cur_r, r)
        else:
            runs.append((cur_l, cur_r))
            cur_l, cur_r = l, r
    runs.append((cur_l, cur_r))

    run_sizes = [r - l + 1 for l, r in runs]
    n_runs = len(runs)
    dirty_entries = sum(run_sizes)
    largest_run = max(run_sizes)

    # ── Check run independence (Section 12) ──
    # For each dirty run, check if sorting it independently would suffice,
    # or if runs interact across clean boundaries.

    # Test: sort each run independently and check if order violations
    # appear at run boundaries after partial repair
    test_depths = depths.copy()
    for l, r in runs:
        test_depths[l:r+1].sort()

    # Check if partial-sorted array still has inversions at run boundaries
    boundary_violations = 0
    for run_idx in range(n_runs - 1):
        boundary = runs[run_idx][1]  # end of this run
        if boundary + 1 < n_reused:
            # Check if the last entry of this run > first entry of next
            # (after independent sorting of each run)
            if test_depths[boundary] > test_depths[boundary + 1]:
                boundary_violations += 1

    # Also check internal clean regions
    clean_region_violations = 0
    for run_idx in range(n_runs):
        l, r = runs[run_idx]
        # Check left boundary
        if l > 0 and test_depths[l-1] > test_depths[l]:
            clean_region_violations += 1
        # Check right boundary
        if r + 1 < n_reused and test_depths[r] > test_depths[r+1]:
            clean_region_violations += 1

    # ── Minimum repair region (Section 10) ──
    # The minimum span that, when sorted, produces correct ordering.
    # Find the first and last entry that changes position in the full sort.
    sorted_all = np.sort(depths)
    misplaced = np.where(depths != sorted_all)[0]
    if len(misplaced) > 0:
        min_repair_l = int(misplaced[0])
        min_repair_r = int(misplaced[-1])
        min_repair_size = min_repair_r - min_repair_l + 1
    else:
        min_repair_l = runs[0][0]
        min_repair_r = runs[-1][1]
        min_repair_size = min_repair_r - min_repair_l + 1

    # The "expansion ratio" = min_repair_size / sum(run_sizes)
    # This measures how much the minimum repair exceeds the dirty run size
    repair_expansion = min_repair_size / max(sum(run_sizes), 1)

    # The full-tile repair ratio = min_repair_size / n
    # (how much of the tile would need to be touched for a correct repair)
    repair_full_ratio = min_repair_size / n

    # ── Separation between dirty runs ──
    clean_gaps = []
    for i in range(n_runs - 1):
        gap = runs[i+1][0] - runs[i][1] - 1
        if gap > 0:
            clean_gaps.append(gap)
    mean_clean_gap = float(np.mean(clean_gaps)) if clean_gaps else 0.0
    median_clean_gap = float(np.median(clean_gaps)) if clean_gaps else 0.0

    # ── Violation density ──
    violation_density = n_inv / max(n_reused - 1, 1)

    return {
        "n_total": int(n),
        "n_reused": int(n_reused),
        "n_new": int(n_new),
        "n_inversions": int(n_inv),
        "dirty_entry_count": int(dirty_entries),
        "dirty_entry_ratio": float(dirty_entries / n) if n > 0 else 0.0,
        "n_runs": int(n_runs),
        "run_sizes": [int(s) for s in run_sizes],
        "mean_run_size": float(np.mean(run_sizes)),
        "median_run_size": float(np.median(run_sizes)),
        "p90_run_size": float(np.percentile(run_sizes, 90)),
        "p95_run_size": float(np.percentile(run_sizes, 95)),
        "p99_run_size": float(np.percentile(run_sizes, 99)),
        "max_run_size": int(largest_run),
        "largest_run_ratio": float(largest_run / n) if n > 0 else 0.0,
        "violation_density": float(violation_density),
        "boundary_violations": int(boundary_violations),
        "clean_region_violations": int(clean_region_violations),
        "runs_interact": bool(boundary_violations > 0 or clean_region_violations > 0),
        "min_repair_l": int(min_repair_l),
        "min_repair_r": int(min_repair_r),
        "min_repair_size": int(min_repair_size),
        "repair_expansion_ratio": float(repair_expansion),
        "repair_full_ratio": float(repair_full_ratio),
        "n_clean_gaps": int(len(clean_gaps)),
        "mean_clean_gap": float(mean_clean_gap),
        "median_clean_gap": float(median_clean_gap),
        "clean_gaps": [int(g) for g in clean_gaps],
    }


# ── Main analysis runner ──

def run_analysis(args):
    d = torch.device("cuda")
    A100_OK = torch.cuda.is_available() and "A100" in torch.cuda.get_device_name(0)
    print(f"[C19-1A] GPU: {torch.cuda.get_device_name(0)} {'OK' if A100_OK else 'WARN: NOT A100'}")
    if not A100_OK:
        print("[C19-1A] WARNING: Not running on A100; results may differ")

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    print(f"[C19-1A] Loading '{args.scene}'...")
    ds = GTData(args.scene, str(ROOT), resolution=args.resolution, device=d)
    sfm = load_ply(str(ROOT / "data" / "official" / "mipnerf360" / args.scene / "point_cloud.ply"), device=d)
    print(f"[C19-1A] SfM: {sfm['xyz'].shape[0]:,}")
    md = GModel(sfm["xyz"].shape[0], sh_degree=0, max_sh_degree=3, device=d)
    md.init_from_sfm(sfm["xyz"],
                     torch.logit(torch.full((sfm["xyz"].shape[0], 1), 0.1, device=d)),
                     sfm["scales"], sfm["rotations"], sfm["shs"])
    extent = float(sfm["xyz"].norm(dim=-1).max().item())
    ec, _ = ds.get_item(0)
    print(f"[C19-1A] Cam: {ec.image_width}x{ec.image_height}  Tile: {args.tile_size}px")
    ts = args.tile_size

    def make_optim():
        return torch.optim.Adam([
            {"params": [md.xyz], "lr": 1.6e-4 * extent, "eps": 1e-15},
            {"params": [md.rotations], "lr": 1e-3, "eps": 1e-15},
            {"params": [md.scales], "lr": 5e-3, "eps": 1e-15},
            {"params": [md.opacity], "lr": 5e-2, "eps": 1e-15},
            {"params": [md.shs], "lr": 2.5e-3, "eps": 1e-15},
        ])

    optim = make_optim()
    prev_state = None
    warmup = 50
    total = warmup + args.steps

    # Per-step rollups
    step_summaries = []

    # Per-tile data collector (across all measurement steps)
    all_tile_metrics = []  # list of dicts, each with tile_id, step, n_total, and dirty metrics
    all_step_aggregates = []

    t0 = time.perf_counter()
    print(f"[C19-1A] Running {total} steps ({warmup} warmup, {args.steps} measure)...")

    for st in range(total):
        # Training step
        tci = st % len(ds)
        tc, ttg = ds.get_item(tci)
        _d = md.forward()
        ti, _, _ = rasterization(
            means=_d["xyz"], quats=_d["rotations"], scales=_d["scales"],
            opacities=_d["opacity"], colors=_d["shs"],
            viewmats=tc.viewmatrix.unsqueeze(0), Ks=tc.K.unsqueeze(0),
            width=tc.image_width, height=tc.image_height,
            tile_size=ts, packed=True, sh_degree=md.sh_degree,
            radius_clip=0., eps2d=0.1, render_mode="RGB")
        loss = closs(ti[0].clamp(0, 1), ttg, lambda_dssim=0.2)["loss"]
        optim.zero_grad(set_to_none=True)
        loss.backward()
        md.accumulate_positional_gradient()
        optim.step()

        # Topology
        dd = dp = False
        if st >= args.densify_start and st < 15000 and st % args.densify_interval == 0:
            e = md.densification(grad_threshold=2e-4, clone_max_screen_size=100., split_max_screen_size=100.)
            if e["cloned"] + e["split"] > 0: dd = True
        if st >= args.prune_start and st % args.prune_interval == 0:
            if md.prune_and_reset(opacity_threshold=0.005, reset_interval=3000, current_step=st) > 0: dp = True
        topo = dd or dp
        if topo:
            optim = make_optim()

        # Eval on fixed camera
        de = md.forward()
        with torch.no_grad():
            _, _, ei = rasterization(
                means=de["xyz"], quats=de["rotations"], scales=de["scales"],
                opacities=de["opacity"], colors=de["shs"],
                viewmats=ec.viewmatrix.unsqueeze(0), Ks=ec.K.unsqueeze(0),
                width=ec.image_width, height=ec.image_height,
                tile_size=ts, packed=True, sh_degree=md.sh_degree,
                radius_clip=0., eps2d=0.1, render_mode="RGB")

        curr = extract(ei, d)

        # ── Measurement ──
        if st >= warmup and prev_state is not None and not topo:
            # — 1. Match (gid,tile) pairs —
            pp, po = torch.sort(prev_state["pair"])
            cp, co = torch.sort(curr["pair"])
            pos = torch.searchsorted(pp, cp)
            ir = pos < prev_state["n"]
            mc = torch.zeros(curr["n"], dtype=torch.bool, device=d)
            mc[ir] = pp[pos[ir]] == cp[ir]

            # Map to entry order
            match_entry = torch.zeros(curr["n"], dtype=torch.bool, device=d)
            match_entry[co] = mc

            # Map each reused pair to its prev index
            prev_idx = torch.full((curr["n"],), -1, dtype=torch.long, device=d)
            mc_nz = mc.nonzero(as_tuple=False).flatten()
            if mc_nz.numel() > 0:
                prev_idx[co[mc_nz]] = po[pos[mc_nz]]

            # Current depths (lower 32 bits of sort key, interpreted as int32)
            c_depth = (curr["k"] & 0xFFFFFFFF).int()

            # Move to CPU for per-tile analysis
            off_c = curr["off"].cpu()
            ends_c = curr["ends"].cpu()
            match_cpu = match_entry.cpu()
            c_depth_cpu = c_depth.cpu()
            prev_idx_cpu = prev_idx.cpu()
            tids_cpu = curr["tile_of"].cpu()
            nt = curr["nt"]

            # Per-tile data for this step
            step_tile_data = []
            n_repair = 0
            n_violating_entries = 0
            total_dirty_entries = 0
            total_dirty_entry_count = 0
            global_dirty_entry_sum = 0

            for ti in range(nt):
                s = int(off_c[ti].item())
                e = int(ends_c[ti].item())
                nti = e - s
                if nti < 1:
                    continue

                # Get reused entries indices within this tile (curr sorted order)
                tile_entries = torch.arange(s, e, dtype=torch.long)
                rmsk = match_cpu[s:e]
                reused_entries = tile_entries[rmsk]
                new_entries = tile_entries[~rmsk]
                nr_t = len(reused_entries)
                nn_t = len(new_entries)

                # Build prev-order sequence of depths for reused entries
                if nr_t >= 1:
                    p_pos = prev_idx_cpu[reused_entries].numpy()
                    # Sort reused entries by prev position → prev order
                    prev_order = np.argsort(p_pos)
                    reused_in_prev_order = reused_entries[prev_order].numpy()
                    # Depths in prev order (with CURR depth values)
                    prev_order_depths = c_depth_cpu[reused_in_prev_order].numpy()
                else:
                    prev_order_depths = np.array([], dtype=np.int32)

                # Append new entry depths at the end
                if nn_t > 0:
                    new_depths = c_depth_cpu[new_entries].numpy()
                    all_depths = np.concatenate([prev_order_depths, new_depths])
                else:
                    all_depths = prev_order_depths

                # Run dirty-run analysis
                tile_result = analyze_tile_dirty_runs(all_depths, nn_t)

                entry = {
                    "step": int(st),
                    "tile_id": int(ti),
                    "n_total": int(nti),
                    "n_reused": int(nr_t),
                    "n_new": int(nn_t),
                    "is_repair": tile_result is not None,
                }

                if tile_result is not None:
                    entry.update(tile_result)
                    n_repair += 1
                    total_dirty_entries += tile_result["dirty_entry_count"]
                    global_dirty_entry_sum += tile_result["dirty_entry_count"]
                    n_violating_entries += nti
                else:
                    entry.update({
                        "n_inversions": 0, "dirty_entry_count": 0, "dirty_entry_ratio": 0.0,
                        "n_runs": 0, "run_sizes": [],
                        "mean_run_size": 0.0, "median_run_size": 0.0, "max_run_size": 0,
                        "largest_run_ratio": 0.0, "violation_density": 0.0,
                        "boundary_violations": 0, "clean_region_violations": 0,
                        "runs_interact": False,
                        "min_repair_size": 0, "repair_expansion_ratio": 1.0,
                        "repair_full_ratio": 0.0,
                        "mean_clean_gap": 0.0, "median_clean_gap": 0.0,
                    })

                step_tile_data.append(entry)
                all_tile_metrics.append(entry)

            # Compute step-level aggregates
            total_entries = sum(t["n_total"] for t in step_tile_data) if step_tile_data else 1
            repair_tiles = [t for t in step_tile_data if t["is_repair"]]
            global_dirty_ratio = global_dirty_entry_sum / total_entries if total_entries > 0 else 0.0
            repair_dirty_ratios = [t["dirty_entry_ratio"] for t in repair_tiles] if repair_tiles else [0.0]
            repair_violation_densities = [t["violation_density"] for t in repair_tiles] if repair_tiles else [0.0]
            repair_run_counts = [t["n_runs"] for t in repair_tiles] if repair_tiles else [0]

            step_agg = {
                "step": int(st),
                "n_tiles": nt,
                "n_repair_tiles": n_repair,
                "repair_tile_ratio": n_repair / nt if nt > 0 else 0.0,
                "total_entries": int(total_entries),
                "n_violating_entries": int(n_violating_entries),
                "global_dirty_entry_ratio": float(global_dirty_ratio),
                "repair_dirty_entry_ratio_mean": float(np.mean(repair_dirty_ratios)) if repair_dirty_ratios else 0.0,
                "repair_dirty_entry_ratio_p50": float(np.median(repair_dirty_ratios)) if repair_dirty_ratios else 0.0,
                "repair_violation_density_p50": float(np.median(repair_violation_densities)) if repair_violation_densities else 0.0,
                "repair_run_count_p50": float(np.median(repair_run_counts)) if repair_run_counts else 0.0,
                "mean_dirty_run_count": float(np.mean(repair_run_counts)) if repair_run_counts else 0.0,
            }
            step_summaries.append(step_agg)

        prev_state = curr

        if st % 25 == 0 or st == total - 1:
            print(f"[{st:04d}] N={md.xyz.shape[0]:,}  isects={curr['n']:,}  topo={topo}", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"\n[C19-1A] Completed in {elapsed:.0f}s. Collected {len(all_tile_metrics)} tile-step records.")

    # ── Aggregate analysis ──
    return build_aggregate_results(all_tile_metrics, step_summaries, args, elapsed)


def build_aggregate_results(all_tile_metrics, step_summaries, args, elapsed):
    """Build the complete aggregate analysis from per-tile data."""

    if not all_tile_metrics:
        return {"error": "no tile data collected"}, {"error": "no data"}

    # Separate repair tiles and all tiles
    repair_tiles = [t for t in all_tile_metrics if t["is_repair"]]
    non_repair_tiles = [t for t in all_tile_metrics if not t["is_repair"]]

    n_total_tiles = len(all_tile_metrics)
    n_repair = len(repair_tiles)

    # ── Section 5: Aggregate distributions ──

    def extract(seq, key):
        vals = [t.get(key, 0) for t in seq]
        return fmt_st(vals)

    agg = {}

    # All repair tiles
    agg["repair_tiles"] = {
        "count": n_repair,
        "total_entries": extract(repair_tiles, "n_total"),
        "dirty_entry_ratio": extract(repair_tiles, "dirty_entry_ratio"),
        "n_runs": extract(repair_tiles, "n_runs"),
        "mean_run_size": extract(repair_tiles, "mean_run_size"),
        "largest_run_ratio": extract(repair_tiles, "largest_run_ratio"),
        "violation_density": extract(repair_tiles, "violation_density"),
        "repair_expansion_ratio": extract(repair_tiles, "repair_expansion_ratio"),
        "repair_full_ratio": extract(repair_tiles, "repair_full_ratio"),
        "run_sizes_flat": extract([{"s": s} for t in repair_tiles for s in t.get("run_sizes", [])], "s"),
    }

    # All tiles (including clean)
    agg["all_tiles"] = {
        "count": n_total_tiles,
        "total_entries": extract(all_tile_metrics, "n_total"),
        "dirty_entry_ratio": extract(all_tile_metrics, "dirty_entry_ratio"),
        "n_runs": extract(all_tile_metrics, "n_runs"),
        "largest_run_ratio": extract(all_tile_metrics, "largest_run_ratio"),
        "repair_full_ratio": extract(all_tile_metrics, "repair_full_ratio"),
    }

    # ── Section 6: Distribution by tile workload ──
    buckets = [
        (1, 32), (33, 64), (65, 128), (129, 256),
        (257, 512), (513, 1024), (1025, 2048), (2049, 4096), (4097, 10**9)
    ]
    tile_size_data = all_tile_metrics if not args.measurement_only else all_tile_metrics
    bucket_results = []
    for lo, hi in buckets:
        bucket_tiles = [t for t in tile_size_data if lo <= t["n_total"] <= hi]
        if not bucket_tiles:
            continue
        bucket_repair = [t for t in bucket_tiles if t["is_repair"]]
        label = f"{lo}-{hi}" if hi < 10**9 else f">{lo-1}"
        br = {
            "bucket": label,
            "tile_count": len(bucket_tiles),
            "mean_entries": float(np.mean([t["n_total"] for t in bucket_tiles])),
            "repair_ratio": len(bucket_repair) / len(bucket_tiles) if bucket_tiles else 0.0,
            "dirty_entry_ratio_mean": float(np.mean([t.get("dirty_entry_ratio", 0) for t in bucket_tiles])),
            "dirty_entry_ratio_p50": float(np.median([t.get("dirty_entry_ratio", 0) for t in bucket_tiles])),
            "mean_run_count": float(np.mean([t.get("n_runs", 0) for t in bucket_tiles])),
            "mean_run_size": float(np.mean([t.get("mean_run_size", 0) for t in bucket_tiles])),
            "largest_run_ratio_mean": float(np.mean([t.get("largest_run_ratio", 0) for t in bucket_tiles])),
            "largest_run_ratio_p50": float(np.median([t.get("largest_run_ratio", 0) for t in bucket_tiles])),
        }
        bucket_results.append(br)
    agg["tile_size_buckets"] = bucket_results

    # ── Section 7: Three quantities ──
    repair_tile_ratio = n_repair / n_total_tiles if n_total_tiles > 0 else 0.0

    all_dirty_entries = sum(t.get("dirty_entry_count", 0) for t in all_tile_metrics)
    all_entries = sum(t["n_total"] for t in all_tile_metrics)
    global_dirty_entry_ratio = all_dirty_entries / all_entries if all_entries > 0 else 0.0

    repair_dirty_entries = sum(t.get("dirty_entry_count", 0) for t in repair_tiles)
    repair_entries = sum(t["n_total"] for t in repair_tiles)
    repair_dirty_entry_ratio = repair_dirty_entries / repair_entries if repair_entries > 0 else 0.0

    agg["three_quantities"] = {
        "repair_tile_ratio": float(repair_tile_ratio),
        "dirty_entry_ratio_in_repair_tiles": float(repair_dirty_entry_ratio),
        "global_dirty_entry_ratio": float(global_dirty_entry_ratio),
        "c18_2_repair_tile_ratio": 0.8240,
        "c18_2_repair_entry_ratio": 0.9112,
    }

    # ── Section 8: Violation density ──
    agg["violation_density"] = extract(repair_tiles, "violation_density")

    # ── Section 9: Clean gaps ──
    all_gaps = [g for t in repair_tiles for g in t.get("clean_gaps", [])]
    if all_gaps:
        agg["clean_gap_sizes"] = fmt_st(all_gaps)
    else:
        agg["clean_gap_sizes"] = {"note": "no multi-run tiles with clean gaps"}

    # ── Section 10: Minimum repair region vs full tile ──
    agg["repair_expansion"] = extract(repair_tiles, "repair_expansion_ratio")
    agg["repair_full_ratio"] = extract(repair_tiles, "repair_full_ratio")

    # ── Section 11: Theoretical sort reduction ──
    # baseline_repair_sort_entries = sum of all entries in repair tiles (full sort)
    baseline_repair_sort_entries = sum(t["n_total"] for t in repair_tiles)

    # run_repair_sort_entries = sum of dirty_entry_count (sort only dirty runs)
    # BUT: a run-level sort might need to sort entries equal to min_repair_size
    # (the minimum span that needs sorting for correct result)
    run_repair_sort_entries = sum(t.get("min_repair_size", 0) for t in repair_tiles)
    # Also compute using just dirty runs (simpler, possibly insufficient)
    dirty_run_sort_entries = sum(t.get("dirty_entry_count", 0) for t in repair_tiles)

    agg["theoretical_sort_reduction"] = {
        "baseline_repair_sort_entries": int(baseline_repair_sort_entries),
        "run_repair_sort_entries": int(run_repair_sort_entries),
        "dirty_run_sort_entries": int(dirty_run_sort_entries),
        "reduction_vs_baseline_min_repair": float(
            1.0 - run_repair_sort_entries / max(baseline_repair_sort_entries, 1)),
        "reduction_vs_baseline_dirty_only": float(
            1.0 - dirty_run_sort_entries / max(baseline_repair_sort_entries, 1)),
    }

    # ── Section 12: Run interaction ──
    interacting_tiles = [t for t in repair_tiles if t.get("runs_interact", False)]
    agg["run_interaction"] = {
        "total_repair_tiles": n_repair,
        "tiles_with_interacting_runs": len(interacting_tiles),
        "interaction_ratio": len(interacting_tiles) / n_repair if n_repair > 0 else 0.0,
        "boundary_violations_total": sum(t.get("boundary_violations", 0) for t in repair_tiles),
        "clean_region_violations_total": sum(t.get("clean_region_violations", 0) for t in repair_tiles),
    }

    # ── Section 13: Training phase analysis ──
    # With 100 measurement steps (50-149), split into early/middle/late
    steps = sorted(set(t["step"] for t in all_tile_metrics))
    n_steps = len(steps)
    if n_steps >= 3:
        split1 = steps[n_steps // 3]
        split2 = steps[2 * n_steps // 3]
        phases = {"early": (steps[0], split1), "middle": (split1, split2), "late": (split2, steps[-1])}
    else:
        phases = {"all": (steps[0], steps[-1])}

    phase_results = {}
    for phase_name, (lo, hi) in phases.items():
        phase_tiles = [t for t in all_tile_metrics if lo <= t["step"] <= hi]
        if not phase_tiles:
            continue
        phase_repair = [t for t in phase_tiles if t["is_repair"]]
        phase_dirty = sum(t.get("dirty_entry_count", 0) for t in phase_tiles)
        phase_entries = sum(t["n_total"] for t in phase_tiles)
        phase_results[phase_name] = {
            "step_range": [int(lo), int(hi)],
            "tile_count": len(phase_tiles),
            "repair_tile_ratio": len(phase_repair) / len(phase_tiles) if phase_tiles else 0.0,
            "global_dirty_entry_ratio": float(phase_dirty / phase_entries) if phase_entries > 0 else 0.0,
            "mean_run_size_in_repair": float(np.mean([t.get("mean_run_size", 0) for t in phase_repair])) if phase_repair else 0.0,
            "largest_run_ratio_mean": float(np.mean([t.get("largest_run_ratio", 0) for t in phase_repair])) if phase_repair else 0.0,
        }
    agg["training_phases"] = phase_results

    # ── Section 14: Stress analysis — Top 20 worst tiles ──
    sorted_by_dirty_ratio = sorted(repair_tiles, key=lambda t: t.get("dirty_entry_ratio", 0), reverse=True)
    sorted_by_largest_run_ratio = sorted(repair_tiles, key=lambda t: t.get("largest_run_ratio", 0), reverse=True)

    def tile_summary(t):
        return {
            "step": t["step"], "tile_id": t["tile_id"],
            "n_total": t["n_total"], "n_inversions": t.get("n_inversions", 0),
            "n_runs": t.get("n_runs", 0), "dirty_entry_count": t.get("dirty_entry_count", 0),
            "dirty_entry_ratio": t.get("dirty_entry_ratio", 0),
            "max_run_size": t.get("max_run_size", 0),
            "largest_run_ratio": t.get("largest_run_ratio", 0),
            "violation_density": t.get("violation_density", 0),
            "repair_expansion_ratio": t.get("repair_expansion_ratio", 1.0),
        }

    agg["stress_analysis"] = {
        "top20_by_dirty_entry_ratio": [tile_summary(t) for t in sorted_by_dirty_ratio[:20]],
        "top20_by_largest_run_ratio": [tile_summary(t) for t in sorted_by_largest_run_ratio[:20]],
    }

    # ── Step-level aggregates for reference ──
    agg["step_summaries"] = step_summaries

    # ── Meta ──
    agg["meta"] = {
        "title": "C19-1A Dirty-Run Locality Gate",
        "date": datetime.now(timezone.utc).isoformat(),
        "scene": args.scene, "steps": args.steps, "warmup": 50,
        "tile_size": args.tile_size, "resolution": args.resolution,
        "seed": args.seed,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "gsplat_version": str(gsv),
        "duration_s": elapsed,
    }

    # ── Decision ──
    gder = agg["three_quantities"]["global_dirty_entry_ratio"]
    p50_der = agg["repair_tiles"]["dirty_entry_ratio"]["p50"]
    p95_der = agg["repair_tiles"]["dirty_entry_ratio"]["p95"]
    p99_der = agg["repair_tiles"]["dirty_entry_ratio"]["p99"]
    p95_run_ratio = agg["repair_tiles"]["largest_run_ratio"]["p95"]
    p99_run_ratio = agg["repair_tiles"]["largest_run_ratio"]["p99"]

    # STRONG GO: global DER < 5%, P50 DER < 5%, P95 DER < 20%
    if gder < 0.05 and p50_der < 0.05 and p95_der < 0.20:
        decision = "RUN-LEVEL GO"
        decision_detail = f"global_dirty_entry_ratio={gder:.4f} (<5%), P50={p50_der:.4f} (<5%), P95={p95_der:.4f} (<20%)"
    # CONDITIONAL: global DER 5-15% or P95 DER 20-40%
    elif (0.05 <= gder <= 0.15) or (0.20 <= p95_der <= 0.40):
        decision = "RUN-LEVEL CONDITIONAL"
        decision_detail = f"global_dirty_entry_ratio={gder:.4f}, P50={p50_der:.4f}, P95={p95_der:.4f}, P99_run_ratio={p99_run_ratio:.4f}"
    # NO-GO: global DER > 30%, or P50 DER > 20%, or P95/P99 run ratios approach full tile
    else:
        decision = "RUN-LEVEL NO-GO"
        decision_detail = f"global_dirty_entry_ratio={gder:.4f}, P50={p50_der:.4f}, P95={p95_der:.4f}, P99_run_ratio={p99_run_ratio:.4f}"

    agg["decision"] = decision
    agg["decision_detail"] = decision_detail

    print(f"\n{'='*70}")
    print(f"  C19-1A DECISION: {decision}")
    print(f"  {decision_detail}")
    print(f"  Global DER={gder:.4f}  P50 DER={p50_der:.4f}  P95 DER={p95_der:.4f}")
    print(f"  Repair tiles: {n_repair}/{n_total_tiles} ({repair_tile_ratio:.4f})")
    print(f"{'='*70}")

    return agg, all_tile_metrics


# ── Report writer ──

def write_report(agg, path: Path):
    m = agg["meta"]
    dec = agg["decision"]
    det = agg["decision_detail"]
    tq = agg["three_quantities"]

    lines = [
        f"# C19-1A — Dirty-Run Locality Gate",
        f"",
        f"**Scene:** `{m['scene']}`  **Steps:** {m['steps']}+{m['warmup']}  **GPU:** {m['gpu']}",
        f"**gsplat:** {m['gsplat_version']}  **Date:** {m['date']}  **Duration:** {m['duration_s']:.0f}s",
        f"",
        f"---",
        f"",
        f"## Decision: **{dec}**",
        f"",
        f"{det}",
        f"",
        f"---",
        f"",
        f"## 1. Method",
        f"",
        f"For each consecutive pair (t, t+1) across {m['steps']} measurement steps:",
        f"",
        f"1. Identify (Gaussian, Tile) pairs common to both steps via `searchsorted`",
        f"2. For each tile: reconstruct reused entries in their PREV sorted order with **CURR depth values**",
        f"3. Detect adjacent inversions: `d[i] > d[i+1]` in the prev-order sequence",
        f"4. Build dirty runs: merge connected inversion spans `[i, i+1]` into contiguous dirty regions",
        f"5. Measure: run sizes, counts, separation, minimum repair region, expansion ratio",
        f"6. Check run independence: sort each run independently and verify no new cross-boundary violations",
        f"",
        f"---",
        f"",
        f"## 2. Correctness",
        f"",
        f"| Check | Value |",
        f"|:---|---:|",
        f"| Tiles analyzed | {agg['all_tiles']['count']:,} |",
        f"| Repair tiles detected | {agg['repair_tiles']['count']:,} |",
        f"| Non-repair tiles | {agg['all_tiles']['count'] - agg['repair_tiles']['count']:,} |",
        f"",
        f"---",
        f"",
        f"## 3. Three Quantities (Section 7)",
        f"",
        f"| Quantity | Value | Interpretation |",
        f"|:---|---:|:---|",
        f"| Repair tile ratio | {tq['repair_tile_ratio']:.4f} | Fraction of tiles with any inversion |",
        f"| Dirty entry ratio (repair tiles) | {tq['dirty_entry_ratio_in_repair_tiles']:.4f} | Fraction of entries in repair tiles that are dirty |",
        f"| **Global dirty entry ratio** | **{tq['global_dirty_entry_ratio']:.4f}** | **Fraction of ALL intersection entries in dirty runs** |",
        f"| C18-2 repair tile ratio | {tq['c18_2_repair_tile_ratio']:.4f} | (previous conservative metric) |",
        f"| C18-2 repair entry ratio | {tq['c18_2_repair_entry_ratio']:.4f} | (previous conservative metric) |",
        f"",
        f"The global dirty entry ratio is the critical number: it shows how many entries truly need re-sorting",
        f"under a run-level repair strategy, vs the C18-2 assumption that entire tiles need re-sorting.",
    ]

    # ── Repair tile distributions ──
    def table_row(label, st, fmt=".4f"):
        # fmt is the format spec WITHOUT leading colon, e.g. ".4f" not ":.4f"
        def ff(v):
            if isinstance(v, str):
                return float(v) if v.replace('.','',1).replace('-','',1).isdigit() else 0.0
            return float(v)
        if st is None:
            return f"| {label} | — | — | — | — | — | — | — | — |"
        return (f"| {label} | {ff(st.get('mean',0)):{fmt}} | {ff(st.get('p10',0)):{fmt}} | "
                f"{ff(st.get('p25',0)):{fmt}} | {ff(st.get('p50',0)):{fmt}} | "
                f"{ff(st.get('p75',0)):{fmt}} | {ff(st.get('p90',0)):{fmt}} | "
                f"{ff(st.get('p95',0)):{fmt}} | {ff(st.get('p99',0)):{fmt}} |")

    lines += [
        f"",
        f"---",
        f"",
        f"## 4. Aggregate Distributions — Repair Tiles",
        f"",
        f"| Metric | Mean | P10 | P25 | P50 | P75 | P90 | P95 | P99 |",
        f"|:---|---:|---:|---:|---:|---:|---:|---:|---:|",
        table_row("Dirty entry ratio", agg["repair_tiles"]["dirty_entry_ratio"]),
        table_row("Run count", agg["repair_tiles"]["n_runs"]),
        table_row("Mean run size", agg["repair_tiles"]["mean_run_size"]),
        table_row("Largest run ratio", agg["repair_tiles"]["largest_run_ratio"]),
        table_row("Violation density", agg["repair_tiles"]["violation_density"]),
        table_row("Repair expansion ratio", agg["repair_tiles"]["repair_expansion_ratio"]),
        table_row("Repair / full tile ratio", agg["repair_tiles"]["repair_full_ratio"]),
    ]

    # ── All tiles ──
    lines += [
        f"",
        f"---",
        f"",
        f"## 5. Aggregate Distributions — All Tiles (including clean)",
        f"",
        f"| Metric | Mean | P10 | P25 | P50 | P75 | P90 | P95 | P99 |",
        f"|:---|---:|---:|---:|---:|---:|---:|---:|---:|",
        table_row("Dirty entry ratio", agg["all_tiles"]["dirty_entry_ratio"]),
        table_row("Largest run ratio", agg["all_tiles"]["largest_run_ratio"]),
    ]

    # ── Violation density ──
    lines += [
        f"",
        f"---",
        f"",
        f"## 6. Violation Density (Section 8)",
        f"",
        f"Fraction of adjacent pairs in the reused sequence that are inversions.",
        f"",
        f"| Mean | P10 | P25 | P50 | P75 | P90 | P95 | P99 |",
        f"|---:|---:|---:|---:|---:|---:|---:|---:|",
        table_row("Violation density", agg["violation_density"]),
        f"",
        f"**Answer**: Are the repair tiles caused by many tiny violations or dense disorder?",
    ]
    vd = agg["violation_density"]
    if vd:
        vd_p50 = vd.get("p50", 0)
        vd_mean = vd.get("mean", 0)
        if vd_p50 < 0.01:
            lines.append(f"P50={vd_p50:.6f} — **Very sparse** — most repair tiles have tiny isolated inversions.")
        elif vd_p50 < 0.05:
            lines.append(f"P50={vd_p50:.6f} — **Sparse** — most repair tiles have a few isolated inversions.")
        elif vd_p50 < 0.15:
            lines.append(f"P50={vd_p50:.6f} — **Moderate** — noticeable but not dominant.")
        else:
            lines.append(f"P50={vd_p50:.6f} — **Dense** — substantial disorder in repaired tiles.")

    # ── Buckets ──
    lines += [
        f"",
        f"---",
        f"",
        f"## 7. Distribution by Tile Workload (Section 6)",
        f"",
        f"| Bucket | Tiles | Mean Entries | Repair Ratio | DER Mean | DER P50 | Runs | Run Size | LRR Mean | LRR P50 |",
        f"|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for br in agg.get("tile_size_buckets", []):
        lines.append(
            f"| {br['bucket']} | {br['tile_count']:,} | {br['mean_entries']:.0f} | "
            f"{br['repair_ratio']:.4f} | {br['dirty_entry_ratio_mean']:.4f} | {br['dirty_entry_ratio_p50']:.4f} | "
            f"{br['mean_run_count']:.2f} | {br['mean_run_size']:.2f} | "
            f"{br['largest_run_ratio_mean']:.4f} | {br['largest_run_ratio_p50']:.4f} |"
        )

    # ── Run interaction ──
    ri = agg.get("run_interaction", {})
    lines += [
        f"",
        f"---",
        f"",
        f"## 8. Run Interaction / Correctness Check (Section 12)",
        f"",
        f"| Metric | Value |",
        f"|:---|---:|",
        f"| Repair tiles checked | {ri.get('total_repair_tiles', 0):,} |",
        f"| Tiles with interacting runs | {ri.get('tiles_with_interacting_runs', 0):,} |",
        f"| Interaction ratio | {ri.get('interaction_ratio', 0):.6f} |",
        f"| Total boundary violations | {ri.get('boundary_violations_total', 0)} |",
        f"| Total clean-region violations | {ri.get('clean_region_violations_total', 0)} |",
        f"",
    ]
    if ri.get("interaction_ratio", 1) > 0.001:
        lines.append("⚠️ Runs interact non-trivially — independent run sorting alone may be insufficient.")
        lines.append("   The minimum repair region (expansion ratio) accounts for this.")
    else:
        lines.append("✅ Runs are effectively independent — sorting dirty runs independently suffices.")

    # ── Theoretical reduction ──
    tsr = agg.get("theoretical_sort_reduction", {})
    lines += [
        f"",
        f"---",
        f"",
        f"## 9. Theoretical Sort Reduction (Section 11)",
        f"",
        f"| Strategy | Sort volume (entries) | Reduction vs baseline |",
        f"|:---|---:|---:|",
        f"| Baseline (full repair tile sort) | {tsr.get('baseline_repair_sort_entries', 0):,} | — |",
        f"| Run-level (min repair region sort) | {tsr.get('run_repair_sort_entries', 0):,} | "
        f"{tsr.get('reduction_vs_baseline_min_repair', 0)*100:.2f}% |",
        f"| Run-level (dirty-only sort) | {tsr.get('dirty_run_sort_entries', 0):,} | "
        f"{tsr.get('reduction_vs_baseline_dirty_only', 0)*100:.2f}% |",
    ]

    # ── Clean gaps ──
    lines += [
        f"",
        f"---",
        f"",
        f"## 10. Separation Between Dirty Runs (Section 9)",
    ]
    cgs = agg.get("clean_gap_sizes", {})
    if isinstance(cgs, dict) and "mean" in cgs:
        lines += [
            f"| Mean | P10 | P25 | P50 | P75 | P90 | P95 | P99 |",
            f"|---:|---:|---:|---:|---:|---:|---:|---:|",
            table_row("Clean gap sizes", cgs),
        ]
        if cgs.get("p50", 0) > 2:
            lines.append(f"Clean gaps are substantial (P50={cgs.get('p50',0):.1f} entries) "
                         "— dirty runs are well-separated.")
        else:
            lines.append(f"Clean gaps are small (P50={cgs.get('p50',0):.1f} entries) "
                         "— dirty runs are close together.")
    else:
        lines.append("No multi-run tiles with clean gaps detected (most repair tiles have 1 run).")

    # ── Training phase ──
    lines += [
        f"",
        f"---",
        f"",
        f"## 11. Training Phase Analysis (Section 13)",
        f"",
        f"| Phase | Steps | Repair Ratio | Global DER | Mean Run Size | Largest Run Ratio |",
        f"|:---|---:|---:|---:|---:|---:|",
    ]
    for phase, pr in agg.get("training_phases", {}).items():
        lines.append(
            f"| {phase} | {pr['step_range'][0]}-{pr['step_range'][1]} | "
            f"{pr['repair_tile_ratio']:.4f} | {pr['global_dirty_entry_ratio']:.4f} | "
            f"{pr['mean_run_size_in_repair']:.2f} | {pr['largest_run_ratio_mean']:.4f} |"
        )

    # ── Stress analysis ──
    lines += [
        f"",
        f"---",
        f"",
        f"## 12. Stress Analysis — Top 20 Worst Tiles (Section 14)",
        f"",
        f"### By Dirty Entry Ratio",
        f"",
        f"| # | Step | Tile | Entries | Inversions | Runs | Dirty | DER | Max Run | LRR | Viol Density |",
        f"|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    top20_der = agg["stress_analysis"]["top20_by_dirty_entry_ratio"]
    for i, t in enumerate(top20_der):
        lines.append(
            f"| {i+1} | {t['step']} | {t['tile_id']:04d} | {t['n_total']} | "
            f"{t['n_inversions']} | {t['n_runs']} | {t['dirty_entry_count']} | "
            f"{t['dirty_entry_ratio']:.4f} | {t['max_run_size']} | {t['largest_run_ratio']:.4f} | "
            f"{t['violation_density']:.4f} |"
        )

    lines += [
        f"",
        f"### By Largest Run Ratio",
        f"",
        f"| # | Step | Tile | Entries | Inversions | Runs | Dirty | DER | Max Run | LRR | Viol Density |",
        f"|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    top20_lrr = agg["stress_analysis"]["top20_by_largest_run_ratio"]
    for i, t in enumerate(top20_lrr):
        lines.append(
            f"| {i+1} | {t['step']} | {t['tile_id']:04d} | {t['n_total']} | "
            f"{t['n_inversions']} | {t['n_runs']} | {t['dirty_entry_count']} | "
            f"{t['dirty_entry_ratio']:.4f} | {t['max_run_size']} | {t['largest_run_ratio']:.4f} | "
            f"{t['violation_density']:.4f} |"
        )

    # ── Decision ──
    lines += [
        f"",
        f"---",
        f"",
        f"## Decision: **{dec}**",
        f"",
        f"{det}",
        f"",
        f"---",
        f"",
    ]

    # Interpretation
    gder = tq["global_dirty_entry_ratio"]
    p50_der = agg["repair_tiles"]["dirty_entry_ratio"]["p50"]
    p95_der = agg["repair_tiles"]["dirty_entry_ratio"]["p95"]
    p99_der = agg["repair_tiles"]["dirty_entry_ratio"]["p99"]

    lines += [
        f"### Interpretation",
        f"",
        f"- **Global dirty entry ratio**: {gder:.4f} ({gder*100:.2f}%) of ALL intersection entries",
        f"  are part of dirty runs. This is the critical metric for work reduction.",
        f"- **P50 dirty entry ratio in repair tiles**: {p50_der:.4f} ({p50_der*100:.2f}%) — ",
        f"  half of repair tiles have at most {p50_der*100:.1f}% of their entries dirty.",
        f"- **P95 dirty entry ratio**: {p95_der:.4f} ({p95_der*100:.2f}%) — "
        f"only 5% of repair tiles exceed this.",
        f"- **P99 dirty entry ratio**: {p99_der:.4f} ({p99_der*100:.2f}%) — pathological cases.",
        f"",
        f"### Situation A vs Situation B",
    ]
    if gder < 0.10 and p50_der < 0.15:
        lines.append(
            f"**Situation A** 🟢 — 82.4% of tiles need repair, but dirty regions are SMALL and LOCAL.\n"
            f"C18-2 failed because its repair policy was too coarse. Run-level repair is viable."
        )
    elif gder > 0.30 or p50_der > 0.30:
        lines.append(
            f"**Situation B** 🔴 — 82.4% of tiles need repair AND dirty regions are LARGE.\n"
            f"Run-level repair ≈ full sort. Candidate should be killed."
        )
    else:
        lines.append(
            f"**Mixed** 🟡 — Moderate locality. Further microbenchmark needed to determine "
            f"whether the savings justify implementation complexity."
        )

    lines.append("")
    path.write_text("\n".join(lines))
    print(f"[C19-1A] Report written: {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="room")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--tile-size", type=int, default=16)
    p.add_argument("--resolution", default="1080p")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--densify-start", type=int, default=300)
    p.add_argument("--densify-interval", type=int, default=100)
    p.add_argument("--prune-start", type=int, default=300)
    p.add_argument("--prune-interval", type=int, default=100)
    p.add_argument("--measurement-only", action="store_true",
                   help="Only analyze measurement steps (50-149)")
    args = p.parse_args()

    _make_output_dirs()

    print(f"{'='*70}")
    print(f"  C19-1A — Dirty-Run Locality Gate")
    print(f"  Scene: {args.scene}  Steps: {args.steps}  Tile: {args.tile_size}")
    print(f"  Resolution: {args.resolution}  Seed: {args.seed}")
    print(f"{'='*70}")

    agg, per_tile = run_analysis(args)

    if "error" in agg:
        print(f"[C19-1A] FATAL: {agg['error']}")
        sys.exit(1)

    # Save results
    json_path = ROOT / "results" / "phase-c19" / "c19-1a_dirty_run_locality.json"
    json_path.write_text(json.dumps(agg, indent=2, default=str))
    print(f"[C19-1A] JSON saved: {json_path}")

    # Save per-tile data (might be large; save only key fields)
    per_tile_trunc = [
        {k: v for k, v in t.items()
         if k in ("step","tile_id","n_total","n_reused","n_new","is_repair",
                  "n_inversions","dirty_entry_count","dirty_entry_ratio",
                  "n_runs","run_sizes","mean_run_size","max_run_size",
                  "largest_run_ratio","violation_density","repair_expansion_ratio",
                  "repair_full_ratio","runs_interact")}
        for t in per_tile
    ]
    per_tile_path = ROOT / "results" / "phase-c19" / "dirty_run" / "per_tile.json"
    per_tile_path.write_text(json.dumps(per_tile_trunc, indent=1, default=str))
    print(f"[C19-1A] Per-tile data saved: {per_tile_path}")

    report_path = ROOT / "reports" / "phase-c19" / "c19-1a_dirty_run_locality.md"
    write_report(agg, report_path)

    print(f"\n{'='*70}")
    print(f"  C19-1A DECISION: {agg['decision']}")
    print(f"  {agg['decision_detail']}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
