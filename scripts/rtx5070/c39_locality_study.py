#!/usr/bin/env python3
"""
C39: Local Invalidation Locality Study.

C38 proved global state reuse impossible (0/499 exact pairs).
This experiment tests whether the changes are spatially LOCALIZED:
- How many Gaussians change per iter? (already known: ~74 from C38)
- How many tiles are affected (dirty)?
- What fraction of 8160 tiles are dirty?
- Is the dirty region spatially concentrated (low entropy) or uniform (high entropy)?

Method:
- Run 500 iterations, fixed camera, no densification (same as C38)
- For each adjacent pair (t, t+1):
  1. Find GS with visibility change (radii 0↔>0) → "changed GS"
  2. Find GS whose tile assignment changed (floor(means2d/tile_size) differs)
  3. Project changed GS to screen → compute affected tiles
  4. Count dirty tiles, compute ratio, compute spatial entropy
"""
import json, math, os, time, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "epic05" / "phase7"))

from gsplat import rasterization
from gaussian_model import GaussianModel
from loss import combined_loss
from dataset import GTDataset


# ─── Helper: extract per-Gaussian global state from packed meta ───
def extract_global_state(info, N, tile_size):
    """Convert packed projection output to global [N, ...] arrays."""
    gids = info["gaussian_ids"]  # [nnz] int32

    # means2d: [nnz, 2] → [N, 2]
    means2d = torch.zeros(N, 2, dtype=torch.float32, device=info["means2d"].device)
    means2d[gids] = info["means2d"]

    # radii: [nnz, 2] or [nnz] → [N] (1D, float)
    radii = info["radii"]
    if radii.dim() > 1:
        radii = radii.max(dim=-1).values
    radii_global = torch.zeros(N, dtype=torch.float32, device=radii.device)
    radii_global[gids] = radii.float()

    # depths: [nnz] or [nnz, 1] → [N]
    depths = info["depths"]
    if depths.dim() > 1:
        depths = depths.squeeze(-1)
    depths_global = torch.zeros(N, dtype=torch.float32, device=depths.device)
    depths_global[gids] = depths.float()

    # tile assignment: floor(means2d / tile_size)
    # This gives the tile (tx, ty) that each GS center falls in
    tile_x = torch.floor(means2d[:, 0] / tile_size).int()
    tile_y = torch.floor(means2d[:, 1] / tile_size).int()

    return {
        "means2d": means2d,
        "radii": radii_global,
        "depths": depths_global,
        "tile_x": tile_x,
        "tile_y": tile_y,
        "gids": gids,
    }


def compute_dirty_tiles_from_changed_gs(state_t, state_t1, tile_size, img_w, img_h):
    """
    Given two global states, identify:
    1. Changed GS (visibility change or tile assignment change)
    2. Tiles affected by those GS (dirty tiles)

    A GS affects tiles = all tiles its bounding box overlaps.
    Bounding box = center ± radius (in pixels).
    """
    N = state_t["means2d"].shape[0]

    # ── 1. Visibility changes ──
    vis_t = state_t["radii"] > 0
    vis_t1 = state_t1["radii"] > 0
    vis_changed = vis_t != vis_t1  # [N] bool

    # ── 2. Tile assignment changes (for GS visible in both) ──
    tile_changed = (state_t["tile_x"] != state_t1["tile_x"]) | \
                   (state_t["tile_y"] != state_t1["tile_y"])

    # Only count tile changes for GS visible in at least one state
    tile_changed = tile_changed & (vis_t | vis_t1)

    # ── 3. All changed GS ──
    changed = vis_changed | tile_changed
    n_changed = changed.sum().item()

    # ── 4. Compute dirty tiles ──
    # For each changed GS, compute tiles it affected at t AND t+1
    # A GS with radius r at center (cx, cy) covers tiles from
    # floor((cx-r)/tile_size) to floor((cx+r)/tile_size), same for y

    dirty_tiles = set()

    changed_idx = changed.nonzero(as_tuple=True)[0]

    for gid in changed_idx:
        gid = gid.item()

        # Tiles at state t (if visible)
        if vis_t[gid]:
            cx, cy = state_t["means2d"][gid, 0].item(), state_t["means2d"][gid, 1].item()
            r = state_t["radii"][gid].item()
            tx_min = max(0, int((cx - r) // tile_size))
            tx_max = min(img_w // tile_size - 1, int((cx + r) // tile_size))
            ty_min = max(0, int((cy - r) // tile_size))
            ty_max = min(img_h // tile_size - 1, int((cy + r) // tile_size))
            for ty in range(ty_min, ty_max + 1):
                for tx in range(tx_min, tx_max + 1):
                    dirty_tiles.add((tx, ty))

        # Tiles at state t+1 (if visible)
        if vis_t1[gid]:
            cx, cy = state_t1["means2d"][gid, 0].item(), state_t1["means2d"][gid, 1].item()
            r = state_t1["radii"][gid].item()
            tx_min = max(0, int((cx - r) // tile_size))
            tx_max = min(img_w // tile_size - 1, int((cx + r) // tile_size))
            ty_min = max(0, int((cy - r) // tile_size))
            ty_max = min(img_h // tile_size - 1, int((cy + r) // tile_size))
            for ty in range(ty_min, ty_max + 1):
                for tx in range(tx_min, tx_max + 1):
                    dirty_tiles.add((tx, ty))

    return n_changed, vis_changed.sum().item(), tile_changed.sum().item(), dirty_tiles


def compute_spatial_entropy(dirty_tiles, n_tiles_w, n_tiles_h):
    """
    Compute spatial entropy of dirty tile distribution.
    Low entropy = concentrated (good for local update).
    High entropy = uniform/spread (bad,接近 full rebuild).
    Max entropy = log2(n_dirty) if all tiles equally likely.
    Normalized entropy = H / log2(n_total_tiles).
    """
    if not dirty_tiles:
        return 0.0, 0.0

    # Create a 2D grid and mark dirty tiles
    grid = np.zeros((n_tiles_h, n_tiles_w), dtype=np.float32)
    for tx, ty in dirty_tiles:
        if 0 <= tx < n_tiles_w and 0 <= ty < n_tiles_h:
            grid[ty, tx] = 1.0

    n_dirty = len(dirty_tiles)
    n_total = n_tiles_w * n_tiles_h

    # Spatial entropy: divide grid into 4x4 quadrants, count dirty in each
    # Then compute Shannon entropy of the distribution
    # This measures spatial concentration vs spread
    n_quad = 4  # 4x4 = 16 quadrants
    quad_counts = np.zeros(n_quad * n_quad, dtype=np.float32)
    qh = n_tiles_h // n_quad
    qw = n_tiles_w // n_quad
    for qi in range(n_quad):
        for qj in range(n_quad):
            quad_counts[qi * n_quad + qj] = grid[qi*qh:(qi+1)*qh, qj*qw:(qj+1)*qw].sum()

    total = quad_counts.sum()
    if total == 0:
        return 0.0, 0.0

    probs = quad_counts / total
    # Shannon entropy (bits)
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    max_entropy = math.log2(min(n_quad * n_quad, n_dirty)) if n_dirty > 1 else 0
    # Normalized: entropy / log2(16) = entropy / 4
    norm_entropy = entropy / 4.0 if n_dirty > 1 else 0.0

    return entropy, norm_entropy


def main():
    print("=" * 72)
    print("C39: Local Invalidation Locality Study")
    print("=" * 72)

    device = "cuda"
    torch.manual_seed(42)

    TILE_SIZE = 16
    PACKED = True
    SH_DEGREE = 3
    EPS2D = 0.1
    RADIUS_CLIP = 0.0
    SPATIAL_LR_SCALE = 46.64

    repo_root = Path(__file__).resolve().parent.parent.parent
    print(f"\nRepo root: {repo_root}")

    # Load dataset
    print("\n  [Loading dataset...]")
    dataset = GTDataset(
        scene="room",
        repo_root=repo_root,
        resolution="1080p",
        device=device,
    )
    cam0 = dataset.get_camera(0)
    gt0 = dataset.get_gt_image(0)
    img_w, img_h = cam0.image_width, cam0.image_height
    n_tiles_w = math.ceil(img_w / TILE_SIZE)  # 120
    n_tiles_h = math.ceil(img_h / TILE_SIZE)  # 68
    n_total_tiles = n_tiles_w * n_tiles_h     # 8160
    print(f"  Camera 0: {img_w}x{img_h}, tiles: {n_tiles_w}x{n_tiles_h} = {n_total_tiles}")

    # Load model
    ckpt_path = repo_root / "results" / "epic05" / "phase7" / "phase7_room_30k_16" / "phase7_room_30k_16_iter5000.pt"
    print(f"\n  Loading checkpoint: {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = GaussianModel.from_checkpoint_state(ckpt["model_state"], device=device)
    N = model.xyz.shape[0]
    print(f"  Model: {N:,} Gaussians, SH degree: {model.sh_degree}")

    # Optimizer
    optimizer = torch.optim.Adam([
        {"params": [model.xyz], "lr": 1.6e-4 * SPATIAL_LR_SCALE, "eps": 1e-15, "betas": (0.9, 0.999)},
        {"params": [model.rotations], "lr": 1e-3, "eps": 1e-15, "betas": (0.9, 0.999)},
        {"params": [model.scales], "lr": 5e-3, "eps": 1e-15, "betas": (0.9, 0.999)},
        {"params": [model.opacity], "lr": 5e-2, "eps": 1e-15, "betas": (0.9, 0.999)},
        {"params": [model.shs], "lr": 2.5e-3, "eps": 1e-15, "betas": (0.9, 0.999)},
    ])

    # ─── Warmup (same as C38) ───
    WARMUP = 50
    N_ITERS = 500

    print(f"\n  Warmup ({WARMUP} iters)...")
    for i in range(WARMUP):
        data = model.forward()
        rendered, _, _ = rasterization(
            means=data["xyz"], quats=data["rotations"], scales=data["scales"],
            opacities=data["opacity"], colors=data["shs"],
            viewmats=cam0.viewmatrix.unsqueeze(0), Ks=cam0.K.unsqueeze(0),
            width=img_w, height=img_h,
            tile_size=TILE_SIZE, packed=PACKED, sh_degree=model.sh_degree,
            radius_clip=RADIUS_CLIP, eps2d=EPS2D, render_mode="RGB",
        )
        loss = combined_loss(rendered[0].clamp(0, 1), gt0, lambda_dssim=0.2)["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    # ─── Recorded iterations ───
    print(f"\n  Recording {N_ITERS} iterations...")

    results = []
    prev_state = None

    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)
    timings = []

    for iter_idx in range(1, N_ITERS + 1):
        data = model.forward()

        ev_start.record()
        rendered, _, info = rasterization(
            means=data["xyz"], quats=data["rotations"], scales=data["scales"],
            opacities=data["opacity"], colors=data["shs"],
            viewmats=cam0.viewmatrix.unsqueeze(0), Ks=cam0.K.unsqueeze(0),
            width=img_w, height=img_h,
            tile_size=TILE_SIZE, packed=PACKED, sh_degree=model.sh_degree,
            radius_clip=RADIUS_CLIP, eps2d=EPS2D, render_mode="RGB",
        )
        ev_end.record()
        torch.cuda.synchronize()
        fwd_ms = ev_start.elapsed_time(ev_end)
        timings.append(fwd_ms)

        # Extract global state and move to CPU for comparison
        cur_state = extract_global_state(info, N, TILE_SIZE)
        cur_state = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in cur_state.items()}

        # Compare with previous iteration
        if prev_state is not None:
            n_changed, n_vis_changed, n_tile_changed, dirty_tiles = compute_dirty_tiles_from_changed_gs(
                prev_state, cur_state, TILE_SIZE, img_w, img_h
            )

            n_dirty = len(dirty_tiles)
            dirty_ratio = n_dirty / n_total_tiles * 100

            entropy, norm_entropy = compute_spatial_entropy(dirty_tiles, n_tiles_w, n_tiles_h)

            # Also compute: dirty tiles from ONLY visibility changes
            # (vs from ALL changes including tile assignment)
            _, _, _, dirty_vis_only = compute_dirty_tiles_from_changed_gs(
                {**prev_state, "tile_x": cur_state["tile_x"], "tile_y": cur_state["tile_y"]},
                {**cur_state, "tile_x": prev_state["tile_x"], "tile_y": prev_state["tile_y"]},
                TILE_SIZE, img_w, img_h
            )

            result = {
                "iter": iter_idx,
                "n_changed_gs": n_changed,
                "n_vis_changed": n_vis_changed,
                "n_tile_changed": n_tile_changed,
                "n_dirty_tiles": n_dirty,
                "dirty_ratio_pct": dirty_ratio,
                "spatial_entropy": entropy,
                "norm_entropy": norm_entropy,
                "n_dirty_vis_only": len(dirty_vis_only),
                "fwd_ms": fwd_ms,
            }
            results.append(result)

            if iter_idx % 50 == 0:
                print(f"  iter={iter_idx:4d}  changed_gs={n_changed:5d}  "
                      f"vis_chg={n_vis_changed:4d}  tile_chg={n_tile_changed:5d}  "
                      f"dirty={n_dirty:5d}/{n_total_tiles} ({dirty_ratio:5.1f}%)  "
                      f"H_norm={norm_entropy:.3f}")
        else:
            print(f"  iter={iter_idx:4d}  (first iter, no comparison)")

        # Save state for next iteration (already on CPU)
        prev_state = cur_state

        # Backward + step
        rendered_img = rendered[0].clamp(0, 1)
        loss = combined_loss(rendered_img, gt0, lambda_dssim=0.2)["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    # ─── Analysis ───
    print(f"\n{'='*72}")
    print("C39 LOCALITY ANALYSIS")
    print(f"{'='*72}")

    n_changed_arr = np.array([r["n_changed_gs"] for r in results])
    n_vis_arr = np.array([r["n_vis_changed"] for r in results])
    n_tile_arr = np.array([r["n_tile_changed"] for r in results])
    n_dirty_arr = np.array([r["n_dirty_tiles"] for r in results])
    dirty_ratio_arr = np.array([r["dirty_ratio_pct"] for r in results])
    entropy_arr = np.array([r["norm_entropy"] for r in results])
    n_dirty_vis_arr = np.array([r["n_dirty_vis_only"] for r in results])
    timings_arr = np.array(timings[1:])  # skip first

    print(f"\n  Total iterations analyzed: {len(results)}")
    print(f"  Total tiles: {n_total_tiles}")

    print(f"\n--- A. Changed Gaussian count ---")
    print(f"  n_changed_gs:    mean={n_changed_arr.mean():.1f}  median={np.median(n_changed_arr):.0f}  "
          f"p90={np.percentile(n_changed_arr,90):.0f}  max={n_changed_arr.max()}")
    print(f"  n_vis_changed:   mean={n_vis_arr.mean():.1f}  median={np.median(n_vis_arr):.0f}  "
          f"p90={np.percentile(n_vis_arr,90):.0f}  max={n_vis_arr.max()}")
    print(f"  n_tile_changed:  mean={n_tile_arr.mean():.1f}  median={np.median(n_tile_arr):.0f}  "
          f"p90={np.percentile(n_tile_arr,90):.0f}  max={n_tile_arr.max()}")

    print(f"\n--- B. Dirty tile count ---")
    print(f"  n_dirty_tiles:   mean={n_dirty_arr.mean():.1f}  median={np.median(n_dirty_arr):.0f}  "
          f"p90={np.percentile(n_dirty_arr,90):.0f}  max={n_dirty_arr.max()}")
    print(f"  n_dirty_vis_only:mean={n_dirty_vis_arr.mean():.1f}  median={np.median(n_dirty_vis_arr):.0f}  "
          f"p90={np.percentile(n_dirty_vis_arr,90):.0f}")

    print(f"\n--- C. Dirty tile ratio ---")
    print(f"  dirty_ratio:     mean={dirty_ratio_arr.mean():.2f}%  median={np.median(dirty_ratio_arr):.2f}%  "
          f"p90={np.percentile(dirty_ratio_arr,90):.2f}%  max={dirty_ratio_arr.max():.2f}%")
    print(f"  dirty_ratio < 10%: {np.mean(dirty_ratio_arr < 10)*100:.1f}% of iterations")
    print(f"  dirty_ratio < 5%:  {np.mean(dirty_ratio_arr < 5)*100:.1f}% of iterations")
    print(f"  dirty_ratio < 1%:  {np.mean(dirty_ratio_arr < 1)*100:.1f}% of iterations")

    print(f"\n--- D. Spatial entropy (normalized, 0=concentrated, 1=uniform) ---")
    print(f"  norm_entropy:    mean={entropy_arr.mean():.3f}  median={np.median(entropy_arr):.3f}  "
          f"p90={np.percentile(entropy_arr,90):.3f}  max={entropy_arr.max():.3f}")

    print(f"\n--- E. Cost comparison ---")
    print(f"  Full forward (project+intersect+rasterize): mean={timings_arr.mean():.2f}ms")
    print(f"  Dirty tiles as fraction of total: {n_dirty_arr.mean()/n_total_tiles*100:.2f}%")
    print(f"  Estimated incremental cost: {timings_arr.mean() * n_dirty_arr.mean() / n_total_tiles:.3f}ms")
    print(f"  Savings if incremental: {timings_arr.mean() - timings_arr.mean() * n_dirty_arr.mean() / n_total_tiles:.3f}ms")
    print(f"  Savings ratio: {(1 - n_dirty_arr.mean() / n_total_tiles) * 100:.1f}%")

    # ─── Distribution of dirty_ratio ───
    print(f"\n--- Dirty ratio distribution ---")
    for pct in [1, 2, 5, 10, 20, 50, 100]:
        frac = np.mean(dirty_ratio_arr <= pct) * 100
        print(f"  <= {pct:3d}%: {frac:.1f}% of iterations")

    # ─── Final verdict ───
    print(f"\n{'='*72}")
    print("DECISION GATE")
    print(f"{'='*72}")

    median_dirty = np.median(dirty_ratio_arr)
    mean_dirty = dirty_ratio_arr.mean()
    p90_dirty = np.percentile(dirty_ratio_arr, 90)

    print(f"\n  Median dirty tile ratio: {median_dirty:.2f}%")
    print(f"  Mean dirty tile ratio:   {mean_dirty:.2f}%")
    print(f"  P90 dirty tile ratio:    {p90_dirty:.2f}%")

    if median_dirty < 10:
        print(f"\n  → KEEP (dirty ratio consistently <10%)")
    elif median_dirty < 50:
        print(f"\n  → MAYBE (dirty ratio 10-50%)")
    else:
        print(f"\n  → DROP (dirty ratio >50%)")

    # ─── Save data ───
    output = {
        "n_total_tiles": n_total_tiles,
        "n_tiles_w": n_tiles_w,
        "n_tiles_h": n_tiles_h,
        "n_gaussians": N,
        "n_iters": N_ITERS,
        "warmup": WARMUP,
        "results": results,
        "summary": {
            "n_changed_gs": {
                "mean": float(n_changed_arr.mean()),
                "median": float(np.median(n_changed_arr)),
                "p90": float(np.percentile(n_changed_arr, 90)),
                "max": int(n_changed_arr.max()),
            },
            "n_dirty_tiles": {
                "mean": float(n_dirty_arr.mean()),
                "median": float(np.median(n_dirty_arr)),
                "p90": float(np.percentile(n_dirty_arr, 90)),
                "max": int(n_dirty_arr.max()),
            },
            "dirty_ratio_pct": {
                "mean": float(dirty_ratio_arr.mean()),
                "median": float(np.median(dirty_ratio_arr)),
                "p90": float(np.percentile(dirty_ratio_arr, 90)),
                "max": float(dirty_ratio_arr.max()),
            },
            "norm_entropy": {
                "mean": float(entropy_arr.mean()),
                "median": float(np.median(entropy_arr)),
                "p90": float(np.percentile(entropy_arr, 90)),
            },
            "fwd_ms": {
                "mean": float(timings_arr.mean()),
                "median": float(np.median(timings_arr)),
            },
        },
    }

    save_path = Path("results/phase-c31/c39_locality_data.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy/torch types to native Python types for JSON
    def convert(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    with open(save_path, "w") as f:
        json.dump(output, f, indent=2, default=convert)
    print(f"\n  Data saved to {save_path}")


if __name__ == "__main__":
    main()
