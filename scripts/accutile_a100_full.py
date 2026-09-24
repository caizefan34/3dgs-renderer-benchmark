#!/usr/bin/env python3
"""AccuTile A100 full validation: correctness, intersections, and timing.

Runs on the A100 using frozen 30K checkpoints for room/bicycle/garden.
Compares B1 (baseline) vs B1A (AccuTile) with matched state.

Usage:
    python3 accutile_a100_full.py --scene room --out /tmp/accutile_a100_results
    python3 accutile_a100_full.py --scene all --out /tmp/accutile_a100_results
"""
import sys, os, json, argparse, time, math
sys.path.insert(0, "/mnt/storage_pool/liaoyuanjun/gsplat-accutile-v153")
import torch
import numpy as np
from gsplat import rasterization

CKPT_BASE = "/mnt/storage_pool/3dgs-renderer-benchmark/repo/results/epic05/phase7"
CAM_BASE = "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360"
SCENES = {
    "room":     {"ckpt": f"{CKPT_BASE}/a100_30k_room_t16_16/a100_30k_room_t16_16_latest.pt",
                 "cams": f"{CAM_BASE}/room/cameras.json"},
    "bicycle":  {"ckpt": f"{CKPT_BASE}/a100_30k_bicycle_t16_16/a100_30k_bicycle_t16_16_latest.pt",
                 "cams": f"{CAM_BASE}/bicycle/cameras.json"},
    "garden":   {"ckpt": f"{CKPT_BASE}/a100_30k_garden_t16_16/a100_30k_garden_t16_16_latest.pt",
                 "cams": f"{CAM_BASE}/garden/cameras.json"},
}


def load_checkpoint(scene):
    path = SCENES[scene]["ckpt"]
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt)
    return state


def load_camera(scene, cam_idx=0):
    path = SCENES[scene]["cams"]
    with open(path) as f:
        cams = json.load(f)
    cam = cams[cam_idx]
    return cam


def make_viewmat_K(cam, device="cuda"):
    rotation = torch.tensor(cam["rotation"], dtype=torch.float32, device=device)
    position = torch.tensor(cam["position"], dtype=torch.float32, device=device)
    viewmat = torch.eye(4, dtype=torch.float32, device=device)
    viewmat[:3, :3] = rotation
    viewmat[:3, 3] = -rotation @ position
    K = torch.tensor([[cam["fx"], 0., cam["width"] / 2],
                      [0., cam["fy"], cam["height"] / 2],
                      [0., 0., 1.]], dtype=torch.float32, device=device)[None]
    return viewmat[None], K, cam["width"], cam["height"]


def prep_tensors(state, device="cuda"):
    means = state["xyz"].to(device).requires_grad_(True)
    quats = state["rotations"].to(device).requires_grad_(True)
    scales = torch.exp(state["scales"].to(device)).requires_grad_(True)
    opacities = torch.sigmoid(state["opacity"].to(device).flatten()).requires_grad_(True)
    colors = state["shs"].to(device).requires_grad_(True)
    sh_degree = int(state.get("sh_degree", 3))
    return means, quats, scales, opacities, colors, sh_degree


def tensor_metrics(a, b):
    """Compare two tensors: max_abs, mean_abs, relative_L2, NaN/Inf counts."""
    a = a.detach()
    b = b.detach()
    diff = (a - b)
    abs_diff = diff.abs()
    a_flat = a.flatten()
    b_flat = b.flatten()
    norm_b = b_flat.norm().item()
    rel_l2 = (diff.flatten().norm().item() / norm_b) if norm_b > 0 else float('inf')
    return {
        "max_abs": float(abs_diff.max()),
        "mean_abs": float(abs_diff.mean()),
        "relative_L2": rel_l2,
        "nan_count_a": int(torch.isnan(a).sum()),
        "nan_count_b": int(torch.isnan(b).sum()),
        "inf_count_a": int(torch.isinf(a).sum()),
        "inf_count_b": int(torch.isinf(b).sum()),
        "shape": list(a.shape),
    }


def run_correctness(scene, device="cuda"):
    """Full correctness gate: forward + backward tensor comparison."""
    state = load_checkpoint(scene)
    cam = load_camera(scene, 0)
    viewmats, Ks, width, height = make_viewmat_K(cam, device)
    sh_degree = int(state.get("sh_degree", 3))

    results = {"scene": scene, "gaussian_count": int(state["xyz"].shape[0]),
               "resolution": [width, height], "sh_degree": sh_degree}

    # Prepare fresh tensors for each run
    def run_once(accutile):
        st = load_checkpoint(scene)
        means, quats, scales, opacities, colors, shd = prep_tensors(st, device)
        inputs = (means, quats, scales, opacities, colors)
        for t in inputs:
            t.grad = None
        rgb, alpha, meta = rasterization(
            means, quats, scales, opacities, colors, viewmats, Ks,
            width, height, tile_size=16, packed=False,
            sh_degree=shd, absgrad=True, accutile=accutile,
        )
        loss = rgb.sum() + alpha.sum()
        loss.backward()
        torch.cuda.synchronize()
        return {
            "rgb": rgb.detach(), "alpha": alpha.detach(),
            "meta": meta, "inputs": inputs,
        }

    off = run_once(False)
    on = run_once(True)

    # Forward comparison
    results["forward"] = {
        "rgb": tensor_metrics(off["rgb"], on["rgb"]),
        "alpha": tensor_metrics(off["alpha"], on["alpha"]),
    }

    # Depth if available (render_mode D)
    # Try depth render
    try:
        st = load_checkpoint(scene)
        means_d, quats_d, scales_d, opacities_d, colors_d, shd_d = prep_tensors(st, device)
        for t in (means_d, quats_d, scales_d, opacities_d, colors_d):
            t.grad = None
        rgb_d, alpha_d, meta_d = rasterization(
            means_d, quats_d, scales_d, opacities_d, colors_d, viewmats, Ks,
            width, height, tile_size=16, packed=False,
            sh_degree=shd_d, render_mode="RGB+D", accutile=False,
        )
        depth_off = rgb_d[..., 3].detach()
        for t in (means_d, quats_d, scales_d, opacities_d, colors_d):
            t.grad = None
        rgb_d2, alpha_d2, meta_d2 = rasterization(
            means_d, quats_d, scales_d, opacities_d, colors_d, viewmats, Ks,
            width, height, tile_size=16, packed=False,
            sh_degree=shd_d, render_mode="RGB+D", accutile=True,
        )
        depth_on = rgb_d2[..., 3].detach()
        results["forward"]["depth"] = tensor_metrics(depth_off, depth_on)
    except Exception as e:
        results["forward"]["depth"] = {"error": str(e)}

    # Backward gradient comparison
    grad_names = ("means", "quats", "scales", "opacities", "colors")
    results["backward"] = {}
    for name, a, b in zip(grad_names, off["inputs"], on["inputs"]):
        ga = a.grad.detach() if a.grad is not None else None
        gb = b.grad.detach() if b.grad is not None else None
        if ga is not None and gb is not None:
            results["backward"][name] = tensor_metrics(ga, gb)
        else:
            results["backward"][name] = {"error": "missing gradient"}

    # Intersection counts
    results["intersections"] = {
        "N_AABB": int(off["meta"]["tiles_per_gauss"].sum()),
        "N_AccuTile": int(on["meta"]["tiles_per_gauss"].sum()),
    }
    results["intersections"]["reduction"] = 1.0 - results["intersections"]["N_AccuTile"] / results["intersections"]["N_AABB"]

    # Intersection distribution stats
    tpg_off = off["meta"]["tiles_per_gauss"].flatten()
    tpg_on = on["meta"]["tiles_per_gauss"].flatten()
    tile_width = math.ceil(width / 16)
    tile_height = math.ceil(height / 16)
    n_tiles = tile_width * tile_height
    results["intersection_stats"] = {
        "n_tiles": n_tiles,
        "avg_per_tile_AABB": float(tpg_off.sum()) / n_tiles,
        "avg_per_tile_AccuTile": float(tpg_on.sum()) / n_tiles,
        "p50_per_gauss_AABB": float(tpg_off.float().median()),
        "p95_per_gauss_AABB": float(tpg_off.float().quantile(0.95)),
        "p99_per_gauss_AABB": float(tpg_off.float().quantile(0.99)),
        "p50_per_gauss_AccuTile": float(tpg_on.float().median()),
        "p95_per_gauss_AccuTile": float(tpg_on.float().quantile(0.95)),
        "p99_per_gauss_AccuTile": float(tpg_on.float().quantile(0.99)),
        "max_per_gauss_AABB": int(tpg_off.max()),
        "max_per_gauss_AccuTile": int(tpg_on.max()),
    }

    return results


def run_benchmark(scene, device="cuda", warmup=20, measure=100):
    """Formal timing benchmark with CUDA events."""
    state = load_checkpoint(scene)
    cam = load_camera(scene, 0)
    viewmats, Ks, width, height = make_viewmat_K(cam, device)
    sh_degree = int(state.get("sh_degree", 3))

    results = {"scene": scene, "warmup": warmup, "measure": measure}

    for mode_name, accutile in [("B1", False), ("B1A", True)]:
        st = load_checkpoint(scene)
        means, quats, scales, opacities, colors, shd = prep_tensors(st, device)

        # Warmup
        for _ in range(warmup):
            for t in (means, quats, scales, opacities, colors):
                t.grad = None
            rgb, alpha, meta = rasterization(
                means, quats, scales, opacities, colors, viewmats, Ks,
                width, height, tile_size=16, packed=False,
                sh_degree=shd, absgrad=True, accutile=accutile,
            )
            (rgb.sum() + alpha.sum()).backward()
            torch.cuda.synchronize()

        # Measure with CUDA events
        times = {"total": [], "forward": [], "backward": []}
        for _ in range(measure):
            for t in (means, quats, scales, opacities, colors):
                t.grad = None

            start = torch.cuda.Event(enable_timing=True)
            fwd_end = torch.cuda.Event(enable_timing=True)
            bwd_end = torch.cuda.Event(enable_timing=True)

            start.record()
            rgb, alpha, meta = rasterization(
                means, quats, scales, opacities, colors, viewmats, Ks,
                width, height, tile_size=16, packed=False,
                sh_degree=shd, absgrad=True, accutile=accutile,
            )
            fwd_end.record()
            (rgb.sum() + alpha.sum()).backward()
            bwd_end.record()
            torch.cuda.synchronize()

            times["forward"].append(start.elapsed_time(fwd_end))
            times["backward"].append(fwd_end.elapsed_time(bwd_end))
            times["total"].append(start.elapsed_time(bwd_end))

        # Compute statistics
        stats = {}
        for key in ("forward", "backward", "total"):
            arr = np.array(times[key])
            stats[key] = {
                "mean_ms": float(arr.mean()),
                "median_ms": float(np.median(arr)),
                "std_ms": float(arr.std()),
                "p95_ms": float(np.percentile(arr, 95)),
            }

        # Intersection count for this run
        stats["intersections"] = int(meta["tiles_per_gauss"].sum())
        results[mode_name] = stats

    # Compute speedups
    for key in ("forward", "backward", "total"):
        b1 = results["B1"][key]["mean_ms"]
        b1a = results["B1A"][key]["mean_ms"]
        results[f"speedup_{key}"] = b1 / b1a if b1a > 0 else float('inf')

    # Delta analysis
    results["delta"] = {
        "intersections": results["B1"]["intersections"] - results["B1A"]["intersections"],
        "forward_ms": results["B1"]["forward"]["mean_ms"] - results["B1A"]["forward"]["mean_ms"],
        "backward_ms": results["B1"]["backward"]["mean_ms"] - results["B1A"]["backward"]["mean_ms"],
        "total_ms": results["B1"]["total"]["mean_ms"] - results["B1A"]["total"]["mean_ms"],
    }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=("room", "bicycle", "garden", "all"), default="all")
    parser.add_argument("--out", default="/tmp/accutile_a100_results")
    parser.add_argument("--mode", choices=("correctness", "benchmark", "both"), default="both")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--measure", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    scenes = list(SCENES.keys()) if args.scene == "all" else [args.scene]

    all_results = {}
    for scene in scenes:
        print(f"\n{'='*60}")
        print(f"Scene: {scene}")
        print(f"{'='*60}")
        scene_results = {"scene": scene}

        if args.mode in ("correctness", "both"):
            print(f"\n--- Correctness gate ---")
            corr = run_correctness(scene)
            scene_results["correctness"] = corr
            print(json.dumps({k: v for k, v in corr.items()
                             if k in ("intersections", "intersection_stats")}, indent=2))

        if args.mode in ("benchmark", "both"):
            print(f"\n--- Timing benchmark (warmup={args.warmup}, measure={args.measure}) ---")
            bench = run_benchmark(scene, warmup=args.warmup, measure=args.measure)
            scene_results["benchmark"] = bench
            print(json.dumps({k: v for k, v in bench.items()
                             if k in ("B1", "B1A", "speedup_forward", "speedup_backward",
                                      "speedup_total", "delta")}, indent=2))

        all_results[scene] = scene_results
        # Save per-scene result
        with open(f"{args.out}/accutile_{scene}.json", "w") as f:
            json.dump(scene_results, f, indent=2)

    # Save combined
    with open(f"{args.out}/accutile_all_scenes.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {args.out}/")


if __name__ == "__main__":
    main()
