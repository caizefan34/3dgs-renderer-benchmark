#!/usr/bin/env python3
"""AccuTile A100 B1/P/A comparison for Room scene.

Compares three variants:
  B1 = frozen AABB baseline (gsplat 1.5.3, accutile=False)
  P  = per-tile conservative conic predicate (gsplat-accutile-v153, accutile=True)
  A  = true upstream AccuTile strip-based (gsplat-true-accutile-v153, accutile=True)

All three use the SAME frozen Room 30K checkpoint and camera 0.
"""
import sys, os, json, math, importlib, time
import torch
import numpy as np

CKPT = "/mnt/storage_pool/3dgs-renderer-benchmark/repo/results/epic05/phase7/a100_30k_room_t16_16/a100_30k_room_t16_16_latest.pt"
CAMS = "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/room/cameras.json"

def load_checkpoint():
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    return ckpt.get("model_state", ckpt)

def load_camera():
    with open(CAMS) as f:
        return json.load(f)[0]

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
    return means, quats, scales, opacities, colors, int(state.get("sh_degree", 3))

def tensor_metrics(a, b):
    a, b = a.detach(), b.detach()
    diff = (a - b)
    abs_diff = diff.abs()
    norm_b = b.flatten().norm().item()
    return {
        "max_abs": float(abs_diff.max()),
        "mean_abs": float(abs_diff.mean()),
        "relative_L2": (diff.flatten().norm().item() / norm_b) if norm_b > 0 else float('inf'),
        "nan_a": int(torch.isnan(a).sum()), "nan_b": int(torch.isnan(b).sum()),
        "inf_a": int(torch.isinf(a).sum()), "inf_b": int(torch.isinf(b).sum()),
    }

def run_variant(gsplat_path, accutile, state, viewmats, Ks, width, height, device="cuda"):
    """Run forward+backward with a specific gsplat variant."""
    # Force reimport of gsplat from the specified path
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("gsplat"):
            del sys.modules[mod_name]
    sys.path.insert(0, gsplat_path)
    import gsplat
    from gsplat import rasterization
    importlib.reload(gsplat)

    means, quats, scales, opacities, colors, shd = prep_tensors(state, device)
    inputs = (means, quats, scales, opacities, colors)
    for t in inputs:
        t.grad = None

    rgb, alpha, meta = rasterization(
        means, quats, scales, opacities, colors, viewmats, Ks,
        width, height, tile_size=16, packed=False,
        sh_degree=shd, absgrad=True, accutile=accutile,
    )
    (rgb.sum() + alpha.sum()).backward()
    torch.cuda.synchronize()

    result = {
        "gsplat_path": gsplat_path,
        "gsplat_version": gsplat.__version__,
        "accutile": accutile,
        "rgb": rgb.detach().clone(),
        "alpha": alpha.detach().clone(),
        "intersections": int(meta["tiles_per_gauss"].sum()),
        "inputs": inputs,
        "meta": meta,
    }
    sys.path.remove(gsplat_path)
    return result

def run_benchmark(gsplat_path, accutile, state, viewmats, Ks, width, height,
                  warmup=20, measure=100, device="cuda"):
    """Formal timing benchmark with CUDA events."""
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("gsplat"):
            del sys.modules[mod_name]
    sys.path.insert(0, gsplat_path)
    import gsplat
    from gsplat import rasterization

    means, quats, scales, opacities, colors, shd = prep_tensors(state, device)

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

    times = {"forward": [], "backward": [], "total": []}
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

    stats = {}
    for key in ("forward", "backward", "total"):
        arr = np.array(times[key])
        stats[key] = {
            "mean_ms": float(arr.mean()),
            "median_ms": float(np.median(arr)),
            "std_ms": float(arr.std()),
            "p95_ms": float(np.percentile(arr, 95)),
        }
    stats["intersections"] = int(meta["tiles_per_gauss"].sum())
    sys.path.remove(gsplat_path)
    return stats

def main():
    device = "cuda"
    state = load_checkpoint()
    cam = load_camera()
    viewmats, Ks, width, height = make_viewmat_K(cam, device)

    variants = {
        "B1": {"path": "/mnt/storage_pool/liaoyuanjun/gsplat-clean-mx", "accutile": False},
        "P":  {"path": "/mnt/storage_pool/liaoyuanjun/gsplat-accutile-v153", "accutile": True},
        "A":  {"path": "/mnt/storage_pool/liaoyuanjun/gsplat-true-accutile-v153", "accutile": True},
    }

    results = {}
    print("=" * 70)
    print("B1/P/A Comparison on Room (30K checkpoint, camera 0)")
    print("=" * 70)
    print(f"Gaussians: {state['xyz'].shape[0]}, Resolution: {width}x{height}, SH: {state.get('sh_degree', 3)}")

    # ── Correctness ──
    print("\n--- Correctness ---")
    correctness = {}
    for name, cfg in variants.items():
        print(f"\nRunning {name} (accutile={cfg['accutile']})...")
        r = run_variant(cfg["path"], cfg["accutile"], state, viewmats, Ks, width, height, device)
        correctness[name] = r
        print(f"  intersections: {r['intersections']}")

    # Compare against B1
    b1 = correctness["B1"]
    for name in ("P", "A"):
        v = correctness[name]
        print(f"\n{name} vs B1 forward:")
        print(f"  RGB: max_abs={tensor_metrics(b1['rgb'], v['rgb'])['max_abs']}, "
              f"rel_L2={tensor_metrics(b1['rgb'], v['rgb'])['relative_L2']}")
        print(f"  alpha: max_abs={tensor_metrics(b1['alpha'], v['alpha'])['max_abs']}, "
              f"rel_L2={tensor_metrics(b1['alpha'], v['alpha'])['relative_L2']}")
        grad_names = ("means", "quats", "scales", "opacities", "colors")
        for gname, a, b in zip(grad_names, b1["inputs"], v["inputs"]):
            ga = a.grad.detach() if a.grad is not None else None
            gb = b.grad.detach() if b.grad is not None else None
            if ga is not None and gb is not None:
                m = tensor_metrics(ga, gb)
                print(f"  grad {gname}: max_abs={m['max_abs']:.4e}, rel_L2={m['relative_L2']:.4e}")

    # ── Benchmark ──
    print("\n--- Timing Benchmark (warmup=20, measure=100) ---")
    benchmark = {}
    for name, cfg in variants.items():
        print(f"\nBenchmarking {name}...")
        stats = run_benchmark(cfg["path"], cfg["accutile"], state, viewmats, Ks, width, height, 20, 100, device)
        benchmark[name] = stats
        print(f"  forward: {stats['forward']['mean_ms']:.2f}ms")
        print(f"  backward: {stats['backward']['mean_ms']:.2f}ms")
        print(f"  total: {stats['total']['mean_ms']:.2f}ms")
        print(f"  intersections: {stats['intersections']}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Variant':<8} {'N_isect':>12} {'Fwd ms':>10} {'Bwd ms':>10} {'Total ms':>10} {'Fwd speedup':>12} {'Bwd speedup':>12} {'Total speedup':>14}")
    b1_fwd = benchmark["B1"]["forward"]["mean_ms"]
    b1_bwd = benchmark["B1"]["backward"]["mean_ms"]
    b1_tot = benchmark["B1"]["total"]["mean_ms"]
    for name in ("B1", "P", "A"):
        s = benchmark[name]
        fwd_sp = b1_fwd / s["forward"]["mean_ms"] if name != "B1" else 1.0
        bwd_sp = b1_bwd / s["backward"]["mean_ms"] if name != "B1" else 1.0
        tot_sp = b1_tot / s["total"]["mean_ms"] if name != "B1" else 1.0
        print(f"{name:<8} {s['intersections']:>12,} {s['forward']['mean_ms']:>10.2f} {s['backward']['mean_ms']:>10.2f} "
              f"{s['total']['mean_ms']:>10.2f} {fwd_sp:>12.4f} {bwd_sp:>12.4f} {tot_sp:>14.4f}")

    # Save results
    output = {
        "scene": "room",
        "gaussian_count": int(state["xyz"].shape[0]),
        "resolution": [width, height],
        "correctness": {
            name: {
                "intersections": v["intersections"],
                "rgb_vs_B1": tensor_metrics(b1["rgb"], v["rgb"]) if name != "B1" else None,
                "alpha_vs_B1": tensor_metrics(b1["alpha"], v["alpha"]) if name != "B1" else None,
                "grad_vs_B1": {
                    gname: tensor_metrics(a.grad.detach(), b.grad.detach())
                    for gname, a, b in zip(("means","quats","scales","opacities","colors"),
                                           b1["inputs"], v["inputs"])
                    if a.grad is not None and b.grad is not None
                } if name != "B1" else None,
            }
            for name, v in correctness.items()
        },
        "benchmark": benchmark,
    }
    out_path = "/tmp/accutile_a100_results/accutile_b1pa_room.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
