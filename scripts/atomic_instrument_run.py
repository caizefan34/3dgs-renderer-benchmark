#!/usr/bin/env python3
"""Direct atomic instrumentation for B1 and B1A backward pass.

Uses gsplat's forward pass to extract metadata (means2d, conics, opacities,
tile_offsets, flatten_ids, render_alphas, last_ids), then runs a standalone
CUDA kernel that replicates the backward kernel's iteration pattern and counts:
  - n_warp_atomics: number of (warp, Gaussian) pairs with >=1 valid pixel
  - n_block_uniques: number of unique (block, Gaussian) pairs with >=1 valid pixel

Each warp-leader event = 11 gpuAtomicAdd calls.
R_atomic_direct = n_warp_atomics * 11
Block aggregation potential = n_warp_atomics / n_block_uniques
"""
import sys, os, json, math, time
import torch
import numpy as np
from pathlib import Path

REPO_ROOT = "/mnt/storage_pool/3dgs-renderer-benchmark/repo"
sys.path.insert(0, f"{REPO_ROOT}/src")
sys.path.insert(0, REPO_ROOT)

GSPLAT_PATH = "/mnt/storage_pool/liaoyuanjun/gsplat-true-accutile-v153"
sys.path.insert(0, GSPLAT_PATH)

CKPT_BASE = f"{REPO_ROOT}/results/epic05/phase7"
CAM_BASE = f"{REPO_ROOT}/data/official/mipnerf360"
ATOMIC_CU = "/tmp/atomic_instrument.cu"

SCENES = {
    "room": {
        "ckpt": f"{CKPT_BASE}/a100_30k_room_t16_16/a100_30k_room_t16_16_latest.pt",
        "cams": f"{CAM_BASE}/room/cameras.json",
    },
    "bicycle": {
        "ckpt": f"{CKPT_BASE}/a100_30k_bicycle_t16_16/a100_30k_bicycle_t16_16_latest.pt",
        "cams": f"{CAM_BASE}/bicycle/cameras.json",
    },
    "garden": {
        "ckpt": f"{CKPT_BASE}/a100_30k_garden_t16_16/a100_30k_garden_t16_16_latest.pt",
        "cams": f"{CAM_BASE}/garden/cameras.json",
    },
}

NUM_CAMERAS = 3  # test 3 cameras per scene

def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt.get("model_state", ckpt)

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

def get_forward_metadata(state, cam, accutile, device="cuda"):
    """Run forward pass and extract all metadata needed for atomic counting."""
    import gsplat
    from gsplat import rasterization
    from gsplat.cuda._wrapper import isect_tiles, isect_offset_encode

    means = state["xyz"].to(device).contiguous()
    quats = state["rotations"].to(device).contiguous()
    scales = torch.exp(state["scales"].to(device)).contiguous()
    opacities = torch.sigmoid(state["opacity"].to(device).flatten()).contiguous()
    colors = state["shs"].to(device).contiguous()
    sh_degree = int(state.get("sh_degree", 3))

    viewmats, Ks, width, height = make_viewmat_K(cam, device)

    # Run full rasterization to get render_alphas and last_ids
    kwargs = dict(sh_degree=sh_degree, absgrad=False, tile_size=16, packed=False, render_mode="RGB")
    if accutile:
        kwargs["accutile"] = True
    rgb, alpha, meta = rasterization(
        means, quats, scales, opacities, colors, viewmats, Ks,
        width, height, **kwargs)

    # Extract means2d, conics, opacities from projection
    # The meta dict has tiles_per_gauss, isect_ids, flatten_ids, isect_offsets
    tile_offsets = meta["isect_offsets"][0, 0].contiguous()  # [tile_height, tile_width]
    flatten_ids = meta["flatten_ids"].contiguous()
    render_alphas = alpha[0, ..., 0].contiguous()  # [H, W]
    # last_ids is saved in the autograd context — we need to get it from the forward kernel
    # We can call the forward pixel kernel directly
    from gsplat.cuda._wrapper import _make_lazy_cuda_func

    # Get means2d and conics from the projection step
    # We need to re-run the projection to get these
    # Actually, the rasterization function internally calls projection, which produces means2d, conics
    # Let's extract them by calling the projection directly

    # The radii are in meta
    radii = meta["radii"].contiguous()
    depths = meta["depths"].contiguous() if "depths" in meta else None

    # Re-run projection to get means2d and conics
    from gsplat import fully_fused_projection
    proj_results = fully_fused_projection(
        means, None, quats, scales, viewmats, Ks, width, height,
        eps2d=0.1, packed=False, sparse_grad=False, calc_compensations=False,
        camera_model="pinhole", opacities=opacities,
    )
    # Returns: radii, means2d, depths, conics, compensations
    means2d_proj = proj_results[1]  # [1, N, 2]
    conics_proj = proj_results[3]   # [1, N, 3]
    means2d = means2d_proj[0].contiguous()  # [N, 2]
    conics = conics_proj[0].contiguous()    # [N, 3]
    opacities_flat = opacities.contiguous()  # [N]

    # Get last_ids from forward pixel kernel
    # rasterize_to_pixels_3dgs_fwd returns (render_colors, render_alphas, last_ids)
    try:
        # colors for fwd kernel need to match the CDIM (3 for RGB)
        colors_2d = colors[0, :, :3].contiguous() if colors.dim() == 3 else colors[:, :3].contiguous()
        fwd_result = _make_lazy_cuda_func("rasterize_to_pixels_3dgs_fwd")(
            means2d.float(),
            conics.float(),
            colors_2d.float(),
            opacities_flat.float(),
            None,  # backgrounds
            None,  # masks
            width, height, 16,
            tile_offsets, flatten_ids,
        )
        last_ids = fwd_result[2].contiguous()  # [H, W]
        # render_alphas from fwd might be double, convert to float
        render_alphas = fwd_result[1].float().contiguous()  # [H, W, 1]
        render_alphas = render_alphas[..., 0].contiguous()  # [H, W]
    except Exception as e:
        print(f"  Warning: couldn't get last_ids directly: {e}")
        # Fallback: use the alpha from rasterization
        last_ids = torch.zeros(height, width, dtype=torch.int32, device=device)

    return {
        "means2d": means2d[0].contiguous(),  # [N, 2]
        "conics": conics[0].contiguous(),    # [N, 3]
        "opacities": opacities_flat,          # [N]
        "tile_offsets": tile_offsets,          # [tile_height, tile_width]
        "flatten_ids": flatten_ids,            # [n_isects]
        "render_alphas": render_alphas,        # [H, W]
        "last_ids": last_ids,                  # [H, W]
        "width": width,
        "height": height,
        "n_isects": int(flatten_ids.shape[0]),
    }

def compile_atomic_counter():
    """Compile the atomic instrumentation CUDA extension."""
    from torch.utils.cpp_extension import load
    print("Compiling atomic instrumentation kernel...")
    ext = load(
        name="atomic_instrument",
        sources=[ATOMIC_CU],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
        verbose=False,
    )
    print("Compiled successfully.")
    return ext

def count_atomics(ext, meta):
    """Run the atomic counter kernel."""
    n_warp = torch.zeros(1, dtype=torch.int64, device="cuda")
    n_block = torch.zeros(1, dtype=torch.int64, device="cuda")

    ext.count_backward_atomics(
        meta["means2d"],
        meta["conics"],
        meta["opacities"],
        meta["width"],
        meta["height"],
        16,  # tile_size
        meta["tile_offsets"],
        meta["flatten_ids"],
        meta["render_alphas"],
        meta["last_ids"],
        n_warp,
        n_block,
    )
    torch.cuda.synchronize()
    return int(n_warp.item()), int(n_block.item())

def benchmark_backward(state, cam, accutile, warmup=20, measure=100, device="cuda"):
    """Benchmark backward pass timing."""
    import gsplat
    from gsplat import rasterization

    means = state["xyz"].to(device).clone().requires_grad_(True)
    quats = state["rotations"].to(device).clone().requires_grad_(True)
    scales = torch.exp(state["scales"].to(device)).clone().requires_grad_(True)
    opacities = torch.sigmoid(state["opacity"].to(device).flatten()).clone().requires_grad_(True)
    colors = state["shs"].to(device).clone().requires_grad_(True)
    sh_degree = int(state.get("sh_degree", 3))

    viewmats, Ks, width, height = make_viewmat_K(cam, device)

    kwargs = dict(sh_degree=sh_degree, absgrad=True, tile_size=16, packed=False, render_mode="RGB")
    if accutile:
        kwargs["accutile"] = True

    # Warmup
    for _ in range(warmup):
        for t in (means, quats, scales, opacities, colors):
            t.grad = None
        rgb, alpha, meta = rasterization(means, quats, scales, opacities, colors,
                                          viewmats, Ks, width, height, **kwargs)
        (rgb.sum() + alpha.sum()).backward()
        torch.cuda.synchronize()

    fwd_times, bwd_times = [], []
    for _ in range(measure):
        for t in (means, quats, scales, opacities, colors):
            t.grad = None
        s = torch.cuda.Event(enable_timing=True)
        fe = torch.cuda.Event(enable_timing=True)
        be = torch.cuda.Event(enable_timing=True)
        s.record()
        rgb, alpha, meta = rasterization(means, quats, scales, opacities, colors,
                                          viewmats, Ks, width, height, **kwargs)
        fe.record()
        (rgb.sum() + alpha.sum()).backward()
        be.record()
        torch.cuda.synchronize()
        fwd_times.append(s.elapsed_time(fe))
        bwd_times.append(fe.elapsed_time(be))

    return {
        "forward_ms": float(np.mean(fwd_times)),
        "backward_ms": float(np.mean(bwd_times)),
        "total_ms": float(np.mean(fwd_times)) + float(np.mean(bwd_times)),
        "intersections": int(meta["tiles_per_gauss"].sum()),
    }

def main():
    device = "cuda"
    os.makedirs("/tmp/accutile_a100_results", exist_ok=True)

    # Compile instrumentation kernel
    ext = compile_atomic_counter()

    all_results = {}

    for scene, paths in SCENES.items():
        print(f"\n{'='*70}")
        print(f"Scene: {scene}")
        print(f"{'='*70}")

        state = load_checkpoint(paths["ckpt"])
        with open(paths["cams"]) as f:
            cams = json.load(f)

        n_cams = len(cams)
        cam_indices = [0, n_cams // 2, n_cams - 1][:NUM_CAMERAS]

        scene_results = []
        for ci in cam_indices:
            cam = cams[ci]
            print(f"\n  Camera {ci} ({cam['width']}x{cam['height']})")

            # Get forward metadata for B1 and B1A
            print("    B1 forward metadata...")
            meta_b1 = get_forward_metadata(state, cam, accutile=False, device=device)
            print(f"    B1: n_isects={meta_b1['n_isects']:,}")

            print("    B1A forward metadata...")
            meta_b1a = get_forward_metadata(state, cam, accutile=True, device=device)
            print(f"    B1A: n_isects={meta_b1a['n_isects']:,}")

            # Count atomics
            print("    Counting B1 atomics...")
            n_warp_b1, n_unique_b1 = count_atomics(ext, meta_b1)
            print(f"    B1: n_warp_atomics={n_warp_b1:,}, n_block_uniques={n_unique_b1:,}")

            print("    Counting B1A atomics...")
            n_warp_b1a, n_unique_b1a = count_atomics(ext, meta_b1a)
            print(f"    B1A: n_warp_atomics={n_warp_b1a:,}, n_block_uniques={n_unique_b1a:,}")

            # Compute metrics
            ATOMICS_PER_EVENT = 11  # 3 v_rgb + 3 v_conic + 2 v_means2d + 2 v_means2d_abs + 1 v_opacity
            r_atomic_b1 = n_warp_b1 * ATOMICS_PER_EVENT
            r_atomic_b1a = n_warp_b1a * ATOMICS_PER_EVENT
            dup_b1 = n_warp_b1 / max(n_unique_b1, 1)
            dup_b1a = n_warp_b1a / max(n_unique_b1a, 1)

            # Benchmark (camera 0 only to save time)
            if ci == 0:
                print("    Benchmarking B1...")
                bench_b1 = benchmark_backward(state, cam, accutile=False, device=device)
                print(f"    B1: fwd={bench_b1['forward_ms']:.2f}ms bwd={bench_b1['backward_ms']:.2f}ms")
                print("    Benchmarking B1A...")
                bench_b1a = benchmark_backward(state, cam, accutile=True, device=device)
                print(f"    B1A: fwd={bench_b1a['forward_ms']:.2f}ms bwd={bench_b1a['backward_ms']:.2f}ms")
            else:
                bench_b1 = None
                bench_b1a = None

            entry = {
                "cam_idx": ci,
                "n_isects_B1": meta_b1["n_isects"],
                "n_isects_B1A": meta_b1a["n_isects"],
                "isect_reduction": 1 - meta_b1a["n_isects"] / meta_b1["n_isects"],
                "n_warp_atomics_B1": n_warp_b1,
                "n_warp_atomics_B1A": n_warp_b1a,
                "n_block_uniques_B1": n_unique_b1,
                "n_block_uniques_B1A": n_unique_b1a,
                "R_atomic_direct_B1": r_atomic_b1,
                "R_atomic_direct_B1A": r_atomic_b1a,
                "duplication_factor_B1": dup_b1,
                "duplication_factor_B1A": dup_b1a,
                "atomic_reduction": 1 - r_atomic_b1a / r_atomic_b1 if r_atomic_b1 > 0 else 0,
            }
            if bench_b1:
                entry["benchmark_B1"] = bench_b1
                entry["benchmark_B1A"] = bench_b1a
                entry["bwd_time_reduction"] = 1 - bench_b1a["backward_ms"] / bench_b1["backward_ms"]
            scene_results.append(entry)

        all_results[scene] = scene_results

    # Summary
    print(f"\n{'='*70}")
    print("DIRECT ATOMIC INSTRUMENTATION SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Scene':<10} {'Cam':>5} {'N_isect B1':>14} {'N_isect B1A':>14} {'Isect Red':>10} "
          f"{'Warp Atom B1':>14} {'Warp Atom B1A':>14} {'Atom Red':>10} "
          f"{'Dup B1':>8} {'Dup B1A':>8}")
    for scene, entries in all_results.items():
        for e in entries:
            print(f"{scene:<10} {e['cam_idx']:>5} {e['n_isects_B1']:>14,} {e['n_isects_B1A']:>14,} "
                  f"{e['isect_reduction']:>10.4f} {e['n_warp_atomics_B1']:>14,} {e['n_warp_atomics_B1A']:>14,} "
                  f"{e['atomic_reduction']:>10.4f} {e['duplication_factor_B1']:>8.3f} {e['duplication_factor_B1A']:>8.3f}")

    # Backward scaling analysis
    print(f"\n{'='*70}")
    print("BACKWARD SCALING ANALYSIS (cam 0 only)")
    print(f"{'='*70}")
    print(f"\n{'Scene':<10} {'Isect Red':>10} {'Bwd Time Red':>12} {'Atom Red':>10} {'B1 bwd ms':>10} {'B1A bwd ms':>10} {'Ratio isect/bwd':>16}")
    for scene, entries in all_results.items():
        e = entries[0]  # cam 0
        if "benchmark_B1" in e:
            isect_red = e["isect_reduction"]
            bwd_red = e.get("bwd_time_reduction", 0)
            atom_red = e["atomic_reduction"]
            b1_bwd = e["benchmark_B1"]["backward_ms"]
            b1a_bwd = e["benchmark_B1A"]["backward_ms"]
            ratio = isect_red / max(bwd_red, 1e-8)
            print(f"{scene:<10} {isect_red:>10.4f} {bwd_red:>12.4f} {atom_red:>10.4f} {b1_bwd:>10.2f} {b1a_bwd:>10.2f} {ratio:>16.2f}")

    # Save
    out_path = "/tmp/accutile_a100_results/atomic_instrumentation.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
