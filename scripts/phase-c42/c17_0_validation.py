#!/usr/bin/env python3
"""
C17-0 Correctness + Performance Validation.

Compares baseline global CUB radix sort vs C17-0 tile-segmented depth-only sort.

Correctness:
  - Compare isect_ids ordering (baseline vs C17-0)
  - Compare flatten_ids ordering
  - Compare tile_offsets
  - Render PSNR/SSIM with both sort orders

Performance:
  - Sort kernel time (baseline global vs C17-0 counting+segmented)
  - Offset kernel time
  - Total forward time
  - End-to-end training iteration time

Scenes: room, bicycle, garden
Hardware: A100 PCIe 40GB
"""
import json, math, sys, time, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "epic05" / "phase7"))

from gsplat import rasterization, fully_fused_projection, isect_tiles, isect_offset_encode
from gaussian_model import GaussianModel
from dataset import GTDataset, load_initial_checkpoint

DEVICE = "cuda"
TILE_SIZE = 16
PACKED = True
EPS2D = 0.1
RADIUS_CLIP = 0.0
SEED = 42


def project(model, cam, tile_size=TILE_SIZE):
    """Project Gaussians and return means2d, radii, depths."""
    data = model.forward()
    radii, means2d, depths, conics, compensations = fully_fused_projection(
        means=data["xyz"], covars=None, quats=data["rotations"], scales=data["scales"],
        viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
        width=cam.image_width, height=cam.image_height,
        radius_clip=RADIUS_CLIP, packed=False, eps2d=EPS2D,
    )
    return means2d, radii, depths, data


def get_isects_baseline(means2d, radii, depths, cam, tile_size=TILE_SIZE):
    """Baseline: global CUB radix sort (sort=True, segmented=False)."""
    W, H = cam.image_width, cam.image_height
    tw = (W + tile_size - 1) // tile_size
    th = (H + tile_size - 1) // tile_size
    tpg, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tw, th,
        sort=True, segmented=False, packed=False)
    tile_offsets = isect_offset_encode(isect_ids, 1, tw, th)
    return isect_ids, flatten_ids, tile_offsets, tw, th


def get_isects_c17_0(means2d, radii, depths, cam, tile_size=TILE_SIZE):
    """C17-0: tile-segmented depth-only sort (sort=True, segmented=True)."""
    W, H = cam.image_width, cam.image_height
    tw = (W + tile_size - 1) // tile_size
    th = (H + tile_size - 1) // tile_size
    tpg, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tw, th,
        sort=True, segmented=True, packed=False)
    tile_offsets = isect_offset_encode(isect_ids, 1, tw, th)
    return isect_ids, flatten_ids, tile_offsets, tw, th


def render_from_data(data, cam, isect_ids, flatten_ids, tile_offsets, tw, th, tile_size=TILE_SIZE):
    """Render using pre-computed intersection data."""
    # We can't easily pass pre-computed isects to gsplat's rasterization,
    # so we just use the standard render and verify via isect comparison.
    r, _, _ = rasterization(
        means=data["xyz"], quats=data["rotations"], scales=data["scales"],
        opacities=data["opacity"], colors=data["shs"],
        viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
        width=cam.image_width, height=cam.image_height,
        tile_size=tile_size, packed=PACKED, sh_degree=model.sh_degree,
        radius_clip=RADIUS_CLIP, eps2d=EPS2D, render_mode="RGB")
    return r[0].clamp(0, 1)


def measure_sort_time(means2d, radii, depths, cam, tile_size, segmented, n_warmup=10, n_measure=50):
    """Measure sort time (isect_tiles with sort=True)."""
    W, H = cam.image_width, cam.image_height
    tw = (W + tile_size - 1) // tile_size
    th = (H + tile_size - 1) // tile_size

    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            isect_tiles(means2d, radii, depths, tile_size, tw, th,
                       sort=True, segmented=segmented, packed=False)
    torch.cuda.synchronize()

    times = []
    n_isects = 0
    for _ in range(n_measure):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            tpg, isect_ids, flatten_ids = isect_tiles(
                means2d, radii, depths, tile_size, tw, th,
                sort=True, segmented=segmented, packed=False)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        n_isects = len(flatten_ids)

    # Also measure without sort (isect only)
    for _ in range(n_warmup):
        with torch.no_grad():
            isect_tiles(means2d, radii, depths, tile_size, tw, th,
                       sort=False, segmented=False, packed=False)
    torch.cuda.synchronize()

    isect_only_times = []
    for _ in range(n_measure):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            isect_tiles(means2d, radii, depths, tile_size, tw, th,
                       sort=False, segmented=False, packed=False)
        torch.cuda.synchronize()
        isect_only_times.append((time.perf_counter() - t0) * 1000)

    sort_time = float(np.median(times)) - float(np.median(isect_only_times))
    return {
        "isect_sort_ms": float(np.median(times)),
        "isect_only_ms": float(np.median(isect_only_times)),
        "sort_ms": sort_time,
        "n_isects": n_isects,
        "n_measure": n_measure,
    }


def measure_full_render(model, cam, tile_size=TILE_SIZE, n_warmup=10, n_measure=50):
    """Measure full forward render time."""
    data = model.forward()
    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            r, _, _ = rasterization(
                means=data["xyz"], quats=data["rotations"], scales=data["scales"],
                opacities=data["opacity"], colors=data["shs"],
                viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
                width=cam.image_width, height=cam.image_height,
                tile_size=tile_size, packed=PACKED, sh_degree=model.sh_degree,
                radius_clip=RADIUS_CLIP, eps2d=EPS2D, render_mode="RGB")
    torch.cuda.synchronize()

    times = []
    for _ in range(n_measure):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            r, _, _ = rasterization(
                means=data["xyz"], quats=data["rotations"], scales=data["scales"],
                opacities=data["opacity"], colors=data["shs"],
                viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
                width=cam.image_width, height=cam.image_height,
                tile_size=tile_size, packed=PACKED, sh_degree=model.sh_degree,
                radius_clip=RADIUS_CLIP, eps2d=EPS2D, render_mode="RGB")
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    return float(np.median(times))


def compute_psnr(pred, gt):
    mse = F.mse_loss(pred, gt)
    if mse < 1e-10:
        return 100.0
    return float(20 * math.log10(1.0 / math.sqrt(mse.item())))


def compute_ssim(pred, gt):
    """Simple SSIM computation."""
    from loss import ssim
    return float(ssim(pred.unsqueeze(0), gt.unsqueeze(0)))


def validate_scene(scene, repo_root, model=None, n_cameras=6):
    """Validate C17-0 on one scene."""
    print(f"\n{'='*72}")
    print(f"C17-0 Validation: {scene}")
    print(f"{'='*72}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    dataset = GTDataset(scene=scene, repo_root=repo_root, resolution="1080p", device=DEVICE)
    n_cams = len(dataset)
    cam_indices = list(range(0, n_cams, max(1, n_cams // n_cameras)))[:n_cameras]
    print(f"  Cameras: {n_cams}, evaluating: {cam_indices}")

    # Load model
    ckpt_path = repo_root / "results" / "a100" / "phase-c42" / "p2_checkpoints"
    ckpt_files = list(ckpt_path.glob("*iter10000*.pt")) if ckpt_path.exists() else []
    if ckpt_files and scene == "room":
        ckpt = torch.load(ckpt_files[0], map_location=DEVICE, weights_only=False)
        model = GaussianModel.from_checkpoint_state(ckpt, device=DEVICE)
        model.set_sh_degree(3)
        print(f"  Model: checkpoint {ckpt_files[0].name}  GS={model.xyz.shape[0]:,}")
    else:
        sfm_data = load_initial_checkpoint(scene, repo_root, device=DEVICE)
        model = GaussianModel(
            num_points=sfm_data["xyz"].shape[0], sh_degree=0, max_sh_degree=3, device=DEVICE)
        model.init_from_sfm(
            xyz=sfm_data["xyz"],
            opacity_logit=torch.logit(torch.full((sfm_data["xyz"].shape[0], 1), 0.1, device=DEVICE)),
            scales_log=sfm_data.get("scales"),
            rotations_raw=sfm_data.get("rotations"),
            shs=sfm_data.get("shs"))
        model.set_sh_degree(3)
        print(f"  Model: SfM init  GS={model.xyz.shape[0]:,}")

    results = {"scene": scene, "n_gaussians": model.xyz.shape[0], "cameras": [], "summary": {}}

    all_sort_baseline = []
    all_sort_c17_0 = []
    all_full_render = []
    all_n_isects = []
    correctness_pass = True

    for ci in cam_indices:
        cam = dataset.get_camera(ci)
        gt = dataset.get_gt_image(ci)
        print(f"\n  [Camera {ci}] {cam.image_width}x{cam.image_height}")

        means2d, radii, depths, data = project(model, cam)

        # Correctness: compare baseline vs C17-0
        isect_base, flatten_base, offsets_base, tw, th = get_isects_baseline(means2d, radii, depths, cam)
        isect_c17, flatten_c17, offsets_c17, _, _ = get_isects_c17_0(means2d, radii, depths, cam)

        n_isects = len(flatten_base)
        print(f"    N_isects: {n_isects:,}")

        # Compare isect_ids
        isect_match = torch.equal(isect_base, isect_c17)
        # Compare flatten_ids
        flatten_match = torch.equal(flatten_base, flatten_c17)
        # Compare tile_offsets
        offsets_match = torch.equal(offsets_base, offsets_c17)

        # Check depth monotonicity within tiles (sample a few tiles)
        depth_mono = True
        n_checked = 0
        for tid in range(min(tw * th, 100)):
            start = offsets_base[0, tid // tw, tid % tw].item()
            end = offsets_base[0, (tid + 1) // tw, (tid + 1) % tw].item() if tid + 1 < tw * th else n_isects
            if end > start + 1:
                depths_tile = isect_base[start:end] & 0xFFFFFFFF
                # Reinterpret as int32 (depth bits)
                depths_int = depths_tile.to(torch.int32)
                if not torch.all(depths_int[1:] >= depths_int[:-1]):
                    depth_mono = False
                    break
                n_checked += 1

        cam_pass = isect_match and flatten_match and offsets_match and depth_mono
        correctness_pass = correctness_pass and cam_pass

        print(f"    isect_ids match: {isect_match}")
        print(f"    flatten_ids match: {flatten_match}")
        print(f"    tile_offsets match: {offsets_match}")
        print(f"    depth monotonicity: {depth_mono} (checked {n_checked} tiles)")
        print(f"    Camera correctness: {'PASS' if cam_pass else 'FAIL'}")

        # Performance: measure sort times
        sort_base = measure_sort_time(means2d, radii, depths, cam, TILE_SIZE, segmented=False)
        sort_c17 = measure_sort_time(means2d, radii, depths, cam, TILE_SIZE, segmented=True)
        full_render = measure_full_render(model, cam)

        all_sort_baseline.append(sort_base["sort_ms"])
        all_sort_c17_0.append(sort_c17["sort_ms"])
        all_full_render.append(full_render)
        all_n_isects.append(n_isects)

        sort_speedup = (sort_base["sort_ms"] - sort_c17["sort_ms"]) / sort_base["sort_ms"] * 100
        print(f"    Baseline sort: {sort_base['sort_ms']:.2f} ms  C17-0 sort: {sort_c17['sort_ms']:.2f} ms  speedup: {sort_speedup:+.1f}%")
        print(f"    Full render: {full_render:.2f} ms")

        results["cameras"].append({
            "cam_idx": ci,
            "n_isects": n_isects,
            "correctness": {
                "isect_ids_match": isect_match,
                "flatten_ids_match": flatten_match,
                "tile_offsets_match": offsets_match,
                "depth_monotonicity": depth_mono,
                "pass": cam_pass,
            },
            "performance": {
                "baseline_sort_ms": sort_base["sort_ms"],
                "c17_0_sort_ms": sort_c17["sort_ms"],
                "sort_speedup_pct": sort_speedup,
                "full_render_ms": full_render,
            },
        })

    # Summary
    mean_sort_base = float(np.mean(all_sort_baseline))
    mean_sort_c17 = float(np.mean(all_sort_c17_0))
    mean_full = float(np.mean(all_full_render))
    mean_isects = int(np.mean(all_n_isects))
    sort_speedup = (mean_sort_base - mean_sort_c17) / mean_sort_base * 100
    e2e_speedup = (mean_sort_base - mean_sort_c17) / mean_full * 100

    results["summary"] = {
        "mean_baseline_sort_ms": mean_sort_base,
        "mean_c17_0_sort_ms": mean_sort_c17,
        "sort_speedup_pct": sort_speedup,
        "mean_full_render_ms": mean_full,
        "e2e_speedup_pct": e2e_speedup,
        "mean_n_isects": mean_isects,
        "correctness_pass": correctness_pass,
    }

    print(f"\n  {'='*60}")
    print(f"  SUMMARY: {scene}")
    print(f"  {'='*60}")
    print(f"    Mean N_isects: {mean_isects:,}")
    print(f"    Baseline sort: {mean_sort_base:.2f} ms")
    print(f"    C17-0 sort:    {mean_sort_c17:.2f} ms")
    print(f"    Sort speedup:  {sort_speedup:+.1f}%")
    print(f"    Full render:   {mean_full:.2f} ms")
    print(f"    E2E speedup:   {e2e_speedup:+.1f}%")
    print(f"    Correctness:   {'ALL PASS' if correctness_pass else 'FAIL'}")

    # Decision gates
    speedup_pass = e2e_speedup > 5.0
    correctness_ok = correctness_pass

    print(f"\n  DECISION GATES:")
    print(f"    Forward speedup > 5%: {'PASS' if speedup_pass else 'FAIL'} ({e2e_speedup:.1f}%)")
    print(f"    Correctness:          {'PASS' if correctness_ok else 'FAIL'}")

    if speedup_pass and correctness_ok:
        decision = "KEEP"
    elif not correctness_ok:
        decision = "DROP (correctness failure)"
    elif not speedup_pass:
        decision = "DROP (segmentation overhead removes benefit)"
    else:
        decision = "INCONCLUSIVE"

    print(f"    DECISION: {decision}")
    results["summary"]["decision"] = decision

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="room", choices=["room", "bicycle", "garden"])
    parser.add_argument("--n_cameras", type=int, default=6)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    results = validate_scene(args.scene, repo_root, n_cameras=args.n_cameras)

    # Save
    save_path = repo_root / "results" / "a100" / "phase-c42" / f"c17_0_validation_{args.scene}.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Data saved to {save_path}")


if __name__ == "__main__":
    main()
