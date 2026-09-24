#!/usr/bin/env python3
"""
R3-D0 — Rasterization stall localization smoke test.

gsplat 1.5.3 API probe: does rasterize_to_pixels stall on real-scene data?

Strategy:
  1. Tiny synthetic control (32×32, 100-1000 Gaussians)
  2. Real Room checkpoint, one camera, one call
  3. Wrapper vs raw _C.rasterize_to_pixels_3dgs_fwd

ENV:
  CUDA_LAUNCH_BLOCKING=1
  TORCH_SHOW_CPP_STACKTRACES=1
"""

import os, sys, time, math, faulthandler, json

faulthandler.enable()
faulthandler.dump_traceback_later(120, repeat=True)  # dump every 2 min

import torch
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(SCRIPT_DIR, "..", "..")
BASELINE_DIR = os.path.join(REPO_ROOT, "baseline", "reference_v1")

# Order matters: baseline/reference_v1 gaussian_model.py does NOT require num_points
# phase7's version requires num_points — don't import it
sys.path.insert(0, BASELINE_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from gaussian_model import GaussianModel  # from baseline/reference_v1
from config import ReferenceV1Config

from gsplat.cuda._wrapper import (
    fully_fused_projection,
    spherical_harmonics,
    isect_tiles,
    isect_offset_encode,
    rasterize_to_pixels,
)
from gsplat.cuda import _backend as gsplat_backend
_C = gsplat_backend._C

torch.set_printoptions(linewidth=200, sci_mode=False)
np.set_printoptions(linewidth=200)

RESULTS = {}

# ----------------------------------------------------------------
def print_tensor_meta(name, t):
    if t is None:
        print(f"  {name}: None")
        return
    print(f"  {name}:")
    print(f"    shape={list(t.shape)} dtype={t.dtype} device={t.device} "
          f"cuda={t.is_cuda} contig={t.is_contiguous()} "
          f"grad={t.requires_grad} numel={t.numel()}")
    print(f"    stride={t.stride()}")

# ----------------------------------------------------------------
def run_tiny_synthetic(device="cuda", tile_size=16):
    """Tiny synthetic control: 64×64, ~100-1000 Gaussians."""
    print("\n" + "=" * 70)
    print("D0-TINY: Synthetic control (64×64, 256 Gaussians)")
    print("=" * 70)

    H, W = 64, 64
    tile_w = (W + tile_size - 1) // tile_size
    tile_h = (H + tile_size - 1) // tile_size
    N = 256

    torch.manual_seed(42)
    xyz = torch.randn(1, N, 3, device=device) * 0.3
    quats = torch.randn(1, N, 4, device=device)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    scales = torch.randn(1, N, 3, device=device).exp() * 0.05
    opacities = torch.randn(N, device=device).sigmoid()
    shs = torch.zeros(1, N, 16, 3, device=device)

    viewmat = torch.eye(4, device=device).unsqueeze(0)
    K = torch.tensor([[W, 0, W/2], [0, W, H/2], [0, 0, 1]], device=device, dtype=torch.float32).unsqueeze(0)

    # Project (gsplat 1.5.3: means=[N,3], viewmats=[C,4,4])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    radii, means2d, depths, conics, compensations = fully_fused_projection(
        xyz[0], None, quats[0], scales[0], viewmat, K, W, H, eps2d=0.1
    )
    # Output is [C, N, ...] with C=1
    torch.cuda.synchronize()
    print(f"  [TINY] projection: {time.perf_counter()-t0:.3f}s")

    # SH (gsplat 1.5.3: dirs=[C,N,3], shs=[C,N,K,3])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    dirs = xyz - viewmat[:, :3, 3].unsqueeze(1)  # [1, N, 3]
    colors = spherical_harmonics(3, dirs, shs)
    torch.cuda.synchronize()
    print(f"  [TINY] SH: {time.perf_counter()-t0:.3f}s")

    # Intersect
    with torch.no_grad():
        tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
            means2d, radii, depths, tile_size, tile_w, tile_h, sort=True
        )
        isect_offsets = isect_offset_encode(isect_ids, 1, tile_w, tile_h)

    opacities_in = opacities.detach().clone().unsqueeze(0)

    n_isects = flatten_ids.shape[0]
    print(f"  [TINY] n_isects={n_isects} tile_h={tile_h} tile_w={tile_w}")
    print(f"  [TINY] isect_offsets shape={list(isect_offsets.shape)}")
    print(f"  [TINY] flatten_ids shape={list(flatten_ids.shape)}")

    # ---- TINY-WRAPPER ----
    print("\n  [TINY-WRAPPER] calling rasterize_to_pixels...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    print("    D0-A BEFORE WRAPPER", flush=True)
    try:
        render, render_alpha = rasterize_to_pixels(
            means2d, conics, colors, opacities_in,
            W, H, tile_size, isect_offsets, flatten_ids,
            backgrounds=None, masks=None,
            packed=False, absgrad=True,
        )
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f"    D0-B AFTER WRAPPER: {dt:.3f}s", flush=True)
        print(f"    render shape={list(render.shape)}")
        TINY_WRAPPER = "PASS"
    except Exception as e:
        dt = time.perf_counter() - t0
        print(f"    D0-B FAILED: {e} ({dt:.3f}s)", flush=True)
        TINY_WRAPPER = f"FAIL: {e}"

    # ---- TINY-RAW ----
    print("\n  [TINY-RAW] calling _C.rasterize_to_pixels_3dgs_fwd...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    print("    D0-C BEFORE RAW FWD", flush=True)
    try:
        renders, alphas, last_ids = _C.rasterize_to_pixels_3dgs_fwd(
            means2d.contiguous(),
            conics.contiguous(),
            colors.contiguous(),
            opacities_in.contiguous(),
            None,  # backgrounds
            None,  # masks
            W, H, tile_size,
            isect_offsets.contiguous(),
            flatten_ids.contiguous(),
        )
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f"    D0-D AFTER RAW FWD: {dt:.3f}s", flush=True)
        print(f"    renders shape={list(renders.shape)}")
        TINY_RAW = "PASS"
    except Exception as e:
        dt = time.perf_counter() - t0
        print(f"    D0-D FAILED: {e} ({dt:.3f}s)", flush=True)
        TINY_RAW = f"FAIL: {e}"

    RESULTS["TINY_WRAPPER"] = TINY_WRAPPER
    RESULTS["TINY_RAW"] = TINY_RAW
    return TINY_WRAPPER, TINY_RAW


# ----------------------------------------------------------------
def run_real_scene(device="cuda", tile_size=16):
    """One real Room camera, one raster call. No W_it replay."""
    print("\n" + "=" * 70)
    print("D0-REAL: Room 30K checkpoint, single camera")
    print("=" * 70)

    checkpoint_path = os.path.join(
        REPO_ROOT, "results", "reference_v1", "room_30k", "checkpoints", "iter_5000.pt"
    )
    if not os.path.exists(checkpoint_path):
        RESULTS["REAL_WRAPPER"] = "SKIP (checkpoint not found)"
        RESULTS["REAL_RAW"] = "SKIP (checkpoint not found)"
        print(f"  Checkpoint not found: {checkpoint_path}")
        return "SKIP", "SKIP"

    config = ReferenceV1Config(scene="room", iterations=30000)

    print("  Loading checkpoint...")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = GaussianModel(max_sh_degree=config.sh_degree)
    model.restore(ckpt, {
        "position_lr_init": config.position_lr_init,
        "position_lr_final": config.position_lr_final,
        "position_lr_delay_mult": config.position_lr_delay_mult,
        "position_lr_max_steps": config.position_lr_max_steps,
        "feature_lr": config.feature_lr,
        "opacity_lr": config.opacity_lr,
        "scaling_lr": config.scaling_lr,
        "rotation_lr": config.rotation_lr,
        "percent_dense": config.percent_dense,
    })
    N = model._xyz.shape[0]
    print(f"  Model loaded: N={N}")

    # Load camera directly (avoid GTDataset which pulls in phase7 gaussian_model)
    from pathlib import Path
    from benchmark_framework import load_cameras_from_json, resize_cameras
    scene_dir = Path(REPO_ROOT) / "data" / "official" / "mipnerf360" / "room"
    cameras = load_cameras_from_json(str(scene_dir / "cameras.json"), device="cpu")
    cameras = resize_cameras(cameras, 1920, 1080)

    cam_sequence = np.load(os.path.join(REPO_ROOT, "data", "camera_sequence.npy"))
    cam_idx = int(cam_sequence[5000])  # first camera
    cam = cameras[cam_idx]

    # Move camera tensors to device
    for attr in ["viewmatrix", "projmatrix", "camera_center", "world_view_transform",
                  "full_proj_transform", "K"]:
        t = getattr(cam, attr, None)
        if isinstance(t, torch.Tensor):
            setattr(cam, attr, t.to(device))

    H, W = cam.image_height, cam.image_width
    print(f"  Camera: idx={cam_idx}, H={H}, W={W}")

    # ---- Extract params ----
    xyz = model.get_xyz
    quats = model.get_rotation
    scales = model.get_scaling
    opacities = model.get_opacity
    shs = model.get_features

    # ---- Projection ----
    print("\n  [PROJECTION] ...", flush=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    radii, means2d, depths, conics, compensations = fully_fused_projection(
        xyz, None, quats, scales,
        cam.viewmatrix.unsqueeze(0), cam.K.unsqueeze(0),
        W, H, eps2d=0.1
    )
    torch.cuda.synchronize()
    print(f"  [PROJECTION] done: {time.perf_counter()-t0:.3f}s")
    print(f"    radii={list(radii.shape)} means2d={list(means2d.shape)}")

    # ---- SH ----
    print("\n  [SH] ...", flush=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    dirs = xyz.unsqueeze(0) - cam.camera_center.unsqueeze(0)
    colors = spherical_harmonics(model.active_sh_degree, dirs, shs.unsqueeze(0))
    torch.cuda.synchronize()
    print(f"  [SH] done: {time.perf_counter()-t0:.3f}s")
    print(f"    colors={list(colors.shape)}")

    # ---- Intersection ----
    tile_w = (W + tile_size - 1) // tile_size
    tile_h = (H + tile_size - 1) // tile_size

    print("\n  [INTERSECTION] ...", flush=True)
    with torch.no_grad():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
            means2d, radii, depths, tile_size, tile_w, tile_h, sort=True
        )
        isect_offsets = isect_offset_encode(isect_ids, 1, tile_w, tile_h)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f"  [INTERSECTION] done: {dt:.3f}s")
        n_isects = flatten_ids.shape[0]
        print(f"    n_isects={n_isects} tile_h={tile_h} tile_w={tile_w}")
        print(f"    isect_offsets shape={list(isect_offsets.shape)}")
        print(f"    flatten_ids shape={list(flatten_ids.shape)}")

    opacities_in = opacities.detach().clone().unsqueeze(0)

    # ---- Print input metadata (D0 section 2) ----
    print("\n  [INPUT METADATA]")
    print_tensor_meta("means2d", means2d)
    print_tensor_meta("conics", conics)
    print_tensor_meta("colors", colors)
    print_tensor_meta("opacities", opacities_in)
    print_tensor_meta("isect_offsets", isect_offsets)
    print_tensor_meta("flatten_ids", flatten_ids)
    print(f"    W={W} H={H} tile_size={tile_size} tile_w={tile_w} tile_h={tile_h} n_isects={n_isects}")

    # ---- Validate intersection structure (D0 section 3) ----
    print("\n  [OFFSET VALIDATION]")
    assert isect_offsets.dtype == torch.int32, f"isect_offsets dtype: {isect_offsets.dtype}"
    assert flatten_ids.dtype == torch.int32, f"flatten_ids dtype: {flatten_ids.dtype}"
    assert isect_offsets.is_cuda, "isect_offsets not on CUDA"
    assert flatten_ids.is_cuda, "flatten_ids not on CUDA"
    assert means2d.is_contiguous(), "means2d not contiguous"
    assert conics.is_contiguous(), "conics not contiguous"
    assert colors.is_contiguous(), "colors not contiguous"
    assert opacities_in.is_contiguous(), "opacities not contiguous"
    assert isect_offsets.is_contiguous(), "isect_offsets not contiguous"
    assert flatten_ids.is_contiguous(), "flatten_ids not contiguous"
    print("    dtype (int32): OK")
    print("    is_cuda: OK")
    print("    is_contiguous: OK")

    offs = isect_offsets[0].reshape(-1)
    offs_cpu = offs.cpu()
    flat_cpu = flatten_ids.cpu()

    print(f"    offset min={int(offs_cpu.min())} max={int(offs_cpu.max())} n_isects={n_isects}")
    print(f"    first 20 offsets: {offs_cpu[:20].tolist()}")
    print(f"    last 20 offsets: {offs_cpu[-20:].tolist()}")
    print(f"    flatten_ids min={int(flat_cpu.min())} max={int(flat_cpu.max())}")

    # Monotonicity check
    diffs = offs_cpu[1:] - offs_cpu[:-1]
    non_mono = (diffs < 0).sum().item()
    print(f"    non-monotonic offset pairs: {non_mono}")

    # flatten_ids legal range: for non-packed, they encode global idx in [C*N]
    max_global = C = 1
    N_gauss = means2d.shape[1]
    legal_min = 0
    legal_max = C * N_gauss - 1
    illegal_ids = ((flat_cpu < legal_min) | (flat_cpu >= C * N_gauss)).sum().item()
    print(f"    flatten_ids legal range: [{legal_min}, {legal_max}]")
    print(f"    illegal flatten_ids: {illegal_ids}")
    OFFSETS_VALID = "YES" if non_mono == 0 and illegal_ids == 0 else "NO"
    print(f"    OFFSETS_VALID: {OFFSETS_VALID}")

    INPUT_METADATA_VALID = "YES" if OFFSETS_VALID == "YES" else "NO"
    RESULTS["INPUT_METADATA_VALID"] = INPUT_METADATA_VALID
    RESULTS["OFFSETS_VALID"] = OFFSETS_VALID
    RESULTS["n_isects"] = int(n_isects)
    RESULTS["tile_h"] = int(tile_h)
    RESULTS["tile_w"] = int(tile_w)
    RESULTS["N"] = int(N_gauss)

    # ---- REAL-WRAPPER ----
    print("\n  [REAL-WRAPPER] calling rasterize_to_pixels...", flush=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    print("    D0-A BEFORE WRAPPER", flush=True)
    try:
        render, render_alpha = rasterize_to_pixels(
            means2d, conics, colors, opacities_in,
            W, H, tile_size, isect_offsets, flatten_ids,
            backgrounds=None, masks=None,
            packed=False, absgrad=True,
        )
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f"    D0-B AFTER WRAPPER: {dt:.3f}s", flush=True)
        print(f"    render shape={list(render.shape)}")
        REAL_WRAPPER = "PASS"
    except Exception as e:
        import traceback
        dt = time.perf_counter() - t0
        traceback.print_exc()
        print(f"    D0-B FAILED: {e} ({dt:.3f}s)", flush=True)
        REAL_WRAPPER = f"FAIL: {e}"

    # ---- REAL-RAW ----
    print("\n  [REAL-RAW] calling _C.rasterize_to_pixels_3dgs_fwd...", flush=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    print("    D0-C BEFORE RAW FWD", flush=True)
    try:
        renders, alphas, last_ids = _C.rasterize_to_pixels_3dgs_fwd(
            means2d.contiguous(),
            conics.contiguous(),
            colors.contiguous(),
            opacities_in.contiguous(),
            None,  # backgrounds
            None,  # masks
            W, H, tile_size,
            isect_offsets.contiguous(),
            flatten_ids.contiguous(),
        )
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f"    D0-D AFTER RAW FWD: {dt:.3f}s", flush=True)
        print(f"    renders shape={list(renders.shape)}")
        REAL_RAW = "PASS"
    except Exception as e:
        import traceback
        dt = time.perf_counter() - t0
        traceback.print_exc()
        print(f"    D0-D FAILED: {e} ({dt:.3f}s)", flush=True)
        REAL_RAW = f"FAIL: {e}"

    RESULTS["REAL_WRAPPER"] = REAL_WRAPPER
    RESULTS["REAL_RAW"] = REAL_RAW
    return REAL_WRAPPER, REAL_RAW


# ----------------------------------------------------------------
def main():
    device = "cuda"
    tile_size = 16

    # Stage 1: Tiny synthetic
    tiny_wrapper, tiny_raw = run_tiny_synthetic(device, tile_size)

    # Stage 2: Real scene (only if tiny passes)
    real_wrapper = "SKIP"
    real_raw = "SKIP"
    if tiny_wrapper == "PASS":
        real_wrapper, real_raw = run_real_scene(device, tile_size)
    else:
        print("\n  Skipping real-scene test (tiny did not pass).")

    # ---- Final report ----
    print("\n" + "=" * 70)
    print("D0 FINAL REPORT")
    print("=" * 70)
    print(f"  TINY_WRAPPER = {RESULTS.get('TINY_WRAPPER', 'N/A')}")
    print(f"  TINY_RAW = {RESULTS.get('TINY_RAW', 'N/A')}")
    print(f"  REAL_WRAPPER = {RESULTS.get('REAL_WRAPPER', 'N/A')}")
    print(f"  REAL_RAW = {RESULTS.get('REAL_RAW', 'N/A')}")
    print(f"  INPUT_METADATA_VALID = {RESULTS.get('INPUT_METADATA_VALID', 'N/A')}")
    print(f"  OFFSETS_VALID = {RESULTS.get('OFFSETS_VALID', 'N/A')}")
    print(f"  n_isects = {RESULTS.get('n_isects', 'N/A')}")
    print(f"  tile_h = {RESULTS.get('tile_h', 'N/A')}")
    print(f"  tile_w = {RESULTS.get('tile_w', 'N/A')}")
    print(f"  N = {RESULTS.get('N', 'N/A')}")

    all_pass = (
        tiny_wrapper == "PASS" and tiny_raw == "PASS"
        and real_wrapper == "PASS" and real_raw == "PASS"
    )
    r3_ready = all_pass
    print(f"\n  R3_READY_TO_RESUME = {'YES' if r3_ready else 'NO'}")

    # Save results JSON
    out_path = os.path.join(SCRIPT_DIR, "d0_result.json")
    with open(out_path, "w") as f:
        json.dump(RESULTS, f, indent=2)
    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
