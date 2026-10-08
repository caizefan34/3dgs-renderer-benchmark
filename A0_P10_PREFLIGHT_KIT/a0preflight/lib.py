"""A0-P10 preflight shared library (remote EPIC-05 A100).

Scope (per user directive + A0_v2 prereg, REVISED per lead review):
  - Remote Stage-0 identity/hash verification of the frozen P12/C0 stack,
    including BYTE-IDENTITY of the authoritative _moonshot kernel sources vs the
    remote frozen C0 files, the room@25K checkpoint sha, the core/c0-ext .so sha,
    and the frozen camera index 19 img_name == DSCF4819.
  - One frozen training camera: room @ 25K, canonical index 19 (DSCF4819).
  - Exact native C0 backward coefficient walk (CPU, float32) to export
    PRIMARY q_alpha / q_color sufficient state for NATIVE tile x batch buckets.
  - Top-32 NATIVE bucket objects ranked ONLY by native accepted-hit count.
  - Preflight sanity checks (NOT full A0-v2 native-gradient parity) + oracle-SVD
    structural analyzer + frozen mechanical P10 verdict (oracle SVD only).

This is a PREFLIGHT/structure test. No DCT/VJP reconstruction, no bicycle/20K,
no full campaign, no production CUDA optimization. P10 checks are named
"preflight sanity"; the full reconstructed-VJP parity is left to formal A0.

The frozen C0 blend-backward math is transcribed EXACTLY from the certified
sources (see A0_EXACT_BACKWARD_FACTORIZATION.md):
  _moonshot/RasterizeToPixels3DGSDevice.cuh :: rasterize_to_pixels_3dgs_blend_bwd
  _moonshot/HigsNativeBackward.cu            :: higs_blend_bwd_px_kernel (PX=2, frozen arm)
  _moonshot/Common.h (== p18_p0/c0_source/Common.h) :: float constants
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys

# ---------------------------------------------------------------------------
# Frozen paths (EPIC-05 A100). These are the exact paths the frozen P12/C0
# training stack used (bootstrap_c0 in the frozen P12 trainer).
# ---------------------------------------------------------------------------
REPO = "/root/3dgs-renderer-benchmark"
C0_WT = "/root/epic05_frozen/higs_c0_final30k_worktree"
C0_SO = ("/root/epic05_build_cache/c0_ext/experimental_gaussian_render_inference_scene_cuda/"
         "experimental_gaussian_render_inference_scene_cuda.so")
CORE_SO = "/root/epic05_build_cache/core/gsplat_cuda/gsplat_cuda.so"
SCENE_CUDA_DIR = "/root/epic05_build_cache/scene/gsplat_scene_cuda"
PHASE7_DIR = os.path.join(REPO, "scripts", "epic05", "phase7")
SRC_DIR = os.path.join(REPO, "src")
REFERENCE_V1_DIR = os.path.join(REPO, "baseline", "reference_v1")
CKPT_DIR = "/root/epic05_n1/runs/n1snap_c0f_s42_{scene}"
FUSED_SSIM_SO = ("/root/miniforge3/envs/gsplat/lib/python3.10/site-packages/"
                 "fused_ssim_cuda.cpython-310-x86_64-linux-gnu.so")
FUSED_SSIM_MOD = "/root/miniforge3/envs/gsplat/lib/python3.10/site-packages/fused_ssim/__init__.py"
P12_TRAINER = "/root/epic05_p12/publication_trainer_p12.py"
PY = "/root/miniforge3/envs/gsplat/bin/python"

# Remote frozen C0 kernel source files (for Stage-0 byte-identity asserts).
# NOTE: HigsNativeBackward.cu lives in the EXPERIMENTAL kernels tree (not the main
# gsplat/cuda/csrc), while RasterizeToPixels3DGSDevice.cuh is in the main csrc.
# Both are byte-identical to the _moonshot/ authoritative reference (verified on
# EPIC-05). If a path is missing on a given host, the byte-identity check degrades
# to a recorded (non-fatal) note rather than a hard blocker.
REMOTE_KERNEL_CU = os.path.join(
    C0_WT, "gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference/HigsNativeBackward.cu")
REMOTE_KERNEL_CUH = os.path.join(C0_WT, "gsplat/cuda/csrc/RasterizeToPixels3DGSDevice.cuh")
# Authoritative local reference (byte-identical to the remote frozen C0 files;
# _moonshot/ and p18_p0/c0_source/ are sha-identical for both).
LOCAL_KERNEL_CU = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "..", "_moonshot", "HigsNativeBackward.cu")
LOCAL_KERNEL_CUH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "_moonshot", "RasterizeToPixels3DGSDevice.cuh")

# Frozen identity hashes (from the certified P12/MG0/Q0 identity records + lead review).
FROZEN_SHA256 = {
    # Core .so (certified C0 lineage; matches P12_SOURCE_IDENTITY c0_lineage.core_so).
    "core_so": "935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e",
    # C0 ext .so (P12 publication lineage = the binaries P12A/P12B were produced with).
    "c0_ext_so": "d26cccf46c7932bfb5fd44ee05e553af27e17c93eaffb7dacf3197258a513f11",
    # Fused-SSIM loss backend (P11-locked).
    "fused_ssim_so": "dd6a4c799b98ff9fa82b4203acce3b36b1529c97155c0ef4cd50fd0f583d8ebb",
    # P12 trainer + frozen recipe sources.
    "p12_trainer": "675b0267f3a1a6a0fa63fadb37c8536481c1c7a6de37a4b7f6ffa98a449050c5",
    "gaussian_model": "68731e375013a6f24aa4d81e5986402b1ac50f51cddf20688e358be705b1a60b",
    "config": "ea8e870476a3ba71c0ae9e1123a6fa5b1cf696bad983bdb6535d6fd04b7acc56",
    "colmap_reader": "b30a2a345f7f37cdf3087298750d714d7117e1e93ad742e766075be304729b1b",
    "phase7_dataset": "e6d8bfb1034bee102abc176060a8c7783951d253426ea7b5ea27daa16c258402",
    "higs_renderer_py": "d599aae3c1864993934d65646bfcb33e07e6046e389f028230752200df0c3a8a",
    # Room/25K checkpoint (lead review provides the authoritative sha).
    "ckpt_room_25000": "b31e5a6b0e733911b736bbad2e4f9d4d804d5929655f64efb51a9e5132fc047d",
    # Room/20K checkpoint (reference; from MG0 identity record).
    "ckpt_room_20000": "c4e41cfef79e50980406fc1fa148af8d04bc8b37b801c36da5cddb2ce8058718",
    # Authoritative frozen kernel sources (local reference; asserted byte-identical
    # to the remote frozen C0 files).
    "kernel_cu": "6efb9d258b4f40539bcb",      # prefix recorded; full sha computed at runtime
    "kernel_cuh": "603d203a8ad675eb55e1",     # prefix recorded; full sha computed at runtime
}

# Frozen C0 recipe (P12/C0).
RECIPE = {
    "arm": "c0",
    "seed": 42,
    "iterations": 30000,
    "loss": "fused",
    "loss_formula": "L = 0.8*L1 + 0.2*(1 - SSIM_fused[padding=same, train=True])",
    "lambda_dssim": 0.2,
    "eps2d": 0.3,
    "sh_degree": 3,
    "tile_size": 16,
    "packed": False,
    "radius_clip": 0.0,
    "render_mode": "RGB",
    "background": "black",
    "absgrad": True,
    "rasterization_mode": "higs_dynamic",
    "backward_mode": "higs_native",
    "camera_model": "pinhole",
    "enable_culling": True,
    "HIGS_PX_RUNTIME": "2",
    "HIGS_BWD_SCALAR_ADJOINT": "scalar_adjoint",
    "HIGS_BWD_H8_MR": "1",
    "HIGS_BWD_ABSGRAD": "1",
    "HIGS_DISABLE_F9": None,  # C0 = F9 ON (env unset)
    "resolution": "1080p",
    "target_wh": [1920, 1080],
}

# Frozen camera rule (Q0): 8 cameras, id_i = round(N*(i+0.5)/8), canonical order.
def frozen_camera_ids(n_cams: int, k: int = 8):
    return [int(round(n_cams * (i + 0.5) / k)) for i in range(k)]

# P10 single frozen training camera: room @ 25K, first frozen camera (index 19).
P10_SCENE = "room"
P10_CKPT = 25000
P10_CAM_INDEX = 19
P10_CAM_IMAGE = "DSCF4819"
P10_TOPK = 32
P10_SEED_UNIFORM = 20260929  # (unused for P10 top-k-only; reserved for full campaign)

# ---------------------------------------------------------------------------
# Frozen C0 float constants (AUTHORITATIVE: _moonshot/Common.h == p18_p0/c0_source/Common.h).
#   #define ALPHA_THRESHOLD         (1.f / 255.f)
#   #define MAX_ALPHA               0.99f
#   #define TRANSMITTANCE_THRESHOLD 1e-4f
#   #define MIN_ONE_MINUS_ALPHA     1e-6f
# ---------------------------------------------------------------------------
ALPHA_THRESHOLD = 1.0 / 255.0
MAX_ALPHA = 0.99
TRANSMITTANCE_THRESHOLD = 1e-4
MIN_ONE_MINUS_ALPHA = 1e-6


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_ext_module(name, so_path):
    spec = importlib.util.spec_from_file_location(name, so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[name] = mod
    return mod


def json_dump(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Bootstrap (mirrors bootstrap_c0 in the frozen P12 trainer, verbatim order)
# ---------------------------------------------------------------------------
def bootstrap_c0():
    """Bind the frozen C0 .so modules exactly as the frozen P12 trainer does."""
    os.environ["HIGS_PX_RUNTIME"] = "2"
    os.environ["HIGS_BWD_SCALAR_ADJOINT"] = "scalar_adjoint"
    os.environ["HIGS_BWD_H8_MR"] = "1"
    os.environ["HIGS_BWD_ABSGRAD"] = "1"
    os.environ.pop("HIGS_DISABLE_F9", None)
    for p in (C0_WT, SCENE_CUDA_DIR, PHASE7_DIR, SRC_DIR, REFERENCE_V1_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)
    core = load_ext_module("gsplat_cuda", CORE_SO)
    sys.modules["gsplat.csrc"] = core
    exp = load_ext_module("experimental_gaussian_render_inference_scene_cuda", C0_SO)
    sys.modules["gsplat.experimental.render.kernels.csrc"] = exp
    sys.modules["experimental_gaussian_render_inference_scene_cuda"] = exp
    return {"arm": "c0",
            "so_sha256": sha256_file(C0_SO),
            "core_sha256": sha256_file(CORE_SO),
            "env": {k: os.environ.get(k) for k in
                    ("HIGS_PX_RUNTIME", "HIGS_BWD_SCALAR_ADJOINT", "HIGS_BWD_H8_MR", "HIGS_BWD_ABSGRAD")}}


def load_dataset(scene, device="cuda"):
    from dataset import GTDataset
    return GTDataset(scene=scene, repo_root=REPO, resolution="1080p", device=device, background="black")


def load_model_params(scene, ckpt_iter, device="cuda"):
    """Load raw frozen parameters (xyz/shs/scaling/rotation/opacity) as plain
    tensors (no optimizer needed for P10 render+backward preflight)."""
    import torch
    ckpt_path = os.path.join(CKPT_DIR.format(scene=scene), f"ckpt_{ckpt_iter}.pt")
    o = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m = {
        "xyz": o["xyz"].to(device).float().contiguous(),
        "shs": o["shs"].to(device).float().contiguous(),
        "scaling": o["scaling"].to(device).float().contiguous(),
        "rotation": o["rotation"].to(device).float().contiguous(),
        "opacity": o["opacity"].to(device).float().contiguous(),
    }
    ident = {
        "ckpt_path": ckpt_path,
        "ckpt_sha256": sha256_file(ckpt_path),
        "ckpt_bytes": os.path.getsize(ckpt_path),
        "scene": o.get("scene", scene),
        "arm": o.get("arm"),
        "iteration": int(o.get("iteration", ckpt_iter)),
        "num_points": int(o.get("num_points", o["xyz"].shape[0])),
        "active_sh_degree": int(o.get("active_sh_degree", 3)),
        "spatial_lr_scale": float(o.get("spatial_lr_scale", 0.0)),
    }
    return m, ident


# ---------------------------------------------------------------------------
# Native C0 render (mirrors render_with_meta C0 branch, verbatim)
# ---------------------------------------------------------------------------
def render_c0(m, cam, sh_degree, eps2d):
    """Frozen C0 forward. Returns (frame [H,W,3] clamped, radii [1,N,2], densification_info).
    densification_info carries the forward-captured state needed by the walk."""
    from gsplat.experimental import rasterize_gaussian_higs_dynamic
    out = rasterize_gaussian_higs_dynamic(
        m["xyz"], m["rotation"], m["scaling"], m["opacity"], m["shs"],
        sh_degree=sh_degree,
        viewmats=cam.viewmatrix.unsqueeze(0).unsqueeze(0),
        Ks=cam.K.unsqueeze(0).unsqueeze(0),
        width=cam.image_width, height=cam.image_height,
        backward_mode="higs_native", enable_culling=True,
        camera_model="pinhole", render_mode="RGB",
        eps2d=eps2d)
    info = out["densification_info"]
    frame = out["frame"].reshape(cam.image_height, cam.image_width, 3)
    return frame.clamp(0, 1), info["radii"], info


# ---------------------------------------------------------------------------
# Frozen fused-SSIM training loss (P12 --loss fused semantics)
# ---------------------------------------------------------------------------
class SepSSIM:
    """Frozen SepSSIM (evaluation metric; also used as a loss cross-check)."""
    def __init__(self, device="cuda", window_size=11, sigma=1.5):
        import torch
        coords = torch.arange(window_size, dtype=torch.float32, device=device)
        coords -= window_size // 2
        k1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        k1d = k1d / k1d.sum()
        self.k_h = k1d.unsqueeze(0).unsqueeze(0).repeat(15, 1, 1, 1).contiguous()
        self.k_v = k1d.unsqueeze(0).unsqueeze(0).repeat(15, 1, 1, 1).permute(0, 1, 3, 2).contiguous()
        self.padding = window_size // 2

    def __call__(self, pred, target):
        import torch.nn.functional as F
        if pred.ndim == 3:
            pred = pred.unsqueeze(0).permute(0, 3, 1, 2)
            target = target.unsqueeze(0).permute(0, 3, 1, 2)
        stacked = torch.cat([pred, target, pred ** 2, target ** 2, pred * target], dim=1)
        b = F.conv2d(stacked, self.k_h, padding=(0, self.padding), groups=15)
        b = F.conv2d(b, self.k_v, padding=(self.padding, 0), groups=15)
        mu_p, mu_t = b[:, 0:3], b[:, 3:6]
        bp2, bt2, bpt = b[:, 6:9], b[:, 9:12], b[:, 12:15]
        mu_p2, mu_t2, mu_pt = mu_p ** 2, mu_t ** 2, mu_p * mu_t
        sp2, st2, spt = bp2 - mu_p2, bt2 - mu_t2, bpt - mu_pt
        ssim_map = (2 * mu_pt + 0.01 ** 2) * (2 * spt + 0.03 ** 2) / \
                   ((mu_p2 + mu_t2 + 0.01 ** 2) * (sp2 + st2 + 0.03 ** 2))
        return 1.0 - ssim_map.mean()


def fused_loss_fn(pred_hwc, target_hwc, lambda_dssim=0.2):
    """Exact P12 fused training loss: L = 0.8*L1 + 0.2*(1 - fused_ssim[padding=same, train=True])."""
    import torch
    import torch.nn.functional as F
    import fused_ssim as _fused
    b = pred_hwc.permute(2, 0, 1).unsqueeze(0).contiguous()
    g = target_hwc.permute(2, 0, 1).unsqueeze(0).contiguous()
    L1 = F.l1_loss(b, g)
    ssim = _fused.fused_ssim(b, g, padding="same", train=True)
    return (1.0 - lambda_dssim) * L1 + lambda_dssim * (1.0 - ssim)


# ---------------------------------------------------------------------------
# CPU float32 emulation of the C0 fast-math (matches the kernel's fp32 ops)
# ---------------------------------------------------------------------------
def _f32(x):
    import numpy as np
    return np.float32(x)


def _cpu_eval_gaussian_weight(conic, dx, dy, opac):
    """CPU mirror of gsplat::eval_gaussian_weight (float32).
    conic = (A, B, C) matching the kernel's vec3 order. sigma = A dx^2 + C dy^2 + B dx dy.
    Returns (vis, alpha, valid) as python floats (fp32-quantized intermediates).
    The kernel computes sigma in fp32 and uses __expf; we replicate fp32 rounding
    at each op. The residual vs the GPU's __expf is ~1 ulp and is covered by the
    "within C0 numerical envelope" preflight-sanity gate (fact doc sec. 5).
    """
    cxx = _f32(conic[0]); cxy = _f32(conic[1]); cyy = _f32(conic[2])
    dx = _f32(dx); dy = _f32(dy); opac = _f32(opac)
    sigma = _f32(0.5) * (cxx * dx * dx + cyy * dy * dy) + cxy * dx * dy
    vis = _f32(math.exp(float(-sigma)))
    alpha = min(float(MAX_ALPHA), float(opac * vis))
    valid = not (bool(sigma < 0.0) or alpha < ALPHA_THRESHOLD)
    return float(vis), alpha, valid


# ---------------------------------------------------------------------------
# The EXACT C0 blend-backward walk for one NATIVE tile x batch bucket.
#
# Transcribed 1:1 from higs_blend_bwd_px_kernel (PX=2, the frozen C0 arm) in
# HigsNativeBackward.cu, which delegates the per-hit math to
# rasterize_to_pixels_3dgs_blend_bwd in RasterizeToPixels3DGSDevice.cuh.
#
# KEY FROZEN-SOURCE SEMANTICS (verified, see A0_EXACT_BACKWARD_FACTORIZATION.md):
#   * T is INITIALIZED TO T_final = 1 - render_alphas[pixel] (kernel line 404),
#     NOT 1.0. render_alphas is the per-pixel accumulated alpha; T_final is the
#     transmittance reaching the background.
#   * PRIMARY q_color[p,i,k] = fac * v_render_c[k]  where fac = alpha * T (line 134).
#     This is the EXACT per-channel color adjoint (the kernel's v_rgb_local).
#   * w_color[p,i] = fac (broadcast) is the scalar color WEIGHT; it is a SECONDARY
#     export, NOT the primary color adjoint coefficient.
#   * PRIMARY q_alpha[p,i] = v_alpha (lines 137-154), the exact scalar adjoint on
#     dL/dalpha before the local alpha-derivative expansion.
#   * The bucket is a NATIVE (tile_id, batch_index b): batch_end = range_end-1-b*W,
#     batch_size = min(W, batch_end+1-range_start), local t = 0..batch_size-1 maps to
#     flatten_ids[batch_end - t]. W = block_size = 256/PX = 128 (PX=2 frozen arm).
#   * Acceptance for pixel p, local t: inside AND (batch_end - t <= bin_final) AND
#     eval_gaussian_weight valid. The backward does NOT re-check the forward
#     TRANSMITTANCE_THRESHOLD early-stop.
#
# For every accepted (pixel, gaussian) we record q_alpha, q_color[3], w_color, and
# the local Gaussian id / local index.
# ---------------------------------------------------------------------------
def tile_bucket_walk(conics, means2d, opacities, colors_eval,
                     tile_range_start, tile_range_end, flatten_ids,
                     render_alphas, last_ids,
                     image_id, tile_id, image_width, image_height, tile_size,
                     v_render_colors, batch_index, bucket_width):
    """Replay the exact C0 blend-backward for one native (tile, batch) bucket.

    bucket_width = native block_size (256/PX = 128 for the frozen PX=2 arm).
    batch_index b selects the native batch:
        batch_end  = tile_range_end - 1 - b*bucket_width
        batch_size = min(bucket_width, batch_end + 1 - tile_range_start)
        g for local t = flatten_ids[batch_end - t]

    Returns dict with:
      accepted   : list of dicts per accepted hit, back-to-front per pixel
      hits       : total accepted hits in the bucket
      per_g      : dict global_gaussian_id -> accepted count
      batch_end, batch_size, range_start, range_end, batch_index
    """
    conics = list(conics)
    means2d = list(means2d)
    opacities = list(opacities)
    colors_eval = list(colors_eval)
    flatten_ids = list(flatten_ids)

    range_start = int(tile_range_start)
    range_end = int(tile_range_end)
    batch_end = range_end - 1 - int(batch_index) * int(bucket_width)
    batch_size = min(int(bucket_width), batch_end + 1 - range_start)
    if batch_size <= 0:
        batch_size = 0

    i0 = (tile_id // ((image_width + tile_size - 1) // tile_size)) * tile_size
    j0 = (tile_id % ((image_width + tile_size - 1) // tile_size)) * tile_size

    accepted = []
    per_g = {}
    hits = 0

    for i in range(tile_size):
        for j in range(tile_size):
            I_abs = i0 + i
            J_abs = j0 + j
            if not (I_abs < image_height and J_abs < image_width):
                continue
            pix = I_abs * image_width + J_abs
            pix = min(pix, image_width * image_height - 1)
            # T initialized to T_final (kernel line 404), NOT 1.0
            T_final = 1.0 - float(render_alphas[image_id, I_abs, J_abs])
            T = T_final
            buffer = [0.0, 0.0, 0.0]
            bin_final = int(last_ids[image_id, I_abs, J_abs])
            vc = v_render_colors[I_abs, J_abs]  # (r,g,b) float32
            for t in range(0, batch_size):
                if (batch_end - t) > bin_final:
                    continue  # per-pixel validity gate
                g = int(flatten_ids[batch_end - t])
                dx = float(means2d[g][0]) - (J_abs + 0.5)
                dy = float(means2d[g][1]) - (I_abs + 0.5)
                vis, alpha, valid = _cpu_eval_gaussian_weight(conics[g], dx, dy, opacities[g])
                if not valid:
                    continue
                # ---- exact blend-bwd per-hit math (frozen source, lines 126-214) ----
                ra = 1.0 / max(MIN_ONE_MINUS_ALPHA, 1.0 - alpha)
                T = T * ra
                fac = alpha * T                              # w_color[p,i] (secondary)
                rgbs = colors_eval[g]                        # (r,g,b)
                # PRIMARY q_color[p,i,k] = fac * vc[k]   (line 134: v_rgb_local[k])
                q_color = (_f32(fac * float(vc[0])),
                           _f32(fac * float(vc[1])),
                           _f32(fac * float(vc[2])))
                # q_alpha (lines 137-154); v_render_a = 0 (RGB-only), backgrounds=None -> 0
                v_alpha = 0.0
                for k in range(3):
                    v_alpha += (rgbs[k] * T - buffer[k] * ra) * float(vc[k])
                # buffer update (line 211-213)
                for k in range(3):
                    buffer[k] += rgbs[k] * fac
                per_g[g] = per_g.get(g, 0) + 1
                hits += 1
                accepted.append({
                    "I": I_abs, "J": J_abs, "g": g, "local_t": t,
                    "vis": _f32(vis), "alpha": _f32(alpha), "T_after": _f32(T),
                    "w_color": _f32(fac),
                    "q_color": q_color,
                    "q_alpha": _f32(v_alpha),
                    "rgbs": tuple(_f32(x) for x in rgbs),
                })
    return {
        "accepted": accepted, "hits": hits, "per_g": per_g,
        "batch_end": int(batch_end), "batch_size": int(batch_size),
        "range_start": range_start, "range_end": range_end,
        "batch_index": int(batch_index),
    }


# ---------------------------------------------------------------------------
# Native bucket enumeration for a tile (split the tile range by bucket_width).
# Returns list of (batch_index, range_start, range_end, batch_end, batch_size).
# ---------------------------------------------------------------------------
def tile_batches(range_start, range_end, bucket_width):
    range_start = int(range_start)
    range_end = int(range_end)
    num_batches = (range_end - range_start + int(bucket_width) - 1) // int(bucket_width)
    out = []
    for b in range(num_batches):
        batch_end = range_end - 1 - b * int(bucket_width)
        batch_size = min(int(bucket_width), batch_end + 1 - range_start)
        out.append({"batch_index": b, "range_start": range_start, "range_end": range_end,
                    "batch_end": int(batch_end), "batch_size": int(batch_size)})
    return out


# ---------------------------------------------------------------------------
# Dense 16x16 coefficient tensors from a bucket walk (prereg sec.4).
# q_alpha[:, i] and q_color[:, i, c] over the 256-pixel C-order flattened tile,
# zero at non-accepted pixels. Columns are the bucket's distinct accepted
# Gaussians in first-accept order (m <= native bucket width).
# ---------------------------------------------------------------------------
def dense_coefficients(bucket_result, tile_origin, tile_size=16):
    """Build dense 16x16 coefficient tensors for one bucket.

    tile_origin = (I0, J0) absolute top-left of the tile.
    Returns (q_alpha [256,m], q_color [256,m,3], w_color [256,m], accepted_mask [256,m] bool,
             col_gaussians [m], col_index {g: c}).
    Pixel dimension is the 256-pixel C-order flatten of the 16x16 tile
    (row I local, then col J local) -- consistent with the frozen analyzer's
    np.reshape(16,16).ravel() C order and the DCT basis build order.
    """
    import numpy as np
    I0, J0 = tile_origin
    cols = []
    col_index = {}
    for h in bucket_result["accepted"]:
        if h["g"] not in col_index:
            col_index[h["g"]] = len(cols)
            cols.append(h["g"])
    m = len(cols)
    q_alpha = np.zeros((tile_size * tile_size, m), dtype=np.float32)
    q_color = np.zeros((tile_size * tile_size, m, 3), dtype=np.float32)
    w_color = np.zeros((tile_size * tile_size, m), dtype=np.float32)
    accepted_mask = np.zeros((tile_size * tile_size, m), dtype=bool)
    for h in bucket_result["accepted"]:
        li = h["I"] - I0
        lj = h["J"] - J0
        p = li * tile_size + lj          # C-order flatten
        c = col_index[h["g"]]
        q_alpha[p, c] = float(h["q_alpha"])
        q_color[p, c, 0] = float(h["q_color"][0])
        q_color[p, c, 1] = float(h["q_color"][1])
        q_color[p, c, 2] = float(h["q_color"][2])
        w_color[p, c] = float(h["w_color"])
        accepted_mask[p, c] = True
    return q_alpha, q_color, w_color, accepted_mask, cols, col_index


# ---------------------------------------------------------------------------
# Fixed 2D orthonormal DCT-II basis, ordered by (fx+fy, fx, fy) -- identical to
# the frozen analyzer a0_structure_analyze_frozen.py. Diagnostic only (does not
# affect the P10 verdict, which is oracle-SVD only).
# ---------------------------------------------------------------------------
def dct_basis_1d(n=16):
    import numpy as np
    B = np.zeros((n, n), dtype=np.float64)
    for k in range(n):
        a = np.sqrt(1.0 / n) if k == 0 else np.sqrt(2.0 / n)
        for x in range(n):
            B[x, k] = a * np.cos(np.pi * (x + 0.5) * k / n)
    return B


def dct_basis_2d():
    import numpy as np
    B1 = dct_basis_1d(16)
    cols = []
    idx = []
    for fy in range(16):
        for fx in range(16):
            img = np.outer(B1[:, fy], B1[:, fx])
            cols.append(img.reshape(-1))
            idx.append((fx, fy))
    order = sorted(range(256), key=lambda i: (sum(idx[i]), idx[i][0], idx[i][1]))
    B = np.stack([cols[i] for i in order], axis=1)
    err = np.max(np.abs(B.T @ B - np.eye(256)))
    if err > 1e-10:
        raise RuntimeError(f"DCT basis not orthonormal: {err}")
    return B


# ---------------------------------------------------------------------------
# Frozen mechanical P10 verdict (oracle SVD only; DCT is diagnostic, ignored).
#   A0P_DROP             : median r99 > 16 for q_alpha OR joint-q_color
#   A0P_STRONG_PROMOTE   : median r995 <= 8 for BOTH
#   A0P_INCONCLUSIVE     : otherwise
# ---------------------------------------------------------------------------
def p10_verdict(rows):
    import numpy as np
    alpha_r99 = [r["alpha"]["r99"] for r in rows]
    color_r99 = [r["color_joint"]["r99"] for r in rows]
    alpha_r995 = [r["alpha"]["r995"] for r in rows]
    color_r995 = [r["color_joint"]["r995"] for r in rows]
    med = lambda xs: float(np.median(xs)) if xs else None
    med_a_r99 = med(alpha_r99); med_c_r99 = med(color_r99)
    med_a_r995 = med(alpha_r995); med_c_r995 = med(color_r995)
    if (med_a_r99 is not None and med_a_r99 > 16) or (med_c_r99 is not None and med_c_r99 > 16):
        code = "A0P_DROP"
    elif (med_a_r995 is not None and med_c_r995 is not None and med_a_r995 <= 8 and med_c_r995 <= 8):
        code = "A0P_STRONG_PROMOTE"
    else:
        code = "A0P_INCONCLUSIVE"
    return {
        "verdict": code,
        "basis": "oracle SVD only (DCT diagnostic; does not affect verdict)",
        "median_alpha_r99": med_a_r99,
        "median_color_r99": med_c_r99,
        "median_alpha_r995": med_a_r995,
        "median_color_r995": med_c_r995,
        "rules": {
            "A0P_DROP": "median r99 > 16 for q_alpha OR joint-q_color",
            "A0P_STRONG_PROMOTE": "median r995 <= 8 for BOTH",
            "A0P_INCONCLUSIVE": "otherwise",
        },
    }
