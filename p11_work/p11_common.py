#!/usr/bin/env python3
"""P11 common: DSH-A recipe bootstrap, model/dataset load, SepSSIM, timing utils.

Everything here is MEASUREMENT-ONLY: no algorithm changes, no new CUDA kernels.
The recipe semantics are copied verbatim from the frozen publication trainer
(/root/3dgs-publication-epic05/publication_trainer_dsha.py, arm gmain) so every
P11 gate measures the exact DSH-A publication recipe on current gsplat main
512d366b.
"""
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------- paths
REPO = "/root/3dgs-renderer-benchmark"
P8DIR = "/root/3dgs-publication-epic05"
FULL30K = os.path.join(P8DIR, "full30k")
GMAIN_TREE = "/root/epic05_p8/gsplat-main-512d366b"
GMAIN_SO = "/root/epic05_p8/build/torchext/gsplat_cuda/gsplat_cuda.so"
REFERENCE_V1_DIR = os.path.join(REPO, "baseline", "reference_v1")
PHASE7_DIR = os.path.join(REPO, "scripts", "epic05", "phase7")
SRC_DIR = os.path.join(REPO, "src")
P11_DIR = "/root/epic05_p11"
STATEGEN_DIR = os.path.join(P11_DIR, "stategen")

SCENES = ["room", "bicycle", "garden"]
EPS2D = 0.3
LAMBDA_DSSIM = 0.2  # loss = 0.8*L1 + 0.2*(1-SSIM)


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


def bootstrap_gmain():
    """Verbatim trainer bootstrap for the gmain arm (direct .so binding)."""
    mod = load_ext_module("gsplat_cuda", GMAIN_SO)
    sys.modules["gsplat_cuda"] = mod
    sys.modules["gsplat.csrc"] = mod
    sys.path.insert(0, GMAIN_TREE)
    sys.path.insert(0, PHASE7_DIR)
    sys.path.insert(0, SRC_DIR)
    sys.path.insert(0, REFERENCE_V1_DIR)
    import gsplat
    import inspect
    sig = inspect.signature(gsplat.rasterization)
    assert "accutile" not in sig.parameters, "accutile present -- wrong tree"
    assert "absgrad" in sig.parameters, "absgrad missing -- wrong tree"
    assert sig.parameters["rasterize_mode"].default == "classic", \
        "rasterize_mode default is not classic -- semantics risk"
    return {"arm": "gmain", "so_sha256": sha256_file(GMAIN_SO),
            "so_path": GMAIN_SO, "gsplat_file": gsplat.__file__,
            "base_commit": "512d366b67073d77ca099ede742683c165dfc23b "
                           "(gsplat main, P8 pin 2026-09-25)"}


def import_recipe():
    """Import recipe modules AFTER bootstrap. Returns the module namespace."""
    from gsplat import rasterization
    from gaussian_model import GaussianModel
    from config import ReferenceV1Config
    from colmap_reader import read_points3D_binary, sfm_to_pcd_data
    from dataset import GTDataset
    from gsplat.cuda._wrapper import (
        rasterize_num_contributing_gaussians,
        rasterize_contributing_gaussian_ids,
    )
    import fused_ssim
    ns = dict(
        rasterization=rasterization,
        GaussianModel=GaussianModel,
        ReferenceV1Config=ReferenceV1Config,
        read_points3D_binary=read_points3D_binary,
        sfm_to_pcd_data=sfm_to_pcd_data,
        GTDataset=GTDataset,
        rasterize_num_contributing_gaussians=rasterize_num_contributing_gaussians,
        rasterize_contributing_gaussian_ids=rasterize_contributing_gaussian_ids,
        fused_ssim=fused_ssim,
    )
    return ns


# ---------------- SepSSIM / PSNR (verbatim from publication_trainer_dsha.py) ----
class SepSSIM:
    def __init__(self, device="cuda", window_size=11, sigma=1.5, dtype=torch.float32):
        coords = torch.arange(window_size, dtype=dtype, device=device)
        coords -= window_size // 2
        k1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        k1d = k1d / k1d.sum()
        self.window_size = window_size
        self.sigma = sigma
        self.dtype = dtype
        n_groups = 15
        self.k_h = k1d.unsqueeze(0).unsqueeze(0).repeat(n_groups, 1, 1, 1).contiguous()
        self.k_v = k1d.unsqueeze(0).unsqueeze(0).repeat(n_groups, 1, 1, 1).permute(0, 1, 3, 2).contiguous()
        self.padding = window_size // 2

    def __call__(self, pred, target):
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


# ---------------- fused-SSIM same-semantics loss (existing public impl) ---------
def fused_total_loss(img_hwc, gt_hwc, padding="same"):
    """Total DSH-A loss with the existing fused-ssim as the SSIM backend.

    img_hwc: [H,W,3] leaf tensor (requires_grad). Returns scalar loss.
    Semantics: L = 0.8*L1 + 0.2*(1 - SSIM), SSIM from fused_ssim(padding="same").
    """
    img_bchw = img_hwc.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
    gt_bchw = gt_hwc.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
    L1 = F.l1_loss(img_bchw, gt_bchw)
    ssim = fused_ssim_fn(img_bchw, gt_bchw, padding=padding)
    return (1.0 - LAMBDA_DSSIM) * L1 + LAMBDA_DSSIM * (1.0 - ssim)


def fused_ssim_fn(img_bchw, gt_bchw, padding="same"):
    import fused_ssim
    return fused_ssim.fused_ssim(img_bchw, gt_bchw, padding=padding, train=True)


def sep_total_loss(img_hwc, gt_hwc, ssim_fn):
    """Total DSH-A loss with the publication SepSSIM backend (verbatim math)."""
    L1 = F.l1_loss(img_hwc, gt_hwc)
    dssim = ssim_fn(img_hwc, gt_hwc)
    return (1.0 - LAMBDA_DSSIM) * L1 + LAMBDA_DSSIM * dssim


# ---------------- model / dataset --------------------------------------------
TRAIN_CFG = {
    "position_lr_init": 0.00016,
    "position_lr_final": 0.0000016,
    "position_lr_delay_mult": 0.01,
    "position_lr_max_steps": 30000,
    "feature_lr": 0.0025,
    "opacity_lr": 0.025,
    "scaling_lr": 0.005,
    "rotation_lr": 0.001,
    "percent_dense": 0.01,
}


def load_dataset(scene):
    """Load GTDataset once per scene (expensive: caches all GT images on CPU)."""
    from dataset import GTDataset
    return GTDataset(scene=scene, repo_root=REPO, resolution="1080p",
                     device="cuda", background="black")


def load_model_and_dataset(scene, ckpt_path=None, state_path=None):
    """Replicate the trainer's init exactly, then (optionally) restore a state.

    ckpt_path: full30k ckpt_30000.pt (capture()+optimizer_state_dict+...)
    state_path: P11 stategen state (same schema).
    Returns (model, dataset, scene_extent, meta)
    """
    dataset = load_dataset(scene)
    model, scene_extent, meta = load_model_for_dataset(
        scene, dataset, ckpt_path=ckpt_path, state_path=state_path)
    meta["n_cameras"] = len(dataset)
    return model, dataset, scene_extent, meta


def load_model_for_dataset(scene, dataset, ckpt_path=None, state_path=None):
    """Model init + state restore given an existing dataset (fast path)."""
    from colmap_reader import read_points3D_binary, sfm_to_pcd_data
    from gaussian_model import GaussianModel
    from config import ReferenceV1Config

    config = ReferenceV1Config()
    config.scene = scene
    config.repo_root = REPO

    scene_extent = compute_scene_extent(dataset)

    sfm_path = os.path.join(REPO, "data", "datasets", "mipnerf360", scene,
                            "sparse", "0", "points3D.bin")
    points3d = read_points3D_binary(sfm_path)
    pcd_data = sfm_to_pcd_data(points3d, sh_degree=config.sh_degree)

    model = GaussianModel(max_sh_degree=config.sh_degree)
    model.create_from_pcd(pcd_data, spatial_lr_scale=scene_extent)
    model.training_setup(TRAIN_CFG)

    meta = {"scene": scene, "scene_extent": scene_extent,
            "initial_N": int(model._xyz.shape[0])}
    if ckpt_path is not None:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.restore(state, TRAIN_CFG)
        # restore() calls training_setup() AFTER assigning accum/denom, which
        # re-zeroes them; re-assign so the restored state is exactly the saved
        # state (values are timing-irrelevant, but keep it faithful).
        if "xyz_gradient_accum" in state:
            model.xyz_gradient_accum = state["xyz_gradient_accum"].cuda()
        if "denom" in state:
            model.denom = state["denom"].cuda()
        meta["state_source"] = ckpt_path
        meta["state_sha256"] = sha256_file(ckpt_path)
        meta["state_iteration"] = state.get("iteration")
        meta["state_N"] = int(model._xyz.shape[0])
        meta["active_sh_degree"] = int(model.active_sh_degree)
    elif state_path is not None:
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        model.restore(state, TRAIN_CFG)
        if "xyz_gradient_accum" in state:
            model.xyz_gradient_accum = state["xyz_gradient_accum"].cuda()
        if "denom" in state:
            model.denom = state["denom"].cuda()
        meta["state_source"] = state_path
        meta["state_sha256"] = sha256_file(state_path)
        meta["state_iteration"] = state.get("iteration")
        meta["state_N"] = int(model._xyz.shape[0])
        meta["active_sh_degree"] = int(model.active_sh_degree)
    return model, scene_extent, meta


def compute_scene_extent(dataset, n_samples=50):
    centers = []
    n = min(n_samples, len(dataset))
    for i in range(n):
        cam = dataset.get_camera(i)
        centers.append(cam.camera_center.cpu().numpy())
    centers = np.array(centers)
    extent = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            extent = max(extent, d)
    return max(extent, 0.1)


def camera_sequence_for(scene, iterations=30000, seed=42):
    """Frozen camera sequence (identical to the campaign trainer)."""
    import random
    n_cameras = None
    # n_cameras comes from the dataset; but we can read it from the campaign file
    # to avoid loading the dataset first when only the sequence is needed.
    seq_path = os.path.join(FULL30K, f"gmain03_s42_{scene}", "camera_sequence.npy")
    if os.path.exists(seq_path):
        seq = np.load(seq_path)
        return seq
    return None


# ---------------- render (verbatim trainer semantics, gmain arm) --------------
def render_with_meta(model, cam, sh_degree, ns=None, absgrad=True,
                     return_inputs=False):
    """Trainer's render path. Returns (r0_unclamped, image_clamped, radii_full, means2d).

    r0 is the graph node for the raw renderer output [H,W,3]; image = r0.clamp(0,1).
    Keeping r0 lets us take the exact dL/dimage adjoint INCLUDING the clamp backward.

    With return_inputs=True, additionally returns `inputs` — the EXACT parameter
    tensors handed to rasterization() (get_rotation/get_scaling/get_opacity/
    get_features create FRESH tensors per call, so autograd.grad targets must be
    these objects, not new property calls).
    """
    from gsplat import rasterization
    means = model.get_xyz
    quats = model.get_rotation
    scales = model.get_scaling
    opacities = model.get_opacity
    colors = model.get_features
    r, _, meta = rasterization(
        means=means, quats=quats,
        scales=scales, opacities=opacities,
        colors=colors,
        viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
        width=cam.image_width, height=cam.image_height,
        tile_size=16, packed=False, sh_degree=sh_degree,
        radius_clip=0.0, eps2d=EPS2D, render_mode="RGB",
        absgrad=absgrad)
    means2d = meta["means2d"]
    if means2d.requires_grad:
        means2d.retain_grad()
    r0 = r[0]
    inputs = [means, quats, scales, opacities, colors]
    if return_inputs:
        return r0, r0.clamp(0, 1), meta["radii"], means2d, inputs
    return r0, r0.clamp(0, 1), meta["radii"], means2d


def renderer_param_inputs(model, cam, sh_degree):
    """The exact tensors the trainer hands to rasterization() (grad inputs)."""
    return dict(
        means=model.get_xyz, quats=model.get_rotation,
        scales=model.get_scaling, opacities=model.get_opacity,
        colors=model.get_features)


# ---------------- GPU policy ---------------------------------------------------
def gpu_uuid(index):
    out = subprocess.run(
        ["nvidia-smi", f"--id={index}", "--query-gpu=uuid,name,driver_version",
         "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip()
    return out


def gpu_compute_apps():
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
         "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip()
    return [l for l in out.splitlines() if l.strip()]


def assert_clean_gpu(index, allow_shared=False):
    """Refuse to run on a GPU with foreign compute processes (clean A100 policy)."""
    info = gpu_uuid(index)
    uuid = info.split(",")[0].strip() if info else "unknown"
    procs = [p for p in gpu_compute_apps() if uuid in p]
    ours = str(os.getpid())
    foreign = [p for p in procs if ours not in p.split(",")[0]]
    if foreign and not allow_shared:
        raise RuntimeError(
            f"GPU {index} ({uuid}) is NOT CLEAN; foreign compute processes: {foreign}. "
            "Refusing timed measurement (clean A100-SXM4 policy). Use another GPU.")
    return {"gpu_index": index, "gpu_info": info, "foreign_processes": foreign}


# ---------------- timing -------------------------------------------------------
def bench_events(fn, warmup=20, iters=100, reps=5, sync_each_rep=True):
    """CUDA-event benchmark. Returns rep medians + median-of-medians (ms).

    Each call is bracketed by its own event pair on the current stream, so
    host-induced stream gaps inside fn are included (same property as the
    trainer's phase timing).
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    rep_meds = []
    rep_means = []
    for _ in range(reps):
        evs = [torch.cuda.Event(enable_timing=True) for _ in range(iters + 1)]
        evs[0].record()
        for i in range(iters):
            fn()
            evs[i + 1].record()
        torch.cuda.synchronize()
        ts = np.array([evs[i].elapsed_time(evs[i + 1]) for i in range(iters)])
        rep_meds.append(float(np.median(ts)))
        rep_means.append(float(np.mean(ts)))
    return {
        "protocol": {"warmup": warmup, "measurements": iters, "reps": reps,
                     "timer": "cuda_events_per_call"},
        "rep_medians_ms": rep_meds,
        "rep_means_ms": rep_means,
        "median_of_medians_ms": float(np.median(rep_meds)),
        "mean_of_medians_ms": float(np.mean(rep_meds)),
    }


def phase_timer():
    """Trainer-style 6-event phase timer."""
    pev = [torch.cuda.Event(enable_timing=True) for _ in range(6)]
    return pev


def dump_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"WROTE {path}")


def env_identity():
    import torch as _t
    return {
        "python": sys.version.split()[0],
        "torch": _t.__version__,
        "cuda_runtime": _t.version.cuda,
        "cudnn": _t.backends.cudnn.version(),
        "hostname": subprocess.run(["hostname"], capture_output=True, text=True).stdout.strip(),
        "gpus": subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total",
             "--format=csv,noheader"], capture_output=True, text=True).stdout.strip().splitlines(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
