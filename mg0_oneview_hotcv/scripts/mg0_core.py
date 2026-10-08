#!/usr/bin/env python3
"""MG0 OneView-GS / HotCV — shared core machinery.

Frozen semantics are inherited VERBATIM from the P12/C0/N1 lineage:
- bootstrap_c0(): identical to publication_trainer_p12.py bootstrap_c0()
  (env switches, frozen c0_ext/core .so via sys.modules injection, path order)
- make_fused_loss(): P12 --loss fused with the P11 .so sha256 lock
- render_view(): the trainer's c0 render branch (eps2d=0.3, higs_native)
- load_scene_dataset / training_cfg_dict / build_model / load_checkpoint:
  identical to the N1 verified restore path
- StateMaster: N1 bit-exact GPU-resident state clone/restore

MG0-specific additions (analysis only; NO renderer/optimizer/loss change):
- ROW LAYOUT (59 dims/Gaussian): xyz(3) | scale(3) | quat(4) | opacity(1)
  | sh_dc(3) | sh_rest(45).  Used only to flatten gradients/controls for
  analysis and storage; the trainer parameterization is untouched.
- flat_denom(): per-element  sqrt(v_t) + eps  in row layout, with the eps of
  the gin optimizer group each element belongs to (P_t(x) = x/(sqrt(v_t)+eps)).
- streaming fp64 accumulators for cross-view scores and estimator metrics.
"""
import hashlib
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------- frozen paths
REPO = "/root/3dgs-renderer-benchmark"
C0_WT = "/root/epic05_frozen/higs_c0_final30k_worktree"
SCENE_CUDA_DIR = "/root/epic05_build_cache/scene/gsplat_scene_cuda"
CORE_SO = "/root/epic05_build_cache/core/gsplat_cuda/gsplat_cuda.so"
C0_SO = ("/root/epic05_build_cache/c0_ext/"
         "experimental_gaussian_render_inference_scene_cuda/"
         "experimental_gaussian_render_inference_scene_cuda.so")
PHASE7_DIR = os.path.join(REPO, "scripts", "epic05", "phase7")
SRC_DIR = os.path.join(REPO, "src")
REFERENCE_V1_DIR = os.path.join(REPO, "baseline", "reference_v1")

MG0_ROOT = "/root/epic05_mg0"
ARTIFACTS = os.path.join(MG0_ROOT, "artifacts")
CONTROLS_DIR = os.path.join(MG0_ROOT, "controls")
LOGS_DIR = os.path.join(MG0_ROOT, "logs")

P12_TRAINER = "/root/epic05_p12/publication_trainer_p12.py"
N1_SNAP_TRAINER = "/root/epic05_n1/publication_trainer_p12_n1snap.py"
P11_FUSED_SO_SHA256 = "dd6a4c799b98ff9fa82b4203acce3b36b1529c97155c0ef4cd50fd0f583d8ebb"
SNAP_RUNS = {
    "room": "/root/epic05_n1/runs/n1snap_c0f_s42_room",
    "bicycle": "/root/epic05_n1/runs/n1snap_c0f_s42_bicycle",
}
SCENES = ("room", "bicycle")
STAGE_CKPTS = (15000, 20000, 25000)
N_STALE_LAG = 5000           # control staleness (5K iterations) frozen by prereg
K_LIST = (2048, 8192, 32768)
K_MAIN = 32768               # max hot set stored per view
N_VIEWS = 64                 # frozen camera subset size
EPS2D = 0.3                  # frozen c0 arm value
LAMBDA_DSSIM = 0.2           # frozen recipe

# ---------------------------------------------------------------- row layout
# optimizer param groups (reference_v1 gaussian_model.training_setup order)
OPT_GROUPS = ("xyz", "shs", "opacity", "scaling", "rotation")
# MG0 analysis groups: (name, col_start, col_stop) into the 59-dim row
GROUPS = (
    ("xyz", 0, 3),
    ("scale", 3, 6),
    ("quat", 6, 10),
    ("opacity", 10, 11),
    ("sh_dc", 11, 14),
    ("sh_rest", 14, 59),
)
ROW_DIM = 59
SH_REST_DIM = 45
PARAM_NAMES = ("_xyz", "_shs", "_scaling", "_rotation", "_opacity")

N1_ACTION_TIMING = {
    "room": "/root/epic05_n1/artifacts/N1_ACTION_TIMING_room.json",
    "bicycle": "/root/epic05_n1/artifacts/N1_ACTION_TIMING_bicycle.json",
}


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


def bootstrap_c0():
    """VERBATIM from publication_trainer_p12.py bootstrap_c0()."""
    os.environ["HIGS_PX_RUNTIME"] = "2"
    os.environ["HIGS_BWD_SCALAR_ADJOINT"] = "scalar_adjoint"
    os.environ["HIGS_BWD_H8_MR"] = "1"
    os.environ["HIGS_BWD_ABSGRAD"] = "1"
    os.environ.pop("HIGS_DISABLE_F9", None)
    sys.path.insert(0, C0_WT)
    sys.path.insert(0, SCENE_CUDA_DIR)
    core = load_ext_module("gsplat_cuda", CORE_SO)
    sys.modules["gsplat.csrc"] = core
    exp = load_ext_module("experimental_gaussian_render_inference_scene_cuda", C0_SO)
    sys.modules["gsplat.experimental.render.kernels.csrc"] = exp
    sys.modules["experimental_gaussian_render_inference_scene_cuda"] = exp
    sys.path.insert(0, PHASE7_DIR)
    sys.path.insert(0, SRC_DIR)
    sys.path.insert(0, REFERENCE_V1_DIR)
    from gsplat.experimental import rasterize_gaussian_higs_dynamic  # noqa: F401
    from gsplat.experimental.render.functional.gaussian_inference import (  # noqa: F401
        _HIGS_DYNAMIC_SCENE)
    return {"arm": "c0", "so_sha256": sha256_file(C0_SO), "so_path": C0_SO,
            "core_sha256": sha256_file(CORE_SO),
            "env": {k: os.environ[k] for k in
                    ("HIGS_PX_RUNTIME", "HIGS_BWD_SCALAR_ADJOINT",
                     "HIGS_BWD_H8_MR", "HIGS_BWD_ABSGRAD")}}


def make_fused_loss():
    """P12 --loss fused semantics with the P11 binary lock (verbatim checks)."""
    import fused_ssim as fused_ssim_mod
    import fused_ssim_cuda as fused_ssim_cuda_mod
    so_path = fused_ssim_cuda_mod.__file__
    so_sha = sha256_file(so_path)
    if so_sha != P11_FUSED_SO_SHA256:
        raise RuntimeError(
            f"MG0 FUSED LOSS LOCK VIOLATION: fused_ssim_cuda.so sha256 {so_sha} "
            f"!= P11-locked {P11_FUSED_SO_SHA256}.")

    def fused_loss(pred_hwc, target_hwc):
        b = pred_hwc.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
        g = target_hwc.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
        L1 = F.l1_loss(b, g)
        ssim = fused_ssim_mod.fused_ssim(b, g, padding="same", train=True)
        return (1.0 - LAMBDA_DSSIM) * L1 + LAMBDA_DSSIM * (1.0 - ssim)

    fused_loss.so_path = so_path
    fused_loss.so_sha256 = so_sha
    return fused_loss


def render_view(model, cam, sh_degree):
    """VERBATIM c0 branch of the trainer's render_with_meta."""
    from gsplat.experimental import rasterize_gaussian_higs_dynamic
    out = rasterize_gaussian_higs_dynamic(
        model.get_xyz, model.get_rotation, model.get_scaling,
        model.get_opacity, model.get_features,
        sh_degree=sh_degree,
        viewmats=cam.viewmatrix.unsqueeze(0).unsqueeze(0),
        Ks=cam.K.unsqueeze(0).unsqueeze(0),
        width=cam.image_width, height=cam.image_height,
        backward_mode="higs_native", enable_culling=True,
        camera_model="pinhole", render_mode="RGB",
        eps2d=EPS2D)
    info = out["densification_info"]
    means2d = info["means2d"]
    frame = out["frame"].reshape(cam.image_height, cam.image_width, 3)
    return frame.clamp(0, 1), info["radii"], means2d


def load_scene_dataset(scene):
    """GTDataset exactly as the trainer builds it (1080p, black, cuda)."""
    from dataset import GTDataset
    return GTDataset(scene=scene, repo_root=REPO, resolution="1080p",
                     device="cuda", background="black")


def training_cfg_dict():
    """The exact config dict the trainer passes to model.training_setup."""
    from config import ReferenceV1Config
    c = ReferenceV1Config()
    return {"position_lr_init": c.position_lr_init,
            "position_lr_final": c.position_lr_final,
            "position_lr_delay_mult": c.position_lr_delay_mult,
            "position_lr_max_steps": c.position_lr_max_steps,
            "feature_lr": c.feature_lr,
            "opacity_lr": c.opacity_lr,
            "scaling_lr": c.scaling_lr,
            "rotation_lr": c.rotation_lr,
            "percent_dense": c.percent_dense}


def scene_reset():
    """Release the HIGS dynamic-scene handle when param tensor objects change."""
    from gsplat.experimental.render.functional.gaussian_inference import (
        _HIGS_DYNAMIC_SCENE)
    _HIGS_DYNAMIC_SCENE.reset()


def load_checkpoint(scene, iteration):
    p = os.path.join(SNAP_RUNS[scene], f"ckpt_{iteration}.pt")
    return torch.load(p, map_location="cpu", weights_only=False)


def build_model(ckpt):
    """GaussianModel.restore + optimizer.load_state_dict (exact semantics)."""
    from gaussian_model import GaussianModel
    m = GaussianModel(max_sh_degree=3)
    m.restore({k: ckpt[k] for k in
               ("active_sh_degree", "xyz", "shs", "scaling", "rotation",
                "opacity", "max_radii2D", "xyz_gradient_accum", "denom",
                "spatial_lr_scale")},
              training_cfg_dict())
    m.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return m


class StateMaster:
    """GPU-resident bit-exact snapshot of (params, Adam moments, step)."""

    def __init__(self, model):
        self.model = model
        self.params = [getattr(model, n) for n in PARAM_NAMES]
        self.opt = model.optimizer
        self.master_p = [p.detach().clone() for p in self.params]
        self.master_state = []
        for p in self.params:
            st = self.opt.state[p]
            self.master_state.append(
                (st["step"].detach().clone(),
                 st["exp_avg"].detach().clone(),
                 st["exp_avg_sq"].detach().clone()))
        self.master_sh = model.active_sh_degree

    def restore(self):
        m = self.model
        for p, mp in zip(self.params, self.master_p):
            p.data.copy_(mp)
        for p, (step, ea, eas) in zip(self.params, self.master_state):
            st = self.opt.state[p]
            st["step"].copy_(step)
            st["exp_avg"].copy_(ea)
            st["exp_avg_sq"].copy_(eas)
        m.active_sh_degree = self.master_sh
        self.opt.zero_grad(set_to_none=True)

    def param_update_norm(self):
        """||theta_now - theta_master|| per MG0 group and global (L2)."""
        vals = {}
        for n in PARAM_NAMES:
            p = getattr(self.model, n)
            i = PARAM_NAMES.index(n)
            vals[n] = p.detach() - self.master_p[i]
        d = rows_from_values(vals)
        per = {}
        tot = 0.0
        for (gname, a, b) in GROUPS:
            s = float((d[:, a:b].double() ** 2).sum())
            per[gname] = math.sqrt(s)
            tot += s
        per["global"] = math.sqrt(tot)
        return per


def view_grad_and_loss(model, cam, gt, fused_loss):
    """N1 pattern: zero_grad -> render -> fused loss -> backward -> flat rows."""
    model.optimizer.zero_grad(set_to_none=True)
    img, radii, means2d = render_view(model, cam, model.active_sh_degree)
    loss = fused_loss(img, gt)
    loss.backward()
    torch.cuda.synchronize()
    g = flat_grad_rows(model)
    return float(loss.item()), g


def rows_from_values(vals):
    """[N,59] rows from per-param tensors: xyz|scale|quat|opacity|dc|rest.

    NB: GaussianModel keeps _opacity as shape [N] (not [N,1]).
    """
    sh = vals["_shs"]
    n = sh.shape[0]
    return torch.cat([
        vals["_xyz"],
        vals["_scaling"],
        vals["_rotation"],
        vals["_opacity"].reshape(n, 1),
        sh[:, 0, :],
        sh[:, 1:, :].reshape(n, SH_REST_DIM),
    ], dim=1)


def flat_grad_rows(model):
    """[N,59] fp32 gradient rows in the frozen layout."""
    return rows_from_values({n: getattr(model, n).grad for n in PARAM_NAMES})


def flat_denom(model):
    """[N,59] fp32  sqrt(v_t)+eps  in row layout (per optimizer-group eps)."""
    st = model.optimizer.state
    eps = {g["name"]: float(g["eps"]) for g in model.optimizer.param_groups}

    def dn(v, e):
        return v.sqrt().add_(e)

    sh_v = st[model._shs]["exp_avg_sq"]
    n = sh_v.shape[0]
    return torch.cat([
        dn(st[model._xyz]["exp_avg_sq"], eps["xyz"]),
        dn(st[model._scaling]["exp_avg_sq"], eps["scaling"]),
        dn(st[model._rotation]["exp_avg_sq"], eps["rotation"]),
        dn(st[model._opacity]["exp_avg_sq"], eps["opacity"]).reshape(n, 1),
        dn(sh_v[:, 0, :], eps["shs"]),
        dn(sh_v[:, 1:, :].reshape(n, SH_REST_DIM), eps["shs"]),
    ], dim=1)


def group_sumsq(t):
    """dict group -> sum of squares over all elements of the group columns (fp64)."""
    out = {}
    for name, a, b in GROUPS:
        s = t[:, a:b]
        out[name] = float((s * s).sum(dtype=torch.float64))
    return out


def group_dot(t1, t2):
    """dict group -> dot product over the group columns (fp64); also global."""
    out = {}
    tot = 0.0
    for name, a, b in GROUPS:
        s = float((t1[:, a:b] * t2[:, a:b]).sum(dtype=torch.float64))
        out[name] = s
        tot += s
    out["global"] = tot
    return out


def quantile_indices(n_total, n_pick):
    """Frozen subset rule: j_k = floor((k+0.5)*N/n), k=0..n-1."""
    return [int(math.floor((k + 0.5) * n_total / n_pick)) for k in range(n_pick)]


def load_camera_set(scene):
    p = os.path.join(ARTIFACTS, "MG0_CAMERA_SET.json")
    d = json.load(open(p))
    return d["scenes"][scene]


def control_path(scene, stage):
    return os.path.join(CONTROLS_DIR, f"{scene}_{stage}_controls.pt")


def save_controls(scene, stage, hot_ids, c_fp16, cbar, meta):
    os.makedirs(CONTROLS_DIR, exist_ok=True)
    path = control_path(scene, stage)
    torch.save({"scene": scene, "stage": stage, "hot_ids": hot_ids,
                "c_fp16": c_fp16, "cbar": cbar, "row_dim": ROW_DIM,
                "meta": meta}, path)
    return path


def load_controls(scene, stage):
    return torch.load(control_path(scene, stage), map_location="cpu",
                      weights_only=False)


def json_dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    print(f"[mg0_core] wrote {path}", flush=True)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    return str(o)


def gpu_mem_gb():
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def a1_median_ms(scene, stage):
    d = json.load(open(N1_ACTION_TIMING[scene]))
    return float(d["cells"][f"{stage}_A1"]["total_wall_ms"]["median"])
