#!/usr/bin/env python3
"""N1 ReBudgetGS probe core — shared machinery for the frozen N1 campaign.

Everything here reproduces the P12/C0 frozen training semantics EXACTLY:
- bootstrap_c0(): verbatim env + frozen binaries from publication_trainer_p12.py
- fused training loss: P11-locked fused-ssim padding='same', L = 0.8*L1 + 0.2*(1-SSIM)
- render path: rasterize_gaussian_higs_dynamic (c0 branch, verbatim)
- model restore: GaussianModel.restore + optimizer.load_state_dict + update_learning_rate
- StateMaster: GPU-resident bit-exact state save/restore for counterfactual actions

No renderer change. No densification. No garden.
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

N1_ROOT = "/root/epic05_n1"
RUNS = os.path.join(N1_ROOT, "runs")
ARTIFACTS = os.path.join(N1_ROOT, "artifacts")

P12_TRAINER = "/root/epic05_p12/publication_trainer_p12.py"
N1_SNAP_TRAINER = os.path.join(N1_ROOT, "publication_trainer_p12_n1snap.py")
P11_FUSED_SO_SHA256 = "dd6a4c799b98ff9fa82b4203acce3b36b1529c97155c0ef4cd50fd0f583d8ebb"
CANONICAL_RUNS = {
    "room": "/root/epic05_p12/runs/p12a/p12a_c0f_s42_room",
    "bicycle": "/root/epic05_p12/runs/p12a/p12a_c0f_s42_bicycle",
}
SNAP_RUNS = {
    "room": os.path.join(RUNS, "n1snap_c0f_s42_room"),
    "bicycle": os.path.join(RUNS, "n1snap_c0f_s42_bicycle"),
}
SCENES = ("room", "bicycle")
CKPTS = (10000, 15000, 20000, 25000)          # snapshots that exist
ANCHOR_CKPTS = (15000, 20000, 25000)          # N1 primary probe states
SKETCH_SRC = {15000: 10000, 20000: 15000, 25000: 20000}  # stale-sketch source

EPS2D = 0.3            # frozen c0 arm value
LAMBDA_DSSIM = 0.2     # frozen recipe
N1_SEED = 20260928     # frozen global campaign seed
SKETCH_DIM = 256       # prereg: fixed projection dimension 256 per group
SKETCH_CHUNK = 1_000_000
HELDOUT_N = 8
N_ANCHORS = 64
A1_REPLICAS = 3

PARAM_NAMES = ("_xyz", "_shs", "_scaling", "_rotation", "_opacity")
# prereg groups: xyz; opacity; scale+rotation; SH
GROUPS = (
    ("xyz", ("_xyz",)),
    ("opacity", ("_opacity",)),
    ("scale_rot", ("_scaling", "_rotation")),
    ("sh", ("_shs",)),
)


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
            f"N1 FUSED LOSS LOCK VIOLATION: fused_ssim_cuda.so sha256 {so_sha} "
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
    """Release the HIGS dynamic-scene renderer handle (P13-Y/P17 discipline).

    Required when the parameter TENSOR OBJECTS change (checkpoint switch:
    build_model creates new nn.Parameters). In-place copy_ mutation
    (StateMaster.restore) is seen live by the handle and needs NO reset —
    the trainer itself mutates params in-place every optimizer.step().
    """
    from gsplat.experimental.render.functional.gaussian_inference import (
        _HIGS_DYNAMIC_SCENE)
    _HIGS_DYNAMIC_SCENE.reset()


def load_checkpoint(scene, iteration):
    p = os.path.join(SNAP_RUNS[scene], f"ckpt_{iteration}.pt")
    # weights_only=False: the file is this campaign's own trainer output;
    # it contains numpy scalars (spatial_lr_scale) that weights_only rejects.
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
        """||theta_now - theta_master|| per group and global (L2)."""
        per = {}
        tot = 0.0
        for (gname, pnames) in GROUPS:
            s = 0.0
            for n in pnames:
                p = getattr(self.model, n)
                i = PARAM_NAMES.index(n)
                d = p.detach() - self.master_p[i]
                s += float(d.pow(2).sum())
            per[gname] = math.sqrt(s)
            tot += s
        per["global"] = math.sqrt(tot)
        return per


@torch.no_grad()
def heldout_eval(model, cams_gts, fused_loss):
    l1s, fuseds, psnrs = [], [], []
    for cam, gt in cams_gts:
        img, _, _ = render_view(model, cam, model.active_sh_degree)
        b = img.unsqueeze(0).permute(0, 3, 1, 2)
        g = gt.unsqueeze(0).permute(0, 3, 1, 2)
        l1s.append(float(F.l1_loss(b, g).item()))
        fuseds.append(float(fused_loss(img, gt).item()))
        mse = F.mse_loss(b, g)
        psnrs.append(float(20 * math.log10(1.0 / math.sqrt(mse.item())))
                     if mse > 1e-10 else 100.0)
    return {"l1": float(np.mean(l1s)), "fused": float(np.mean(fuseds)),
            "psnr": float(np.mean(psnrs))}


def capture_flat_grads(model):
    """Clone current .grad into the 4 prereg groups (flat lists)."""
    out = {}
    for gname, pnames in GROUPS:
        out[gname] = [getattr(model, n).grad.detach().clone() for n in pnames]
    return out


def free_flat_grads(g):
    for v in g.values():
        del v[:]
    g.clear()


def pair_grad_metrics(g1, g2):
    """Per-group + global-normalized cosine, per-group norm ratio."""
    per_group = {}
    dots, na2, nb2 = {}, {}, {}
    for gname, _ in GROUPS:
        dot = 0.0
        a2 = 0.0
        b2 = 0.0
        for ta, tb in zip(g1[gname], g2[gname]):
            fa = ta.reshape(-1).float()
            fb = tb.reshape(-1).float()
            dot += float((fa * fb).sum())
            a2 += float(fa.pow(2).sum())
            b2 += float(fb.pow(2).sum())
        na, nb = math.sqrt(a2), math.sqrt(b2)
        cos = dot / (na * nb) if na > 0 and nb > 0 else float("nan")
        per_group[gname] = {"cosine": cos, "norm_a": na, "norm_b": nb,
                            "norm_ratio_b_over_a": (nb / na) if na > 0 else float("nan")}
        dots[gname] = dot
        na2[gname] = a2
        nb2[gname] = b2
    ga = math.sqrt(sum(na2.values()))
    gb = math.sqrt(sum(nb2.values()))
    gcos = sum(dots.values()) / (ga * gb) if ga > 0 and gb > 0 else float("nan")
    return {"per_group": per_group, "global_cosine": gcos,
            "global_norm_a": ga, "global_norm_b": gb}


def _stable_seed(s):
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "little") % (2 ** 63)


class Sketcher:
    """Globally frozen dense Rademacher projection, 256 rows per group.

    sign(row j, col i) is a deterministic function of (group, chunk, j, i)
    via a per-(group, chunk) seeded CUDA generator. No learned projection.
    """

    def __init__(self, dim=SKETCH_DIM, chunk=SKETCH_CHUNK, seed=N1_SEED):
        self.dim = dim
        self.chunk = chunk
        self.seed = seed

    def _signs(self, group_idx, chunk_idx, ncols, device):
        g = torch.Generator(device=device)
        g.manual_seed(_stable_seed(f"N1SKETCH|{self.seed}|{group_idx}|{chunk_idx}"))
        r = torch.randint(0, 2, (self.dim, ncols), generator=g, device=device,
                          dtype=torch.int8)
        return r.to(torch.float32).mul_(2.0).sub_(1.0)

    def sketch_batch(self, group_idx, G):
        """G: (B, d) float32 cuda -> (B, dim) sketch, scale 1/sqrt(d)."""
        B, d = G.shape
        out = torch.zeros(self.dim, B, device=G.device, dtype=torch.float32)
        for c0 in range(0, d, self.chunk):
            c1 = min(c0 + self.chunk, d)
            s = self._signs(group_idx, c0 // self.chunk, c1 - c0, G.device)
            out.add_(s @ G[:, c0:c1].T)
        out.div_(math.sqrt(d))
        return out.T.contiguous()


def grads_to_group_matrix(flat_grads_list, group_name):
    """Stack a list of captured flat grads into (B, d) for one group."""
    group_idx = [g for g, _ in GROUPS].index(group_name)
    pnames = GROUPS[group_idx][1]
    cols = []
    for fg in flat_grads_list:
        cols.append(torch.cat([t.reshape(-1) for t in fg[group_name]])
                    if len(pnames) > 1 else fg[group_name][0].reshape(-1))
    return torch.stack(cols, dim=0).float(), group_idx


def normalize_sketches(per_group_sketch):
    """Concatenate L2-normalized group sketches -> (B, 4*dim)."""
    parts = []
    for gname, _ in GROUPS:
        s = per_group_sketch[gname]
        parts.append(s / s.norm(dim=1, keepdim=True).clamp_min(1e-30))
    return torch.cat(parts, dim=1)


def jaccard(vis_a, vis_b):
    inter = int((vis_a & vis_b).sum())
    union = int((vis_a | vis_b).sum())
    return inter / union if union > 0 else float("nan")


def gpu_mem_gb():
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def json_dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    print(f"[n1_core] wrote {path}")


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    return str(o)
