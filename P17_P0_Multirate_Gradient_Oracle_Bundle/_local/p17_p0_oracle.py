#!/usr/bin/env python3
"""P17-P0 — Bandwidth-Matched Multirate Gradient Oracle.

One process per scene. Frozen P17_P0_PREREG.md pipeline:
  identity -> freeze cameras -> d=1 parity -> coarse probes d=2/4/8
  -> footprint-only tau assignment (2.0 primary, 1.0/4.0 sensitivity)
  -> hybrid one-step Adam oracle -> unit JSONs + raw NPZs.

Constraints honored: no redesign, no threshold tuning, no new loss,
no production renderer, no training, no speedup claim, camera set frozen
before any gradient probe. Independent renders always start from a fresh
_HIGS_DYNAMIC_SCENE.reset() state (P13-Y discipline); mark_dirty() is
never used as a substitute for reset.
"""
import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ------------------------------------------------------------------ config
FROZEN_HASHES = {
    "p12_trainer": ("675b0267f3a1a6a0fa63fadb37c8536481c1c7a6de37a4b7f6ffa98a449050c5",
                    "/root/epic05_p12/publication_trainer_p12.py"),
    "c0_ext_so": ("d26cccf46c7932bfb5fd44ee05e553af27e17c93eaffb7dacf3197258a513f11",
                  "/root/epic05_build_cache/c0_ext/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so"),
    "core_so": ("935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e",
                "/root/epic05_build_cache/core/gsplat_cuda/gsplat_cuda.so"),
    "fused_ssim_so": ("dd6a4c799b98ff9fa82b4203acce3b36b1529c97155c0ef4cd50fd0f583d8ebb",
                      "/root/miniforge3/envs/gsplat/lib/python3.10/site-packages/fused_ssim_cuda.cpython-310-x86_64-linux-gnu.so"),
}
CKPT_PATH = {
    "room": "/root/epic05_p13x/runs/room_diag/ckpt_16000.pt",
    "bicycle": "/root/epic05_p13x/runs/bicycle_diag/ckpt_16000.pt",
}
CKPT_SHA = {
    "room": "17c1f5904472f24dbb86bc8f19318dc31341deb2f0d891a014662a49d75edc6b",
    "bicycle": "0c36f5c7f127a23ba471e917bcf3ecbf6f9843c295c62bed50e1090aeb6d8890",
}
REPO = "/root/3dgs-renderer-benchmark"
C0_WT = "/root/epic05_frozen/higs_c0_final30k_worktree"
SCENE_CUDA_DIR = "/root/epic05_build_cache/scene/gsplat_scene_cuda"
PHASE7 = os.path.join(REPO, "scripts", "epic05", "phase7")
SRC = os.path.join(REPO, "src")
REF_V1 = os.path.join(REPO, "baseline", "reference_v1")

EPS2D_FULL = 0.3
LAMBDA_DSSIM = 0.2
TILE = 16
TAUS = (2.0, 1.0, 4.0)
DS = (2, 4, 8)
PARAM_NAMES = ("xyz", "shs", "opacity", "scaling", "rotation")
GROUP_KEYS = {"xyz": "means", "scaling": "scales", "rotation": "quats",
              "opacity": "opacity", "shs": "shs"}
N_PROBE = 12
AUDIT_QUANTILES = (0.2, 0.4, 0.6, 0.8)
STEP_T = 16001  # it0=16000 frozen; the single oracle step = iteration 16001
B1, B2, EPS_ADAM = 0.9, 0.999, 1e-15


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def jwrite(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
    print(f"[p17] wrote {path}", flush=True)


def log(msg):
    print(f"[p17 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_ext_module(name, so_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[name] = mod
    return mod


def bootstrap_c0():
    os.environ["HIGS_PX_RUNTIME"] = "2"
    os.environ["HIGS_BWD_SCALAR_ADJOINT"] = "scalar_adjoint"
    os.environ["HIGS_BWD_H8_MR"] = "1"
    os.environ["HIGS_BWD_ABSGRAD"] = "1"
    os.environ.pop("HIGS_DISABLE_F9", None)
    sys.path.insert(0, C0_WT)
    sys.path.insert(0, SCENE_CUDA_DIR)
    core = load_ext_module("gsplat_cuda", FROZEN_HASHES["core_so"][1])
    sys.modules["gsplat.csrc"] = core
    exp = load_ext_module(
        "experimental_gaussian_render_inference_scene_cuda",
        FROZEN_HASHES["c0_ext_so"][1])
    sys.modules["gsplat.experimental.render.kernels.csrc"] = exp
    sys.modules["experimental_gaussian_render_inference_scene_cuda"] = exp
    sys.path.insert(0, PHASE7)
    sys.path.insert(0, SRC)
    sys.path.insert(0, REF_V1)


def verify_identity(out_root, scene):
    checks = {}
    ok = True
    for name, (expected, path) in FROZEN_HASHES.items():
        exists = os.path.exists(path)
        h = sha256_file(path) if exists else None
        match = (h == expected)
        ok = ok and match
        checks[name] = {"path": path, "exists": exists, "sha256": h,
                        "expected": expected, "match": match}
    ck = {"path": CKPT_PATH[scene], "exists": os.path.exists(CKPT_PATH[scene]),
          "sha256": None, "size_bytes": None}
    if ck["exists"]:
        ck["sha256"] = sha256_file(CKPT_PATH[scene])
        ck["size_bytes"] = os.path.getsize(CKPT_PATH[scene])
        ck["lineage"] = (
            "P13-X 16K C0 run (room_diag / bicycle_diag): the exact states "
            "certified in P14 (P14_SOURCE_IDENTITY.json checkpoints_16k). "
            "Strongest existing identity evidence among the five 16K "
            "candidates (pilot_room16k, room_diag, room_diagoff, "
            "bicycle_diag, bicycle_diagoff). No cross-run parameter/optimizer "
            "mixing: both halves come from the same checkpoint file.")
    ident = {
        "phase": "P17-P0",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scene": scene,
        "frozen_binaries": checks,
        "checkpoint": ck,
        "prereg": {"seed": 42, "packed": False, "tile": TILE,
                   "eps2d": EPS2D_FULL,
                   "renderer": "classic C0 (rasterize_gaussian_higs_dynamic)",
                   "resolution": "1080p", "background": "black",
                   "HIGS env": {"HIGS_PX_RUNTIME": "2",
                                "HIGS_BWD_SCALAR_ADJOINT": "scalar_adjoint",
                                "HIGS_BWD_H8_MR": "1",
                                "HIGS_BWD_ABSGRAD": "1"}},
        "verdict_identity": None,
    }
    if not ok:
        ident["verdict_identity"] = "ABORT_IDENTITY_MISMATCH"
        jwrite(os.path.join(out_root, "P17_SOURCE_IDENTITY.json"), ident)
        sys.exit(2)
    ident["verdict_identity"] = "IDENTITY_VERIFIED"
    jwrite(os.path.join(out_root, "P17_SOURCE_IDENTITY.json"), ident)
    log(f"{scene}: identity verified")


class SceneCtx:
    def __init__(self, scene):
        self.scene = scene
        bootstrap_c0()
        from dataset import GTDataset
        from gaussian_model import GaussianModel
        from config import ReferenceV1Config
        from gsplat.experimental.render.functional import (
            gaussian_inference as gi)
        self.gi = gi
        self._dyn = gi._HIGS_DYNAMIC_SCENE
        self.config = ReferenceV1Config()
        self.config.scene = scene
        self.dataset = GTDataset(scene=scene, repo_root=REPO,
                                 resolution="1080p", device="cuda",
                                 background="black")
        self.n_cameras = len(self.dataset)
        self.model = GaussianModel(max_sh_degree=self.config.sh_degree)
        ckpt = torch.load(CKPT_PATH[scene], map_location="cpu",
                          weights_only=False)
        self.model.restore(ckpt, {
            "position_lr_init": self.config.position_lr_init,
            "position_lr_final": self.config.position_lr_final,
            "position_lr_delay_mult": self.config.position_lr_delay_mult,
            "position_lr_max_steps": self.config.position_lr_max_steps,
            "feature_lr": self.config.feature_lr,
            "opacity_lr": self.config.opacity_lr,
            "scaling_lr": self.config.scaling_lr,
            "rotation_lr": self.config.rotation_lr,
            "percent_dense": self.config.percent_dense,
        })
        self.model.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.N = int(self.model._xyz.shape[0])
        self.it0 = int(ckpt.get("iteration", 16000))
        self.init_params = {n: self._param(n).detach().clone().contiguous()
                            for n in PARAM_NAMES}
        self.init_opt = {}
        for n in PARAM_NAMES:
            st = self.model.optimizer.state[self._param(n)]
            self.init_opt[n] = {k: st[k].clone()
                                for k in ("exp_avg", "exp_avg_sq", "step")}
        self._lr_by_param = {}
        for g in self.model.optimizer.param_groups:
            self._lr_by_param[id(g["params"][0])] = (g["name"], g["lr"])
        self.lr_by_name = {g["name"]: g["lr"]
                           for g in self.model.optimizer.param_groups}
        self._cam_cache = {}
        self.cur_ci = None
        # the C0 dynamic tracker is a module global; a scene switch or
        # topology change would require mark_dirty — P17 never changes
        # topology, so we simply mark dirty once at load (trainer does the
        # same) and reset() before every independent render.
        self._dyn.mark_dirty()
        log(f"{scene}: restored N={self.N:,} it0={self.it0} "
            f"n_cameras={self.n_cameras} active_sh={self.model.active_sh_degree}")

    def _param(self, name):
        return getattr(self.model, {"xyz": "_xyz", "shs": "_shs",
                                    "opacity": "_opacity",
                                    "scaling": "_scaling",
                                    "rotation": "_rotation"}[name])

    def group_lr(self, param):
        return self._lr_by_param[id(param)][1]

    def restore_initial(self):
        """Bit-exact restore of frozen 16K params + optimizer + fresh scene."""
        for n in PARAM_NAMES:
            p = self._param(n)
            p.data.copy_(self.init_params[n])
            p.grad = None
            st = self.model.optimizer.state[p]
            st["exp_avg"].copy_(self.init_opt[n]["exp_avg"])
            st["exp_avg_sq"].copy_(self.init_opt[n]["exp_avg_sq"])
            st["step"].copy_(self.init_opt[n]["step"])
        self._dyn.reset()

    def cam_gt(self, ci):
        if ci not in self._cam_cache:
            self._cam_cache[ci] = self.dataset.get_item(ci)
        return self._cam_cache[ci]

    def camera_record(self, ci):
        cam, gt = self.cam_gt(ci)
        return {"index": ci,
                "image_name": getattr(cam, "image_name", None)
                or getattr(cam, "img_name", None),
                "viewmatrix": cam.viewmatrix.detach().cpu().numpy().tolist(),
                "K": cam.K.detach().cpu().numpy().tolist(),
                "width": int(cam.image_width),
                "height": int(cam.image_height),
                "gt_sha256": sha256_bytes(
                    gt.contiguous().cpu().numpy().tobytes())}

    def fused_loss(self, image, gt, gt_coarse=None):
        """Frozen fused loss L = 0.8*L1 + 0.2*(1-SSIM).

        gt_coarse: deterministic area-averaged GT for coarse-resolution
        probes (prereg section 5); must match image size.
        """
        import fused_ssim
        g = gt_coarse if gt_coarse is not None else gt
        b = image.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
        g = g.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
        L1 = F.l1_loss(b, g)
        ssim = fused_ssim.fused_ssim(b, g, padding="same", train=True)
        return (1.0 - LAMBDA_DSSIM) * L1 + LAMBDA_DSSIM * (1.0 - ssim)

    def render_train(self, cam, K, width, height, eps2d):
        """Exact trainer compute path (publication_trainer_p12.py c0 arm)."""
        m = self.model
        out = self.gi.rasterize_gaussian_higs_dynamic(
            m.get_xyz, m.get_rotation, m.get_scaling, m.get_opacity,
            m.get_features,
            sh_degree=m.active_sh_degree,
            viewmats=cam.viewmatrix.unsqueeze(0).unsqueeze(0),
            Ks=K.unsqueeze(0).unsqueeze(0),
            width=width, height=height,
            backward_mode="higs_native", enable_culling=True,
            camera_model="pinhole", render_mode="RGB",
            eps2d=eps2d)
        frame = out["frame"].reshape(height, width, 3)
        return frame.clamp(0, 1), out

    def grad_groups(self, per_gauss=True):
        """Raw optimizer-space gradients (CPU float32 contiguous).

        per_gauss=True: per-Gaussian [N, ...] blocks — means/scales/quats/
        opacity [N, k], shs [N, 48], plus derived views sh0/shN [N, 3]/[N, 45].
        per_gauss=False: flat 1-D tensors in optimizer layout.
        """
        N = self.N
        g = {}
        for n in PARAM_NAMES:
            p = self._param(n)
            t = p.grad
            if t is None:
                t = torch.zeros_like(p.data)
            t = t.detach().float().contiguous().cpu()
            key = GROUP_KEYS[n]
            if per_gauss:
                g[key] = t.reshape(N, -1)
            else:
                g[key] = t.reshape(-1)
        nsh0 = N * 3
        if per_gauss:
            g["sh0"] = g["shs"][:, :3].clone()
            g["shN"] = g["shs"][:, 3:].clone()
        else:
            g["sh0"] = g["shs"][:nsh0].clone()
            g["shN"] = g["shs"][nsh0:].clone()
        return g

    def visible_ids(self, cam, K, width, height, eps2d):
        if not torch.is_tensor(K):
            K = torch.as_tensor(K, device="cuda", dtype=torch.float32)
        vm = cam.viewmatrix.unsqueeze(0).unsqueeze(0).contiguous()
        Kc = K.unsqueeze(0).unsqueeze(0).contiguous()
        with torch.no_grad():
            vis_ids, _, _ = self.gi._cull_gaussians_batched(
                self.model._xyz.detach(),
                F.normalize(self.model._rotation.detach(), dim=-1),
                torch.exp(self.model._scaling.detach()),
                vm, Kc, width, height, eps2d=eps2d, near_plane=0.01,
                far_plane=1e10, radius_clip=0.0, camera_model="pinhole")
        return vis_ids.cpu().numpy().astype(np.int64)

    def producer_full(self, cam, K, width, height, eps2d):
        """Producer over ALL Gaussians (master-space output [N, ...]).

        NOTE: rows of Gaussians not projected onto the image have
        UNINITIALIZED conic storage (radii == 0) — the same artifact
        documented for P2-1a-r3. Only rows with radii > 0 carry valid
        F9 conics; consumers must filter on the returned radii.
        """
        from gsplat.experimental.render.kernels import _backend as _ib
        backend = _ib._C
        m = self.model
        if not torch.is_tensor(K):
            K = torch.as_tensor(K, device="cuda", dtype=torch.float32)
        vm = cam.viewmatrix.unsqueeze(0).contiguous()
        Kc = K.unsqueeze(0).contiguous()
        N = m._xyz.shape[0]
        with torch.no_grad():
            cam_pos = backend.higs_camera_positions_from_viewmats(vm)
            (radii, means2d, depths, conics, op_bc,
             colors_eval) = backend.higs_gatherless_projected_producer(
                torch.arange(N, device="cuda", dtype=torch.int64),
                m._xyz.detach().contiguous(),
                F.normalize(m._rotation.detach(), dim=-1).contiguous(),
                torch.exp(m._scaling.detach()).contiguous(),
                torch.sigmoid(m._opacity.detach()).squeeze(-1).contiguous(),
                m.get_features.detach().contiguous(),
                vm, Kc, cam_pos, width, height, eps2d, 0.01, 1e10, 0.0)
        return (radii.reshape(-1, 2).cpu(), means2d.reshape(-1, 2).cpu(),
                depths.reshape(-1).cpu(), conics.reshape(-1, 3).cpu())

    def producer(self, cam, K, width, height, eps2d, vis_ids):
        """Exact C0 gatherless projected producer (P13-X/P14 metadata path)."""
        from gsplat.experimental.render.kernels import _backend as _ib
        backend = _ib._C
        m = self.model
        if not torch.is_tensor(K):
            K = torch.as_tensor(K, device="cuda", dtype=torch.float32)
        vm = cam.viewmatrix.unsqueeze(0).contiguous()
        Kc = K.unsqueeze(0).contiguous()
        with torch.no_grad():
            cam_pos = backend.higs_camera_positions_from_viewmats(vm)
            (radii, means2d, depths, conics, op_bc,
             colors_eval) = backend.higs_gatherless_projected_producer(
                torch.as_tensor(vis_ids, device="cuda", dtype=torch.int64),
                m._xyz.detach().contiguous(),
                F.normalize(m._rotation.detach(), dim=-1).contiguous(),
                torch.exp(m._scaling.detach()).contiguous(),
                torch.sigmoid(m._opacity.detach()).squeeze(-1).contiguous(),
                m.get_features.detach().contiguous(),
                vm, Kc, cam_pos, width, height, eps2d, 0.01, 1e10, 0.0)
        return (radii.reshape(-1, 2).cpu(), means2d.reshape(-1, 2).cpu(),
                depths.reshape(-1).cpu(), conics.reshape(-1, 3).cpu())

    def intersect_counts(self, cam, K, width, height, eps2d, vis_ids):
        """Exact per-Gaussian tile-intersection counts (P13-X/P14 path)."""
        from gsplat.experimental.render.kernels import _backend as _ib
        backend = _ib._C
        m = self.model
        if not torch.is_tensor(K):
            K = torch.as_tensor(K, device="cuda", dtype=torch.float32)
        tw, th = math.ceil(width / TILE), math.ceil(height / TILE)
        vm = cam.viewmatrix.unsqueeze(0).contiguous()
        Kc = K.unsqueeze(0).contiguous()
        with torch.no_grad():
            cam_pos = backend.higs_camera_positions_from_viewmats(vm)
            (radii, means2d, depths, conics, op_bc,
             _colors) = backend.higs_gatherless_projected_producer(
                torch.as_tensor(vis_ids, device="cuda", dtype=torch.int64),
                m._xyz.detach().contiguous(),
                F.normalize(m._rotation.detach(), dim=-1).contiguous(),
                torch.exp(m._scaling.detach()).contiguous(),
                torch.sigmoid(m._opacity.detach()).squeeze(-1).contiguous(),
                m.get_features.detach().contiguous(),
                vm, Kc, cam_pos, width, height, eps2d, 0.01, 1e10, 0.0)
            N = vis_ids.shape[0]
            radii = radii.reshape(1, 1, N, 2)
            means2d = means2d.reshape(1, 1, N, 2)
            depths = depths.reshape(1, 1, N)
            conics = conics.reshape(1, 1, N, 3)
            op_bc = op_bc.reshape(1, 1, N)
            _, isect_ids, flatten_ids = torch.ops.gsplat.intersect_tile(
                means2d, radii, depths, conics, op_bc, None, None, 1,
                TILE, tw, th, True, False)
        counts = torch.bincount(flatten_ids.reshape(-1), minlength=N)
        return (counts.cpu().numpy().astype(np.int64), int(isect_ids.numel()))


def area_avg_2d(x, d):
    """Deterministic area average over contiguous d x d blocks. [H,W,C]."""
    H, W, C = x.shape
    assert H % d == 0 and W % d == 0, f"{H}x{W} not divisible by {d}"
    y = x.reshape(H // d, d, W // d, d, C)
    return y.mean(axis=(1, 3))


def scale_K(K, d):
    K = K.astype(np.float64).copy()
    K[0, 0] /= d
    K[1, 1] /= d
    K[0, 2] /= d
    K[1, 2] /= d
    return K


def select_probe_ids(n_cam):
    ids = [int(math.floor((k + 0.5) * n_cam / N_PROBE)) for k in range(N_PROBE)]
    if len(set(ids)) < N_PROBE:
        # dedup: fill with nearest unused sorted IDs
        used = list(dict.fromkeys(ids))
        pool = sorted(set(range(n_cam)) - set(used))
        while len(used) < N_PROBE and pool:
            used.append(pool.pop(0))
        if len(used) < N_PROBE:
            raise RuntimeError("cannot build 12 probe cameras")
        ids = used
    return ids


def select_audit_ids(n_cam, probe_set):
    order = [i for i in range(n_cam) if i not in probe_set]
    out, used = [], set()
    for q in AUDIT_QUANTILES:
        p = q * (len(order) - 1)
        lo, hi = int(math.floor(p)), min(int(math.ceil(p)), len(order) - 1)
        cands = [lo] if lo == hi else (
            [lo, hi] if p - lo <= 0.5 else [hi, lo])
        for c in cands:
            if c not in used:
                used.add(c)
                out.append(order[c])
                break
    return out


def build_camera_set(ctx, out_root):
    probes = select_probe_ids(ctx.n_cameras)
    audits = select_audit_ids(ctx.n_cameras, set(probes))
    records = {str(ci): ctx.camera_record(ci)
               for ci in sorted(set(probes) | set(audits))}
    camset = {
        "scene": ctx.scene,
        "n_cameras": ctx.n_cameras,
        "algorithm": ("prereg section 4: sorted camera-ID quantiles, "
                      "j_k = floor((k+0.5)*N/12); audit = quantiles "
                      "0.2/0.4/0.6/0.8 of remaining sorted IDs"),
        "probe_ids": probes,
        "audit_ids": audits,
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "camera_records": records,
    }
    jwrite(os.path.join(out_root, "P17_CAMERA_SET.json"), camset)
    log(f"{ctx.scene}: CAMERA SET FROZEN probes={probes} audits={audits}")
    return camset


def sigma_min_from_conics(conics):
    """Minor/major-axis std (pixels) from 3-element normalized conics.

    F9 representation (verified by P2-1a-r3): (a, b, c) is the symmetric
    INVERSE 2D covariance upper triangle, i.e. the 2D Gaussian is
    exp(-0.5 (x-m)^T [[a,b],[b,c]] (x-m)).  Covariance eigenvalues are the
    reciprocal of the conic eigenvalues, so the minor-axis std is
    sqrt(1 / lambda_max(conic)).
    """
    a = conics[:, 0]; b = conics[:, 1]; c = conics[:, 2]
    tr = (a + c) / 2.0
    d = (a - c) / 2.0
    disc = torch.sqrt(d * d + b * b)
    lam_max = torch.clamp_min(tr + disc, 1e-12)
    lam_min = torch.clamp_min(tr - disc, 1e-12)
    return torch.sqrt(1.0 / lam_max), torch.sqrt(1.0 / lam_min)


def render_backward_grads(ctx, cam, K, width, height, eps2d,
                          gt_coarse=None):
    """Fresh-scene differentiable render + fused loss + backward.

    gt_coarse: [H,W,3] area-averaged GT for coarse probes (prereg s5).
    """
    ctx.restore_initial()
    image, out = ctx.render_train(cam, K, width, height, eps2d)
    gt = ctx.cam_gt(ctx.cur_ci)[1]
    loss = ctx.fused_loss(image, gt, gt_coarse=gt_coarse)
    loss.backward()
    grads = ctx.grad_groups()
    grads_flat = ctx.grad_groups(per_gauss=False)
    return (grads, grads_flat, image.detach().float().reshape(-1).cpu(),
            float(loss.item()),
            out["metadata"])


def cosine(a, b):
    na = float(torch.linalg.norm(a))
    nb = float(torch.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return None
    return float(torch.dot(a, b) / (na * nb + 1e-30))


def rel_l2(a, b):
    nb = float(torch.linalg.norm(b))
    if nb == 0.0:
        return None
    return float(torch.linalg.norm(a - b) / (nb + 1e-30))


def top10_sign(a, b):
    k = max(1, int(0.1 * b.numel()))
    topk = torch.topk(b.abs(), k).indices
    return float((torch.sign(a[topk]) == torch.sign(b[topk])).float().mean())


def metrics_for(ctx, pop, full, hyb):
    """Fidelity of hybrid vs full on the d>1 population, per raw parameter
    group (prereg section 10). Inputs are per-Gaussian [N, k] blocks; each
    group is restricted to the population and flattened before metrics."""
    out = {}
    for key in ("means", "scales", "quats", "opacity", "sh0", "shN"):
        fm = full[key][pop].reshape(-1)
        hm = hyb[key][pop].reshape(-1)
        c = cosine(fm, hm)
        r = rel_l2(hm, fm)
        s = top10_sign(hm, fm)
        out[key] = {
            "cosine": None if c is None else min(c, 1.0),
            "rel_l2": r,
            "top10_sign": s,
        }
    return out


def assign_d(sigma_min, tau):
    """d_i = max{d in {1,2,4,8}: sigma_min_i/d >= tau}, else 1.
    Vectorized: for each d, keep Gaussians passing that floor, overwrite in
    ascending d so the largest passing d wins.
    """
    d = torch.ones_like(sigma_min, dtype=torch.int64)
    for dd in (2, 4, 8):
        d = torch.where(sigma_min / dd >= tau, torch.full_like(d, dd), d)
    return d


def dense_adam_step(ctx, cam, K0, W0, H0):
    """One ordinary full-resolution C0 Adam step; returns L_before.

    Leaves the model in the POST-STEP state; the caller evaluates L_after
    with poststep_eval (which resets the dynamic scene but not params).
    """
    model = ctx.model
    it = STEP_T
    ctx.restore_initial()
    model.update_learning_rate(it)
    image, _ = ctx.render_train(cam, torch.as_tensor(
        K0, device="cuda", dtype=torch.float32), W0, H0, EPS2D_FULL)
    gt = ctx.cam_gt(ctx.cur_ci)[1]
    loss = ctx.fused_loss(image, gt, gt_coarse=None)
    L_before = float(loss.item())
    loss.backward()
    model.optimizer.step()
    model.optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return L_before


def poststep_eval(ctx, cam, K0, W0, H0, gt):
    """Exact full-res loss of the current (post-step) state on one camera.
    Resets only the dynamic scene (P13-Y); params stay stepped."""
    ctx._dyn.reset()
    with torch.no_grad():
        return float(ctx.fused_loss(
            ctx.render_train(cam, torch.as_tensor(
                K0, device="cuda", dtype=torch.float32), W0, H0,
                EPS2D_FULL)[0], gt).item())


def apply_hybrid_adam_step(ctx, cam, K0, W0, H0, hyb_grads):
    """One Adam step from the same cloned state using hybrid raw gradients.

    Exact torch.optim.Adam(non-amsgrad) semantics on the raw optimizer
    parameterization; t = STEP_T so the LR schedule and bias-correction
    match the dense reference exactly.
    """
    model = ctx.model
    it = STEP_T
    ctx.restore_initial()
    model.update_learning_rate(it)
    image, _ = ctx.render_train(cam, torch.as_tensor(
        K0, device="cuda", dtype=torch.float32), W0, H0, EPS2D_FULL)
    gt = ctx.cam_gt(ctx.cur_ci)[1]
    loss = ctx.fused_loss(image, gt, gt_coarse=None)
    L_before = float(loss.item())
    # drop autograd grads; we drive the optimizer manually below
    for n in PARAM_NAMES:
        ctx._param(n).grad = None
    bc1 = 1.0 - B1 ** it
    bc2 = 1.0 - B2 ** it
    for n in PARAM_NAMES:
        p = ctx._param(n)
        st = model.optimizer.state[p]
        lr = ctx.group_lr(p)
        g = hyb_grads[GROUP_KEYS[n]].to(p.device)
        if g.dim() > 1:
            g = g.reshape(p.data.shape)
        else:
            g = g.view_as(p.data)
        m_, v_ = st["exp_avg"], st["exp_avg_sq"]
        m_.lerp_(g, 1.0 - B1)
        v_.mul_(B2).addcmul_(g, g, value=1.0 - B2)
        denom = (v_.sqrt() / math.sqrt(bc2)).add_(EPS_ADAM)
        p.data.addcdiv_(m_, denom, value=-lr / bc1)
        st["step"] += 1
    torch.cuda.synchronize()
    return L_before


def audit_losses(ctx, cam_ids):
    """Evaluate the CURRENT (post-step) state on audit cameras.

    No restore_initial here: the caller runs this right after dense_adam_step
    / apply_hybrid_adam_step, which leave the model in the stepped state.
    Each render resets the dynamic scene (P13-Y) but never the parameters.
    """
    out = []
    for ci in cam_ids:
        ctx.cur_ci = ci
        cam, gt = ctx.cam_gt(ci)
        K = torch.as_tensor(
            cam.K.detach().cpu().numpy().astype(np.float64),
            device="cuda", dtype=torch.float32)
        with torch.no_grad():
            out.append(float(ctx.fused_loss(
                ctx.render_train(cam, K, cam.image_width,
                                 cam.image_height, EPS2D_FULL)[0], gt
            ).item()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, choices=("room", "bicycle"))
    ap.add_argument("--gpu", default="3")
    ap.add_argument("--out", default="/root/epic05_p17")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    out_root = args.out
    units_dir = os.path.join(out_root, "P17_UNITS", args.scene)
    raw_dir = os.path.join(out_root, "P17_RAW", args.scene)
    Path(units_dir).mkdir(parents=True, exist_ok=True)
    Path(raw_dir).mkdir(parents=True, exist_ok=True)

    verify_identity(out_root, args.scene)
    ctx = SceneCtx(args.scene)

    def smi(query):
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=" + query,
                                "--format=csv,noheader", "-i", str(args.gpu)],
                               capture_output=True, text=True, timeout=30)
            return r.stdout.strip()
        except Exception as e:
            return f"failed: {e}"

    def procs():
        try:
            r = subprocess.run(["nvidia-smi",
                                "--query-compute-apps=gpu_uuid,pid,used_memory",
                                "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=30)
            return r.stdout.strip()
        except Exception as e:
            return f"failed: {e}"

    contamination = {
        "scene": args.scene, "gpu_index": int(args.gpu),
        "gpu_uuid": smi("uuid"),
        "procs_before": procs(),
        "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    camset = build_camera_set(ctx, out_root)
    probe_ids, audit_ids = camset["probe_ids"], camset["audit_ids"]

    jwrite(os.path.join(out_root, "P17_STATE_IDENTITY.json"), {
        "scene": args.scene,
        "checkpoint": CKPT_PATH[args.scene],
        "checkpoint_sha256": CKPT_SHA[args.scene],
        "iteration": ctx.it0,
        "N_gaussians": ctx.N,
        "active_sh_degree": int(ctx.model.active_sh_degree),
        "spatial_lr_scale": float(ctx.model.spatial_lr_scale),
        "lr_by_name": {k: float(v) for k, v in ctx.lr_by_name.items()},
        "optimizer": "torch.optim.Adam betas=(0.9,0.999) eps=1e-15, "
                     "restored verbatim from checkpoint optimizer_state_dict",
        "step_iteration": STEP_T,
        "frozen_note": "params+optimizer cloned bit-exact per arm; no "
                       "densification/prune/opacity reset; fresh "
                       "_HIGS_DYNAMIC_SCENE.reset() before every render",
    })

    parity_rows = {}
    n_units = 0
    for ci in probe_ids:
        ctx.cur_ci = ci
        cam, gt = ctx.cam_gt(ci)
        H0, W0 = int(cam.image_height), int(cam.image_width)
        K0 = cam.K.detach().cpu().numpy().astype(np.float64)
        K0t = torch.as_tensor(K0, device="cuda", dtype=torch.float32)
        camK = {d: torch.as_tensor(scale_K(K0, d), device="cuda",
                                   dtype=torch.float32) for d in DS}

        # ---------------- d=1 parity: reference vs oracle --------------
        # The reference backward leaves the full-res raw grads in the model;
        # capture them immediately (grad_groups on a later restore would read
        # None grads and silently return zeros).
        _, ref_flat, ref_frame, ref_loss, ref_meta = render_backward_grads(
            ctx, cam, K0t, W0, H0, EPS2D_FULL)
        ref_pg = ctx.grad_groups()  # per-Gaussian [N, ...] blocks
        _, orb_flat, orb_frame, orb_loss, orb_meta = render_backward_grads(
            ctx, cam, K0t, W0, H0, EPS2D_FULL)
        rgb_max = float((ref_frame - orb_frame).abs().max())
        loss_rel = abs(ref_loss - orb_loss) / (abs(ref_loss) + 1e-30)
        grad_cos = {}
        all_finite = True
        for key in ("means", "scales", "quats", "opacity", "sh0", "shN"):
            if not (torch.isfinite(ref_flat[key]).all()
                    and torch.isfinite(orb_flat[key]).all()):
                all_finite = False
                continue
            grad_cos[key] = cosine(ref_flat[key], orb_flat[key])
        vis1 = ctx.visible_ids(cam, K0, W0, H0, EPS2D_FULL)
        vis1b = ctx.visible_ids(cam, K0, W0, H0, EPS2D_FULL)
        vis_eq = bool(np.array_equal(np.sort(vis1), np.sort(vis1b)))
        parity_pass = (rgb_max <= 1e-6 and loss_rel <= 1e-7 and all_finite
                       and vis_eq and all(
                           v is not None and v >= 0.999999
                           for v in grad_cos.values()))
        parity_rows[ci] = {
            "rgb_max_abs": rgb_max, "loss_ref": ref_loss,
            "loss_oracle": orb_loss, "loss_rel_error": loss_rel,
            "grad_cosine": grad_cos, "all_finite": bool(all_finite),
            "visible_id_set_equal": vis_eq, "n_visible": int(vis1.shape[0]),
            "pass": bool(parity_pass),
        }
        log(f"{args.scene} cam{ci}: d1 parity rgb_max={rgb_max:.3e} "
            f"loss_rel={loss_rel:.3e} vis_eq={vis_eq} "
            f"cos={ {k: (f'{v:.6f}' if v is not None else 'NA') for k, v in grad_cos.items()} }"
            f" -> {'PASS' if parity_pass else 'FAIL'}")
        if not parity_pass:
            jwrite(os.path.join(units_dir, f"camera_{ci}.json"), {
                "scene": args.scene, "camera_id": ci, "tau": 2.0,
                "d1_parity_pass": False,
                "oracle_interaction_ratio": None,
                "full_cost_fraction_d2plus": None,
                "full_cost_fraction_d4plus": None,
                "full_cost_fraction_d8": None,
                "metrics": {k: {"cosine": None, "rel_l2": None,
                                "top10_sign": None}
                            for k in ("means", "scales", "quats", "opacity",
                                      "sh0", "shN")},
                "step": {"reference_improving": None, "R_step": None,
                         "P_audit": None}})
            n_units += 1
            continue

        # ---------------- coarse probes d = 2/4/8 -----------------------
        coarse, coarse_vis = {}, {}
        gt_np = gt.cpu().numpy()
        for d in DS:
            Hd, Wd = H0 // d, W0 // d
            eps_d = EPS2D_FULL / (d * d)
            gt_coarse = torch.as_tensor(
                area_avg_2d(gt_np, d), device="cuda", dtype=torch.float32)
            _, g_d, frame_d, loss_d, meta_d = render_backward_grads(
                ctx, cam, camK[d], Wd, Hd, eps_d, gt_coarse=gt_coarse)
            vis_d = ctx.visible_ids(cam, scale_K(K0, d), Wd, Hd, eps_d)
            g_d_pg = {k: v.reshape(ctx.N, -1) for k, v in g_d.items()}
            coarse[d] = {"grads": g_d_pg, "flat": g_d, "loss": loss_d}
            coarse_vis[d] = vis_d
            log(f"{args.scene} cam{ci}: d={d} n_vis={vis_d.shape[0]} "
                f"loss={loss_d:.6f}")

        # full-res exact metadata (counts, conics, sigmas)
        counts1, n_isects1 = ctx.intersect_counts(
            cam, K0, W0, H0, EPS2D_FULL, vis1)
        # conics computed in MASTER space so sigma indexing is 1:1 with
        # the parameter index (the prereg routing rule is per-Gaussian and
        # view-dependent; the assignment itself uses no gradient data).
        # Only validly-projected rows (radii>0) have valid conic storage;
        # unprojected rows are assigned d=1 (sigma=0 cannot pass any
        # sigma/d >= tau floor, so the floor check is safe).
        rad1, _, _, conics1 = ctx.producer_full(
            cam, K0, W0, H0, EPS2D_FULL)
        vis_rows = rad1[:, 0] > 0
        sig_min1, sig_max1 = sigma_min_from_conics(conics1)
        sig_min1[~vis_rows] = 0.0
        sig_max1[~vis_rows] = 0.0
        # per-cam sigma stats for logging: only valid projected rows (radii>0)
        if bool(vis_rows.any()):
            sigv = sig_min1[vis_rows]
            log(f"{args.scene} cam{ci}: sigma_min(visible) "
                f"p10={float(torch.quantile(sigv, 0.1)):.2f} "
                f"med={float(torch.median(sigv)):.2f} "
                f"p90={float(torch.quantile(sigv, 0.9)):.2f} "
                f"max={float(sigv.max()):.2f}")
        coarse_counts = {}
        for d in DS:
            coarse_counts[d], _ = ctx.intersect_counts(
                cam, scale_K(K0, d), W0 // d, H0 // d, EPS2D_FULL / (d * d),
                coarse_vis[d])

        # ---- raw payloads ------------------------------------------------
        npz = {
            "counts_1": counts1, "vis_ids_1": vis1,
            "conics_1": conics1.numpy(),
            "sig_min_1": sig_min1.numpy(), "sig_max_1": sig_max1.numpy(),
            "K_full": K0, "K_2": scale_K(K0, 2), "K_4": scale_K(K0, 4),
            "K_8": scale_K(K0, 8),
            "gt_area_avg_d2": area_avg_2d(
                gt.cpu().numpy(), 2),
            "gt_area_avg_d4": area_avg_2d(gt.cpu().numpy(), 4),
            "gt_area_avg_d8": area_avg_2d(gt.cpu().numpy(), 8),
        }
        for d in DS:
            npz[f"counts_{d}"] = coarse_counts[d]
        np.savez_compressed(
            os.path.join(raw_dir, f"camera_{ci}_footprints.npz"), **npz)
        np.savez_compressed(
            os.path.join(raw_dir, f"camera_{ci}_grad_full.npz"),
            **{f"grad_{k}": v.numpy() for k, v in ref_flat.items()})
        for d in DS:
            np.savez_compressed(
                os.path.join(raw_dir, f"camera_{ci}_grad_d{d}.npz"),
                **{f"grad_{k}": v.numpy()
                   for k, v in coarse[d]["flat"].items()})

        # ---- tau assignments + fidelity (primary tau=2 drives the unit) --
        # ref_pg: full-res raw grads captured right after the parity backward
        ref_grads = ref_pg
        tau_blocks = {}
        for tau in TAUS:
            d_full = assign_d(sig_min1.cpu(), tau)  # [N] CPU
            N = ctx.N
            hyb = {k: torch.zeros_like(ref_grads[k]) for k in ref_grads}
            assigned = torch.zeros(N, dtype=torch.int64)
            cost_by_d = torch.zeros(N, dtype=torch.float64)
            vis1_t = torch.as_tensor(vis1, dtype=torch.int64)
            cnt1_full = torch.zeros(N, dtype=torch.float64)
            cnt1_full[vis1_t] = torch.as_tensor(
                counts1, dtype=torch.float64)
            for d in DS:
                sel = (d_full == d)
                if not bool(sel.any()):
                    continue
                vis_d = torch.as_tensor(coarse_vis[d], dtype=torch.int64)
                cnt_d_full = torch.zeros(N, dtype=torch.float64)
                cnt_d_full[vis_d] = torch.as_tensor(
                    coarse_counts[d], dtype=torch.float64)
                for k in ref_grads:
                    if k in ("sh0", "shN"):
                        continue  # sh0/shN are views of the shs block
                    g_d_full = torch.zeros_like(ref_grads[k])
                    g_d_full[vis_d] = coarse[d]["grads"][k][: vis_d.numel()]
                    hyb[k] = torch.where(sel.unsqueeze(1), g_d_full, hyb[k])
                assigned[sel] = d
                cost_by_d[sel] = cnt_d_full[sel]
            cnt1_tot = float(cnt1_full.sum())
            C_oracle = (float(cost_by_d.sum()) / cnt1_tot
                        if cnt1_tot > 0 else None)
            fr = {}
            for dd, lab in ((2, "d2plus"), (4, "d4plus"), (8, "d8")):
                mask = assigned >= dd
                fr[lab] = (float(cnt1_full[mask].sum()) / cnt1_tot
                           if cnt1_tot > 0 else None)
            pop_full = torch.zeros(N, dtype=torch.bool)
            pop_full[torch.where(assigned > 1)[0]] = True
            n_d2 = int(pop_full.sum())
            # fidelity metrics (prereg s10): per-Gaussian blocks restricted
            # to the d>1 population, flattened per group
            mtr = (metrics_for(ctx, pop_full, ref_grads, hyb) if n_d2 > 0
                   else {k: {"cosine": None, "rel_l2": None,
                             "top10_sign": None}
                         for k in ("means", "scales", "quats", "opacity",
                                   "sh0", "shN")})
            tau_blocks[tau] = {
                "hyb": hyb,
                "n_assigned_d2plus": n_d2,
                "cost_by_d": {"d1": int((d_full == 1).sum().item()),
                              "d2": int((d_full == 2).sum().item()),
                              "d4": int((d_full == 4).sum().item()),
                              "d8": int((d_full == 8).sum().item())},
                "oracle_interaction_ratio": C_oracle,
                "full_cost_fraction_d2plus": fr["d2plus"],
                "full_cost_fraction_d4plus": fr["d4plus"],
                "full_cost_fraction_d8": fr["d8"],
                "metrics": mtr,
            }

        # ---- primary tau=2: hybrid one-step Adam oracle -----------------
        # The hybrid gradient is the d=1 gradient for d=1-assigned Gaussians
        # and the coarse-d gradient for coarse-assigned ones; both were
        # captured from independent fresh-scene backwar
        tau2 = tau_blocks[2.0]
        gt_probe = ctx.cam_gt(ci)[1]
        L_before_d = dense_adam_step(ctx, cam, K0, W0, H0)
        L_after_dense = poststep_eval(ctx, cam, K0, W0, H0, gt_probe)
        L_before_h = apply_hybrid_adam_step(
            ctx, cam, K0, W0, H0,
            {k: v.reshape(-1) for k, v in tau2["hyb"].items()})
        L_after_hyb = poststep_eval(ctx, cam, K0, W0, H0, gt_probe)
        assert abs(L_before_d - L_before_h) < 1e-12, "L_before mismatch"
        G_dense = L_before_d - L_after_dense
        G_hyb = L_before_h - L_after_hyb
        # prereg section 11: "If G_dense <= 0, mark the probe
        # STEP_NONIMPROVING_REFERENCE". Literal trigger: zero or negative.
        ref_improving = bool(G_dense > 0.0)
        R_step = (G_hyb / max(G_dense, 1e-12)) if ref_improving else None
        ctx.restore_initial()
        L_audit_before = audit_losses(ctx, audit_ids)
        # re-derive post-step states for audit (same steps, re-executed)
        dense_adam_step(ctx, cam, K0, W0, H0)
        L_audit_after_dense = audit_losses(ctx, audit_ids)
        apply_hybrid_adam_step(
            ctx, cam, K0, W0, H0,
            {k: v.reshape(-1) for k, v in tau2["hyb"].items()})
        L_audit_after_hyb = audit_losses(ctx, audit_ids)
        P_audit_vals = [
            (L_audit_after_hyb[i] - L_audit_after_dense[i])
            / (L_audit_before[i] + 1e-30) for i in range(len(audit_ids))]
        P_audit = float(np.mean(P_audit_vals))

        unit = {
            "scene": args.scene,
            "camera_id": ci,
            "tau": 2.0,
            "d1_parity_pass": True,
            "oracle_interaction_ratio": tau2["oracle_interaction_ratio"],
            "full_cost_fraction_d2plus": tau2["full_cost_fraction_d2plus"],
            "full_cost_fraction_d4plus": tau2["full_cost_fraction_d4plus"],
            "full_cost_fraction_d8": tau2["full_cost_fraction_d8"],
            "metrics": {k: {kk: v[kk] for kk in ("cosine", "rel_l2",
                                                 "top10_sign")}
                        for k, v in tau2["metrics"].items()},
            "step": {"reference_improving": ref_improving,
                     "R_step": R_step, "P_audit": P_audit},
            "detail": {
                "n_visible_full": int(vis1.shape[0]),
                "n_isects_full": n_isects1,
                "n_assigned_d2plus": tau2["n_assigned_d2plus"],
                "cost_by_d": tau2["cost_by_d"],
                "L_before": L_before_d,
                "L_after_dense_probe": L_after_dense,
                "L_after_hyb_probe": L_after_hyb,
                "G_dense": G_dense, "G_hyb": G_hyb,
                "P_audit_per_cam": P_audit_vals,
                "L_audit_before": L_audit_before,
                "L_audit_after_dense": L_audit_after_dense,
                "L_audit_after_hyb": L_audit_after_hyb,
                "sensitivity": {
                    str(t): {
                        "n_assigned_d2plus": tau_blocks[t]["n_assigned_d2plus"],
                        "oracle_interaction_ratio": tau_blocks[t]["oracle_interaction_ratio"],
                        "cost_by_d": tau_blocks[t]["cost_by_d"],
                        "metrics": {
                            k: {kk: v[kk] for kk in
                                ("cosine", "rel_l2", "top10_sign")}
                            for k, v in tau_blocks[t]["metrics"].items()},
                    } for t in (1.0, 4.0)},
            },
        }
        jwrite(os.path.join(units_dir, f"camera_{ci}.json"), unit)
        n_units += 1
        log(f"{args.scene} cam{ci}: C_oracle={tau2['oracle_interaction_ratio']:.4f} "
            f"n_d2+={tau2['n_assigned_d2plus']} "
            f"means_cos={tau2['metrics']['means']['cosine']} "
            f"R_step={R_step} P_audit={P_audit:.3e}")

    jwrite(os.path.join(out_root, "P17_D1_PARITY.json"),
           {"scene": args.scene, "cameras": parity_rows})

    contamination["end_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                             time.gmtime())
    contamination["procs_after"] = procs()
    try:
        contamination["peak_memory_mib"] = int(
            torch.cuda.max_memory_allocated() / (1024 ** 2))
    except Exception:
        pass
    jwrite(os.path.join(out_root, "P17_CONTAMINATION.json"),
           {args.scene: contamination})
    log(f"{args.scene}: DONE units={n_units}")


if __name__ == "__main__":
    main()
