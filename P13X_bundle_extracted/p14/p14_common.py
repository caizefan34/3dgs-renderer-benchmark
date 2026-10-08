"""P14 FreezeGS / Optimization-Induced Stationarity Oracle - shared library.

Mirrors the exact frozen C0 stack verified in P12/P13-X:
  bootstrap_c0 / loss / render / restore are byte-for-byte reuses of the
  p13x_x6_oracle.py patterns (validated on the same checkpoints).
"""
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
import zlib

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------- paths
REPO = "/root/3dgs-renderer-benchmark"
C0_WT = "/root/epic05_frozen/higs_c0_final30k_worktree"
C0_SO = ("/root/epic05_build_cache/c0_ext/"
         "experimental_gaussian_render_inference_scene_cuda/"
         "experimental_gaussian_render_inference_scene_cuda.so")
CORE_SO = "/root/epic05_build_cache/core/gsplat_cuda/gsplat_cuda.so"
SCENE_CUDA_DIR = "/root/epic05_build_cache/scene/gsplat_scene_cuda"
REF_V1 = os.path.join(REPO, "baseline", "reference_v1")
PHASE7 = os.path.join(REPO, "scripts", "epic05", "phase7")
SRC = os.path.join(REPO, "src")
P14_ROOT = "/root/epic05_p14"
P13X_ROOT = "/root/epic05_p13x"

# frozen baseline identities (P12/P13-X prereg; any mismatch => ABORT)
PREREG_HASHES = {
    "c0_ext_so": "d26cccf46c7932bfb5fd44ee05e553af27e17c93eaffb7dacf3197258a513f11",
    "core_so": "935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e",
    "fused_ssim_so": "dd6a4c799b98ff9fa82b4203acce3b36b1529c97155c0ef4cd50fd0f583d8ebb",
    "p12_trainer": "675b0267f3a1a6a0fa63fadb37c8536481c1c7a6de37a4b7f6ffa98a449050c5",
}

SCENES = ("room", "bicycle")
CKPT = {
    "room": "/root/epic05_p13x/runs/room_diag/ckpt_16000.pt",
    "bicycle": "/root/epic05_p13x/runs/bicycle_diag/ckpt_16000.pt",
}
Q_LIST_STAGE_A = (0.10, 0.20, 0.25, 0.30, 0.50)
Q_LIST_LE25 = (0.10, 0.20, 0.25)
H_STAGE_A = 32
H_STAGE_B = (8, 128)
LAMBDA_DSSIM = 0.2
EPS2D = 0.3
TILE = 16
CACHE_SEG_BYTES_FP32 = 256 * (4 * 4) + 16   # 4112 (C_R,C_G,C_B,T @ FP32 + meta)
CACHE_SEG_BYTES_FP16 = 256 * (4 * 2) + 16  # 2064
BW_BYTES_PER_S = 1.0e12                    # conservative A100 HBM write charge

PARAM_NAMES = ("xyz", "shs", "opacity", "scaling", "rotation")
SMOKE = os.environ.get("P14_SMOKE") == "1"


def o1_dir():
    return os.path.join(P14_ROOT,
                        "P14_O1_UNITS_SMOKE" if SMOKE else "P14_O1_UNITS")


def o2_dir():
    return os.path.join(P14_ROOT,
                        "P14_O2_UNITS_SMOKE" if SMOKE else "P14_O2_UNITS")


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b):
    import hashlib
    return hashlib.sha256(b).hexdigest()


def load_ext_module(name, so_path):
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
    core = load_ext_module("gsplat_cuda", CORE_SO)
    sys.modules["gsplat.csrc"] = core
    exp = load_ext_module(
        "experimental_gaussian_render_inference_scene_cuda", C0_SO)
    sys.modules["gsplat.experimental.render.kernels.csrc"] = exp
    sys.modules["experimental_gaussian_render_inference_scene_cuda"] = exp
    sys.path.insert(0, PHASE7)
    sys.path.insert(0, SRC)
    sys.path.insert(0, REF_V1)


def gpu_snapshot():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader"], capture_output=True, text=True,
            timeout=20)
        return out.stdout.strip()
    except Exception as e:  # pragma: no cover
        return f"snapshot failed: {e}"


def p12_camera_sequence(n_cameras, iterations, seed=42):
    """Byte-identical mirror of the trainer's frozen camera sequence."""
    import random
    viewpoint_stack = list(range(n_cameras))
    seq = []
    rng = random.Random(seed)
    for _ in range(1, iterations + 1):
        if not viewpoint_stack:
            viewpoint_stack = list(range(n_cameras))
        rand_idx = rng.randint(0, len(viewpoint_stack) - 1)
        seq.append(viewpoint_stack.pop(rand_idx))
    return np.array(seq, dtype=np.int32)


# ---------------------------------------------------------------- scene ctx
class SceneCtx:
    """Dataset + restored model + exact render/loss/eval/stream helpers."""

    def __init__(self, scene, gpu="4", reps_note=""):
        self.scene = scene
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu)
        bootstrap_c0()
        from dataset import GTDataset
        from gaussian_model import GaussianModel
        from config import ReferenceV1Config
        self.config = ReferenceV1Config()
        self.config.scene = scene
        self.dataset = GTDataset(scene=scene, repo_root=REPO,
                                 resolution="1080p", device="cuda",
                                 background="black")
        self.n_cameras = len(self.dataset)
        self.model = GaussianModel(max_sh_degree=self.config.sh_degree)
        ckpt = torch.load(CKPT[scene], map_location="cpu",
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
        # frozen initial state (bitwise identical clone for every arm)
        self.init_params = {
            n: self._param(n).detach().clone().contiguous()
            for n in PARAM_NAMES}
        self.init_opt = {}
        for n in PARAM_NAMES:
            st = self.model.optimizer.state[self._param(n)]
            self.init_opt[n] = {
                "exp_avg": st["exp_avg"].clone(),
                "exp_avg_sq": st["exp_avg_sq"].clone(),
                "step": st["step"].clone(),
            }
        self._cam_cache = {}
        # the C0 dynamic-forward tracker is a module global that pins the
        # Gaussian count across forwards; loading a different scene's
        # model in the same process must mark the scene dirty (the
        # trainer does the same when densify/prune changes N; P14 never
        # changes N within a scene)
        from gsplat.experimental.render.functional.gaussian_inference \
            import _HIGS_DYNAMIC_SCENE
        _HIGS_DYNAMIC_SCENE.mark_dirty()
        print(f"[p14:{scene}] restored N={self.N:,} it0={self.it0} "
              f"n_cameras={self.n_cameras}", flush=True)

    # ---- params ------------------------------------------------------
    def _param(self, name):
        return {"xyz": self.model._xyz, "shs": self.model._shs,
                "opacity": self.model._opacity,
                "scaling": self.model._scaling,
                "rotation": self.model._rotation}[name]

    def restore_initial(self):
        for n in PARAM_NAMES:
            self._param(n).data.copy_(self.init_params[n])
            st = self.model.optimizer.state[self._param(n)]
            st["exp_avg"].copy_(self.init_opt[n]["exp_avg"])
            st["exp_avg_sq"].copy_(self.init_opt[n]["exp_avg_sq"])
            st["step"].copy_(self.init_opt[n]["step"])

    # ---- cameras -----------------------------------------------------
    def cam_gt(self, ci):
        if ci not in self._cam_cache:
            cam, gt = self.dataset.get_item(ci)
            self._cam_cache[ci] = (cam, gt)
        return self._cam_cache[ci]

    # ---- loss --------------------------------------------------------
    def fused_loss(self, image, gt):
        import fused_ssim as fused_ssim_mod
        b = image.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
        g = gt.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
        L1 = F.l1_loss(b, g)
        ssim = fused_ssim_mod.fused_ssim(b, g, padding="same", train=True)
        return (1.0 - LAMBDA_DSSIM) * L1 + LAMBDA_DSSIM * (1.0 - ssim)

    # ---- differentiable training render (exact trainer compute path;
    # _higs_dynamic_forward is the inner Function the trainer's wrapper
    # calls — used directly (validated P13-X X6 pattern) because the
    # wrapper's densification tracker is per-process global state that
    # cannot track two scenes in one process; P14 never densifies) ----
    def render_train(self, cam):
        from gsplat.experimental.render.functional.gaussian_inference \
            import _higs_dynamic_forward
        m = self.model
        out = _higs_dynamic_forward(
            m.get_xyz, m.get_rotation, m.get_scaling, m.get_opacity,
            m.get_features,
            viewmats=cam.viewmatrix.unsqueeze(0).unsqueeze(0),
            Ks=cam.K.unsqueeze(0).unsqueeze(0),
            width=cam.image_width, height=cam.image_height,
            sh_degree=m.active_sh_degree,
            backward_mode="higs_native", enable_culling=True,
            camera_model="pinhole", render_mode="RGB", eps2d=EPS2D)
        frame = out["frame"].reshape(cam.image_height,
                                     cam.image_width, 3)
        return frame.clamp(0, 1)

    def render_eval(self, cam):
        with torch.no_grad():
            return self.render_train(cam)

    # ---- eval --------------------------------------------------------
    def eval_cameras(self, cam_ids):
        losses, psnrs = [], []
        for ci in cam_ids:
            cam, gt = self.cam_gt(ci)
            image = self.render_eval(cam)
            losses.append(float(self.fused_loss(image, gt)))
            mse = float(torch.mean((image - gt) ** 2))
            psnrs.append(-10.0 * math.log10(max(mse, 1e-12)))
        return {"loss": float(np.mean(losses)),
                "psnr": float(np.mean(psnrs)),
                "losses": losses, "psnrs": psnrs}

    # ---- stream extraction (exact C0 pieces, no raster needed) -------
    def stream_pieces(self, cam):
        """Cull -> F9 producer -> intersect_tile. Returns dict with
        vis ids, per-isect global ids, tile ids, offsets, tw, th."""
        from gsplat.cuda._wrapper import isect_offset_encode
        from gsplat.experimental.render.kernels import _backend as _ib
        from gsplat.experimental.render.functional.gaussian_inference \
            import _cull_gaussians_batched
        backend = _ib._C
        m = self.model
        width, height = int(cam.image_width), int(cam.image_height)
        tw, th = math.ceil(width / TILE), math.ceil(height / TILE)
        vm = cam.viewmatrix.unsqueeze(0).unsqueeze(0).contiguous()
        K = cam.K.unsqueeze(0).unsqueeze(0).contiguous()
        means = m._xyz.detach()
        quats = F.normalize(m._rotation.detach(), dim=-1)
        scales = torch.exp(m._scaling.detach())
        opacities = torch.sigmoid(m._opacity.detach()).squeeze(-1)
        colors = m.get_features.detach().contiguous()
        with torch.no_grad():
            vis_ids, _, _ = _cull_gaussians_batched(
                means, quats, scales, vm, K, width, height,
                eps2d=EPS2D, near_plane=0.01, far_plane=1e10,
                radius_clip=0.0, camera_model="pinhole")
            if vis_ids.numel() == 0:
                return {"empty": True, "tw": tw, "th": th,
                        "n_tiles": tw * th}
            cam_pos = backend.higs_camera_positions_from_viewmats(
                vm[0].contiguous())
            radii, means2d, depths, conics, op_bc, colors_eval = (
                backend.higs_gatherless_projected_producer(
                    vis_ids.contiguous(), means.contiguous(),
                    quats.contiguous(), scales.contiguous(),
                    opacities.contiguous(), colors.contiguous(),
                    vm[0].contiguous(), K[0].contiguous(), cam_pos,
                    width, height, EPS2D, 0.01, 1e10, 0.0))
            C, N = 1, vis_ids.numel()
            radii = radii.reshape(1, C, N, 2)
            means2d = means2d.reshape(1, C, N, 2)
            depths = depths.reshape(1, C, N)
            conics = conics.reshape(1, C, N, 3)
            op_bc = op_bc.reshape(1, C, N)
            _, isect_ids, flatten_ids = torch.ops.gsplat.intersect_tile(
                means2d, radii, depths, conics, op_bc, None, None, C,
                TILE, tw, th, True, False)
        vis = vis_ids.cpu().numpy().astype(np.int64)
        flat = flatten_ids.cpu().numpy().astype(np.int64)
        iids = isect_ids.cpu().numpy().astype(np.int64)
        master_flat = vis[flat]
        n_tiles = th * tw
        tile_bits = max(1, (n_tiles - 1).bit_length())
        tile_of_isect = (iids >> 32) & ((1 << tile_bits) - 1)
        offsets = isect_offset_encode(
            isect_ids, C, tw, th).cpu().numpy().astype(
            np.int64).reshape(-1)
        return {"empty": False, "vis_ids": vis, "master_flat": master_flat,
                "tile_of_isect": tile_of_isect, "offsets": offsets,
                "tw": tw, "th": th, "n_tiles": n_tiles,
                "n_isects": int(master_flat.shape[0]),
                "n_visible": int(vis.shape[0])}

    def visible_set(self, cam):
        """V_c from the REAL C0 renderer culling (no CPU approximation)."""
        from gsplat.experimental.render.functional.gaussian_inference \
            import _cull_gaussians_batched
        m = self.model
        width, height = int(cam.image_width), int(cam.image_height)
        vm = cam.viewmatrix.unsqueeze(0).unsqueeze(0).contiguous()
        K = cam.K.unsqueeze(0).unsqueeze(0).contiguous()
        with torch.no_grad():
            vis_ids, _, _ = _cull_gaussians_batched(
                m._xyz.detach(), F.normalize(m._rotation.detach(), dim=-1),
                torch.exp(m._scaling.detach()),
                vm, K, width, height, eps2d=EPS2D, near_plane=0.01,
                far_plane=1e10, radius_clip=0.0, camera_model="pinhole")
        return vis_ids.cpu().numpy().astype(np.int64)


def _selftest_tile_stream():
    """Synthetic-stream regression check for tile_stream_hashes.

    tile0 = [A A F F F A F] -> 3 active isects, 2 frozen segments,
             3 A/F boundaries; tile1 = [F F F] -> 1 segment;
    tile2 = [A A] -> 0 segments. Totals: n_act=5, n_seg=3, n_bound=3.
    """
    master = np.array([5, 6, 7, 8, 9, 10, 11, 20, 21, 22, 30, 31],
                      dtype=np.int64)
    active = np.zeros(32, dtype=bool)
    active[[5, 6, 10, 30, 31]] = True
    offsets = np.array([0, 7, 10], dtype=np.int64)  # start convention
    seq, mem, tlen, n_act, n_seg, n_bound = tile_stream_hashes(
        master, None, offsets, active, 3)
    assert (n_act, n_seg, n_bound) == (5, 3, 3), \
        f"tile_stream_hashes selftest FAILED: {(n_act, n_seg, n_bound)}"
    assert list(tlen) == [7, 3, 2]


# ---------------------------------------------------------------- streams
def tile_stream_hashes(master_flat, tile_of_isect, offsets, active_bool,
                       n_tiles):
    """Per-tile sequence hash + membership hash + counts.

    Returns (seq_crc [n_tiles] uint32, len [n_tiles], n_active_isect,
    n_frozen_seg, n_boundaries) for ONE camera.
    Fully deterministic: crc32 over int64 (global_id, active_flag) pairs
    in tile order; membership crc over sorted global ids per tile.
    """
    lab = active_bool[master_flat]     # True = ACTIVE gaussian isect
    frz = ~lab                          # True = FROZEN gaussian isect
    M = master_flat.shape[0]
    # segment starts: FROZEN runs (cacheable constant-alpha segments):
    # frz & (first-of-tile | prev-isect-not-frozen)
    is_tile_start = np.zeros(M, dtype=bool)
    tile_start_idx = offsets[offsets < M]  # starts of nonempty tiles
    is_tile_start[tile_start_idx] = True
    prev_frz = np.zeros(M, dtype=bool)
    prev_frz[1:] = frz[:-1]
    seg_start = frz & (is_tile_start | ~prev_frz)
    n_seg = int(seg_start.sum())
    # active/frozen boundaries (transitions within tiles, either direction)
    prev_lab = np.zeros(M, dtype=bool)
    prev_lab[1:] = lab[:-1]
    boundary = (lab != prev_lab) & ~is_tile_start
    n_bound = int(boundary.sum())
    n_act = int(lab.sum())
    # per-tile crc (python loop over tiles; ~n_tiles * 20us)
    seq_crc = np.zeros(n_tiles, dtype=np.uint64)
    mem_crc = np.zeros(n_tiles, dtype=np.uint64)
    tlen = np.zeros(n_tiles, dtype=np.int64)
    packed = (master_flat << 1) | lab.astype(np.int64)
    olen = offsets.shape[0]
    starts = offsets[:n_tiles] if olen >= n_tiles else np.pad(
        offsets, (0, n_tiles - olen))
    ends = np.empty(n_tiles, dtype=np.int64)
    for t in range(n_tiles):
        s = int(starts[t])
        e = int(starts[t + 1]) if t + 1 < olen else M
        if e < s:
            e = s
        tlen[t] = e - s
        if e > s:
            seq_crc[t] = zlib.crc32(packed[s:e].tobytes()) & 0xFFFFFFFF
            mem_crc[t] = zlib.crc32(np.sort(master_flat[s:e]).tobytes()) \
                & 0xFFFFFFFF
    return seq_crc, mem_crc, tlen, n_act, n_seg, n_bound


def label_stream(pieces, active_bool):
    """Label ONE camera's stream with the active mask; returns per-camera
    metrics + per-tile hashes (q-dependent, state comes from pieces)."""
    if pieces.get("empty"):
        nt = pieces["n_tiles"]
        return {"empty": True, "n_tiles": nt, "n_isects": 0,
                "n_visible": 0, "n_active_isect": 0, "n_frozen_seg": 0,
                "n_boundaries": 0,
                "seq_crc": np.zeros(nt, dtype=np.uint64),
                "mem_crc": np.zeros(nt, dtype=np.uint64),
                "tlen": np.zeros(nt, dtype=np.int64)}
    seq_crc, mem_crc, tlen, n_act, n_seg, n_bound = tile_stream_hashes(
        pieces["master_flat"], pieces["tile_of_isect"], pieces["offsets"],
        active_bool, pieces["n_tiles"])
    return {"empty": False, "n_tiles": pieces["n_tiles"],
            "n_isects": pieces["n_isects"],
            "n_visible": pieces["n_visible"],
            "n_active_isect": n_act, "n_frozen_seg": n_seg,
            "n_boundaries": n_bound, "seq_crc": seq_crc,
            "mem_crc": mem_crc, "tlen": tlen}


def label_cluster(pieces_list, active_bool):
    """Aggregate labeled stream metrics over a camera cluster."""
    per_cam = []
    tot_act = tot_seg = tot_isect = tot_bound = 0
    for pc in pieces_list:
        d = label_stream(pc, active_bool)
        d = dict(d)
        d["camera"] = pc["camera"]
        per_cam.append(d)
        if not d["empty"]:
            tot_act += d["n_active_isect"]
            tot_seg += d["n_frozen_seg"]
            tot_isect += d["n_isects"]
            tot_bound += d["n_boundaries"]
    return {
        "per_camera": [{k: (int(v) if isinstance(v, (int, np.integer))
                            else v)
                        for k, v in d.items()
                        if k not in ("seq_crc", "mem_crc", "tlen")}
                       for d in per_cam],
        "totals": {
            "n_isects": tot_isect,
            "n_active_isect": tot_act,
            "n_frozen_segments": tot_seg,
            "active_fraction": (tot_act / tot_isect) if tot_isect else 0.0,
            "n_boundaries": tot_bound,
            "n_compiled": tot_act + tot_seg,
            "R_stream": ((tot_act + tot_seg) / tot_isect)
            if tot_isect else 0.0,
        },
        "_cams": per_cam,
    }


def collect_pieces(ctx, cam_ids):
    """Exact C0 stream pieces for a camera list at the CURRENT state."""
    out = []
    for ci in cam_ids:
        cam, _ = ctx.cam_gt(ci)
        pc = ctx.stream_pieces(cam)
        pc["camera"] = int(ci)
        out.append(pc)
    return out


def rebuild_comparison(pre, post):
    """Tile-level pre/post comparison for R_rebuild.

    A tile counts as rebuilt iff its depth-sorted (global_id, active_flag)
    sequence differs (length or crc). Classification: membership change
    (sorted-id crc differs) vs order-only change.
    """
    n_rebuild = n_mem = n_order = n_tiles = 0
    per_cam = []
    for a, b in zip(pre["_cams"], post["_cams"]):
        n_tiles += a["n_tiles"]
        if a.get("empty") and b.get("empty"):
            per_cam.append({"camera": a["camera"], "rebuilt": 0})
            continue
        same_len = a["tlen"] == b["tlen"]
        same_seq = same_len & (a["seq_crc"] == b["seq_crc"])
        rebuilt = ~same_seq
        same_mem = a["mem_crc"] == b["mem_crc"]
        mem_changed = rebuilt & (~same_mem)
        order_changed = rebuilt & same_mem & same_len
        n_rebuild += int(rebuilt.sum())
        n_mem += int(mem_changed.sum())
        n_order += int(order_changed.sum())
        per_cam.append({"camera": a["camera"],
                        "rebuilt": int(rebuilt.sum())})
    return {"n_tiles": n_tiles, "n_tiles_rebuild": n_rebuild,
            "n_tiles_membership_change": n_mem,
            "n_tiles_order_only_change": n_order,
            "R_rebuild": (n_rebuild / n_tiles) if n_tiles else 0.0,
            "per_camera": per_cam}


# ---------------------------------------------------------------- Adam
def masked_adam_step(ctx, active_idx, t, name_to_group_lr):
    """Row-masked Adam step with EXACT torch.optim.Adam semantics on the
    active rows; frozen rows are never read-written => bitwise unchanged.

    Mirrors torch.optim.Adam (non-amsgrad): 
      m.lerp_(g, 1-b1); v.mul_(b2).addcmul_(g,g,value=1-b2)
      denom = (v.sqrt()/sqrt(bc2)).add_(eps); p.addcdiv_(m, denom, -lr/bc1)
    """
    for n in PARAM_NAMES:
        p = ctx._param(n)
        g = p.grad
        assert g is not None, f"no grad for {n}"
        st = ctx.model.optimizer.state[p]
        m, v = st["exp_avg"], st["exp_avg_sq"]
        lr = float(name_to_group_lr[n])
        b1, b2 = 0.9, 0.999
        eps = 1e-15
        ga = g[active_idx]
        ma = m[active_idx]
        va = v[active_idx]
        ma.lerp_(ga, 1 - b1)
        va.mul_(b2).addcmul_(ga, ga, value=1 - b2)
        bc1 = 1 - b1 ** t
        bc2 = 1 - b2 ** t
        denom = (va.sqrt() / math.sqrt(bc2)).add_(eps)
        pa = p.data[active_idx]
        pa.addcdiv_(ma, denom, value=-lr / bc1)
        p.data[active_idx] = pa
        m[active_idx] = ma
        v[active_idx] = va


class FreezeGuard:
    """Per-step bitwise verifier: frozen rows of p/exp_avg/exp_avg_sq must
    stay bit-identical through every step of the arm."""

    def __init__(self, ctx, frozen_idx):
        self.refs = {}
        for n in PARAM_NAMES:
            p = ctx._param(n)
            st = ctx.model.optimizer.state[p]
            self.refs[n] = (p.data[frozen_idx].clone(),
                            st["exp_avg"][frozen_idx].clone(),
                            st["exp_avg_sq"][frozen_idx].clone())
        self.frozen_idx = frozen_idx
        self.checks = 0

    def verify(self, ctx):
        for n in PARAM_NAMES:
            p = ctx._param(n)
            st = ctx.model.optimizer.state[p]
            pr, mr, vr = self.refs[n]
            assert torch.equal(p.data[self.frozen_idx], pr), \
                f"BITWISE FREEZE VIOLATION (param {n})"
            assert torch.equal(st["exp_avg"][self.frozen_idx], mr), \
                f"BITWISE FREEZE VIOLATION (exp_avg {n})"
            assert torch.equal(st["exp_avg_sq"][self.frozen_idx], vr), \
                f"BITWISE FREEZE VIOLATION (exp_avg_sq {n})"
        self.checks += 1


def rows_sha256(ctx, idx_tensor, which):
    """sha256 over the gathered rows of all 5 params (or their Adam state).
    which in {'param','exp_avg','exp_avg_sq'}"""
    h_parts = []
    for n in PARAM_NAMES:
        p = ctx._param(n)
        if which == "param":
            t = p.data[idx_tensor]
        else:
            t = ctx.model.optimizer.state[p][which][idx_tensor]
        h_parts.append(t.detach().cpu().numpy().tobytes())
    return sha256_bytes(b"".join(h_parts))


def params_finite(ctx):
    for n in PARAM_NAMES:
        if not bool(torch.isfinite(ctx._param(n).data).all()):
            return False
    return True


# ---------------------------------------------------------------- arm runner
def run_arm(ctx, cam_schedule, H, freeze=None, log=print,
            freeze_check_every=1):
    """Run H training steps from the CURRENT state (caller restores initial
    first). freeze=None -> dense torch.optim.Adam (D0/D1 arms);
    freeze=(active_idx, frozen_idx) -> masked Adam (F arms) with per-step
    bitwise guard. Iterations continue at ctx.it0 + 1 .. ctx.it0 + H.
    Returns dict of timings (ms) and freeze check count."""
    t0, t1, t2, t3, t4 = [], [], [], [], []
    guard = FreezeGuard(ctx, freeze[1]) if freeze is not None else None
    for k in range(H):
        it = ctx.it0 + 1 + k
        torch.cuda.synchronize()
        ev = [torch.cuda.Event(True) for _ in range(5)]
        ev[0].record()
        ctx.model.update_learning_rate(it)
        ci = int(cam_schedule[k % len(cam_schedule)])
        cam, gt = ctx.cam_gt(ci)
        image = ctx.render_train(cam)
        torch.cuda.synchronize()
        ev[1].record()
        loss = ctx.fused_loss(image, gt)
        torch.cuda.synchronize()
        ev[2].record()
        loss.backward()
        torch.cuda.synchronize()
        ev[3].record()
        if freeze is None:
            ctx.model.optimizer.step()
            ctx.model.optimizer.zero_grad(set_to_none=True)
        else:
            active_idx, _ = freeze
            lrs = {g["name"]: g["lr"]
                   for g in ctx.model.optimizer.param_groups}
            masked_adam_step(ctx, active_idx, it, lrs)
            ctx.model.optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        ev[4].record()
        if freeze is not None and ((k + 1) % freeze_check_every == 0
                                   or k == H - 1):
            guard.verify(ctx)
        t0.append(ev[0].elapsed_time(ev[1]))
        t1.append(ev[1].elapsed_time(ev[2]))
        t2.append(ev[2].elapsed_time(ev[3]))
        t3.append(ev[3].elapsed_time(ev[4]))
        t4.append(ev[0].elapsed_time(ev[4]))
    return {"fwd_ms_med": float(np.median(t0)),
            "loss_ms_med": float(np.median(t1)),
            "bwd_ms_med": float(np.median(t2)),
            "opt_ms_med": float(np.median(t3)),
            "e2e_ms_med": float(np.median(t4)),
            "freeze_checks": 0 if guard is None else guard.checks}


def jwrite(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"[p14] wrote {path}", flush=True)


# module-level regression guard: any import of p14_common verifies the
# stream-labeling semantics on a synthetic stream with known answer
_selftest_tile_stream()
