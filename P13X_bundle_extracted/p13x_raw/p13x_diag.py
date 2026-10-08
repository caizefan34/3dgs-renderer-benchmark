#!/usr/bin/env python3
"""P13-X side-channel diagnostics module (X1..X5 data collection).

DESIGN CONTRACT (P13X_DSH_TASK.md implementation constraint):
  * Never alters camera sampling, optimizer values, densification decisions,
    training RNG state, renderer configuration, loss semantics, or the P12
    phase-timing structure. All work runs AFTER the per-iteration phase
    events + synchronize, i.e. outside every timed region.
  * Exact renderer data only: X1/X2/X3 use the SAME frozen CUDA operators
    the C0 forward uses, re-run under torch.no_grad() on the current model
    state (F9 gatherless producer path when active, otherwise the baseline
    projection path -- mirroring _native_forward_capture branch-for-branch).
    No Python geometric approximation is used anywhere.
  * RNG transparency: torch CPU+CUDA RNG states are saved/restored around
    every diagnostic block (mirrors the P12 fused-loss probe pattern).
  * The training scene tracker (_HIGS_DYNAMIC_SCENE) is only touched by the
    X5 differentiable probe, which calls _higs_dynamic_forward on the LIVE
    model tensors exactly as the trainer does. Post-15K the Gaussian count
    is constant and no mark_dirty occurs; the handle version counter is
    monotonic bookkeeping only. Verified by the P13-X contamination
    preflight (bitwise trajectory comparison).
"""
import json
import math
import os
import time
import zlib

import numpy as np
import torch
import torch.nn.functional as F

DCT_N = 16
EPS_LIST = [0.0, 1e-6, 1e-5, 1e-4]
X4_K_LIST = [4, 8, 16, 32, 64, 128, 256]
X5_K_LIST = [4, 8, 16, 32, 64, 128, 256]
X4_REL_L2_THRESHOLDS = [0.01, 0.05, 0.10]     # preregistered before results
X4_COSINE_THRESHOLDS = [0.99, 0.995, 0.999]   # preregistered before results
# attribute matrix columns (per visible Gaussian, renderer inputs):
# 0:2 means2d | 2:5 conic | 5:6 opacity | 6:9 color_eval | 9:10 depth | 10:12 radii
ATTR_GROUPS = [("mean2d", slice(0, 2)), ("conic", slice(2, 5)),
               ("opacity", slice(5, 6)), ("color", slice(6, 9)),
               ("depth", slice(9, 10)), ("radius", slice(10, 12))]


# ---------------------------------------------------------------- DCT basis
def _dct2_basis(n=DCT_N):
    """Orthonormal DCT-II basis D (D @ D.T = I), fixed and data-independent."""
    k = torch.arange(n, dtype=torch.float64).unsqueeze(1)
    x = torch.arange(n, dtype=torch.float64).unsqueeze(0)
    D = torch.cos(math.pi * (2.0 * x + 1.0) * k / (2.0 * n))
    D[0, :] *= 1.0 / math.sqrt(2.0)
    return (D * math.sqrt(2.0 / n)).float()


def _zigzag_pairs(n=DCT_N):
    """Standard JPEG zig-zag traversal of the n x n coefficient grid."""
    pairs = []
    for s in range(2 * n - 1):
        if s % 2 == 0:
            rng = range(s, -1, -1) if s < n else range(n - 1, s - n, -1)
            for i in rng:
                j = s - i
                if 0 <= i < n and 0 <= j < n:
                    pairs.append((i, j))
        else:
            lo = 0 if s < n else s - n + 1
            hi = s + 1 if s < n else n
            for i in range(lo, hi):
                j = s - i
                if 0 <= i < n and 0 <= j < n:
                    pairs.append((i, j))
    assert len(pairs) == n * n
    return pairs


_ZIGZAG = _zigzag_pairs(DCT_N)
_DCT = None
_DCT_MASKS = None


def _get_dct(device):
    global _DCT, _DCT_MASKS
    if _DCT is None or _DCT.device != device:
        _DCT = _dct2_basis().to(device)
        _DCT_MASKS = {}
        for K in X4_K_LIST:
            m = torch.zeros(DCT_N, DCT_N, dtype=torch.float32, device=device)
            for (ky, kx) in _ZIGZAG[:K]:
                m[ky, kx] = 1.0
            _DCT_MASKS[K] = m
    return _DCT


def _tiles_of(g):
    """[H,W,3] -> [T, C, 16, 16] full in-image tiles + (TH, TW)."""
    H, W, C = g.shape
    TH, TW = H // DCT_N, W // DCT_N
    t = (g[: TH * DCT_N, : TW * DCT_N, :].permute(2, 0, 1)
         .reshape(C, TH, DCT_N, TW, DCT_N)
         .permute(1, 3, 0, 2, 4).reshape(-1, C, DCT_N, DCT_N))
    return t, TH, TW


def _dct_reconstruct(adj, K):
    """adj [H,W,3] -> keep-K low-frequency zig-zag DCT reconstruction.

    Zero-pads partial edge tiles to full 16x16 and crops back, so K=256
    (full orthonormal basis) reproduces the input exactly (internal
    validation); lower K isolates the pure compression effect.
    """
    H, W, C = adj.shape
    TH, TW = -(-H // DCT_N), -(-W // DCT_N)
    Hp, Wp = TH * DCT_N, TW * DCT_N
    padded = torch.zeros(Hp, Wp, C, dtype=adj.dtype, device=adj.device)
    padded[:H, :W] = adj
    D = _get_dct(adj.device)
    tiles = (padded.permute(2, 0, 1)
             .reshape(C, TH, DCT_N, TW, DCT_N)
             .permute(1, 3, 0, 2, 4).reshape(-1, C, DCT_N, DCT_N))
    if tiles.shape[0] == 0:
        return torch.zeros_like(adj)
    coeffs = torch.matmul(torch.matmul(D, tiles), D.transpose(0, 1))
    masked = coeffs * _DCT_MASKS[K].unsqueeze(0).unsqueeze(0)
    recon = torch.matmul(torch.matmul(D.transpose(0, 1), masked), D)
    out = (recon.reshape(TH, TW, C, DCT_N, DCT_N)
           .permute(2, 0, 3, 1, 4).reshape(C, Hp, Wp).permute(1, 2, 0))
    return out[:H, :W].contiguous()


def _crc32(arr):
    return zlib.crc32(arr.astype(np.int64).tobytes()) & 0xFFFFFFFF


# ---------------------------------------------------------------- main class
class P13XDiag:
    def __init__(self, outdir, scene, n_cameras, iterations, seed, arm,
                 diag_start=15000, diag_end=30000, diag_stride=250,
                 n_anchors=8, lags=(250, 500, 1000, 2500),
                 x4_samples=256, x5_enabled=True, x5_anchors=8):
        self.outdir = outdir
        self.scene = scene
        self.n_cameras = int(n_cameras)
        self.iterations = int(iterations)
        self.seed = seed
        self.arm = arm
        self.diag_start, self.diag_end = int(diag_start), int(diag_end)
        self.diag_stride = int(diag_stride)
        self.n_anchors = int(n_anchors)
        self.lags = [int(x) for x in lags]
        self.x4_samples = int(x4_samples)
        self.x5_enabled = bool(x5_enabled)
        self.x5_anchors = int(x5_anchors)

        end_eff = min(self.diag_end, self.iterations)
        self.diag_iters = sorted(
            set(range(self.diag_start, end_eff + 1, self.diag_stride)))
        span = max(1, self.diag_end - self.diag_start)
        self.x4_iters = sorted(set(
            self.diag_start + int(round((k + 1) * span / self.x4_samples))
            for k in range(self.x4_samples)))
        self.x4_iters = [i for i in self.x4_iters if i <= self.iterations]
        self.x4_set = set(self.x4_iters)
        self._diag_set = set(self.diag_iters)

        self.records = {"x1": [], "x2": [], "x3": [], "x4": [], "x5": [],
                        "snapshots": []}
        self._snaps = {}
        self._model = None
        self._dataset = None
        self._loss_fn = None
        self._eps2d = None
        self._anchor_data = {}
        self.anchors = []
        self.diag_wall_s = 0.0
        self.enabled_iters = self._diag_set | self.x4_set
        self._f9_active = None

    # ------------------------------------------------------------ bind
    def bind(self, model, dataset, loss_fn, eps2d):
        self._model = model
        self._dataset = dataset
        self._loss_fn = loss_fn
        self._eps2d = float(eps2d)
        raw = np.round(np.linspace(0, self.n_cameras - 1, self.n_anchors)
                       ).astype(int)
        self.anchors = sorted(set(int(i) for i in raw))
        for a in self.anchors:
            self._anchor_data[a] = dataset.get_item(a)
            self._snaps[a] = {}
        if self._loss_fn is None:
            self.x5_enabled = False

    @property
    def f9_active(self):
        """F9 gatherless producer active exactly when the training forward
        would enable it (same conditions as _HigsAutogradFunction.forward)."""
        if self._f9_active is None:
            m = self._model
            self._f9_active = bool(
                m.active_sh_degree == 3
                and m._shs.shape[1:] == (16, 3)
                and os.getenv("HIGS_DISABLE_F9", "0") != "1")
        return self._f9_active

    def write_protocol(self):
        os.makedirs(os.path.join(self.outdir, "raw"), exist_ok=True)
        proto = {
            "scene": self.scene, "arm": self.arm, "seed": self.seed,
            "iterations": self.iterations,
            "diag_window": [self.diag_start, self.diag_end],
            "diag_stride": self.diag_stride,
            "diag_iterations": self.diag_iters,
            "anchor_rule": "unique(round(linspace(0, n_cameras-1, n_anchors)))"
                           " over the true sorted COLMAP training camera list",
            "anchors": self.anchors,
            "lags_requested": self.lags,
            "lags_measurable": [L for L in self.lags
                                if any(i - L in self._diag_set
                                       for i in self.diag_iters)],
            "x4": {"samples": self.x4_samples, "iterations": self.x4_iters,
                   "rule": "diag_start + round((k+1)*span/samples), k=0..samples-1",
                   "tile": 16, "basis": "fixed orthonormal DCT-II per channel",
                   "selection": "fixed low-frequency zig-zag",
                   "K": X4_K_LIST,
                   "rel_l2_thresholds": X4_REL_L2_THRESHOLDS,
                   "cosine_thresholds": X4_COSINE_THRESHOLDS},
            "x5": {"enabled": self.x5_enabled,
                   "anchors": self.anchors[: self.x5_anchors], "K": X5_K_LIST},
            "x3": {"eps_list": EPS_LIST,
                   "attr_groups": [g[0] for g in ATTR_GROUPS]},
            "data_source": ("exact frozen C0 CUDA ops re-run under no_grad "
                            "(culling -> F9 producer or fully_fused_projection "
                            "-> torch.ops.gsplat.intersect_tile -> "
                            "isect_offset_encode -> rasterize_to_pixels_3dgs)"),
            "rng_guard": "torch CPU+CUDA RNG saved/restored around every "
                         "diagnostic block",
        }
        with open(os.path.join(self.outdir, "raw", "P13X_DIAG_PROTOCOL.json"),
                  "w") as f:
            json.dump(proto, f, indent=2)

    # ------------------------------------------------------------ helpers
    def _cam_tensors(self, cam):
        vm = cam.viewmatrix.unsqueeze(0).unsqueeze(0).contiguous()
        K = cam.K.unsqueeze(0).unsqueeze(0).contiguous()
        return vm, K

    def _render_inputs_detached(self):
        """Activated renderer inputs, mirroring render_with_meta exactly."""
        m = self._model
        n_active = (m.active_sh_degree + 1) ** 2
        return (m._xyz.detach(),
                F.normalize(m._rotation.detach(), dim=-1),
                torch.exp(m._scaling.detach()),
                torch.sigmoid(m._opacity.detach()).squeeze(-1),
                m._shs[:, :n_active, :].detach().contiguous())

    def _activated_from_raw(self, xyz, rot, scaling, opacity, shs):
        """Apply the model's getter activations to raw updated params."""
        m = self._model
        n_active = (m.active_sh_degree + 1) ** 2
        return (xyz,
                F.normalize(rot, dim=-1),
                torch.exp(scaling),
                torch.sigmoid(opacity).squeeze(-1),
                shs[:, :n_active, :].contiguous())

    def _forward_ops(self, means, quats, scales, opacities, colors, cam,
                     sh_degree):
        """Exact frozen C0 forward ops on arbitrary tensors (no tracker).
        Returns (frame_raw [H,W,3], vis_ids, radii, means2d, depths, conics,
                 op_bc, colors_eval, isect_ids, flatten_ids, tw, th)."""
        from gsplat.cuda._wrapper import (fully_fused_projection,
                                          isect_offset_encode,
                                          _make_lazy_cuda_func)
        from gsplat.experimental.render.kernels import _backend as _ib
        backend = _ib._C
        width, height = int(cam.image_width), int(cam.image_height)
        tw, th = math.ceil(width / 16), math.ceil(height / 16)
        vm, K = self._cam_tensors(cam)
        with torch.no_grad():
            from gsplat.experimental.render.functional.gaussian_inference \
                import _cull_gaussians_batched
            vis_ids, _, _ = _cull_gaussians_batched(
                means, quats, scales, vm, K, width, height,
                eps2d=self._eps2d, near_plane=0.01, far_plane=1e10,
                radius_clip=0.0, camera_model="pinhole")
            C = 1
            if vis_ids.numel() == 0:
                return (torch.zeros(height, width, 3, device=means.device),
                        vis_ids, None, None, None, None, None, None,
                        None, None, tw, th)
            f9 = (sh_degree == 3 and colors.dim() == 3
                  and colors.shape[1] == 16
                  and os.getenv("HIGS_DISABLE_F9", "0") != "1")
            if f9:
                cam_pos = backend.higs_camera_positions_from_viewmats(
                    vm[0].contiguous())
                radii, means2d, depths, conics, op_bc, colors_eval = (
                    backend.higs_gatherless_projected_producer(
                        vis_ids.contiguous(), means.contiguous(),
                        quats.contiguous(), scales.contiguous(),
                        opacities.contiguous(), colors.contiguous(),
                        vm[0].contiguous(), K[0].contiguous(), cam_pos,
                        width, height, self._eps2d, 0.01, 1e10, 0.0))
                N = vis_ids.numel()
                radii = radii.reshape(1, C, N, 2)
                means2d = means2d.reshape(1, C, N, 2)
                depths = depths.reshape(1, C, N)
                conics = conics.reshape(1, C, N, 3)
                op_bc = op_bc.reshape(1, C, N)
                colors_eval = colors_eval.reshape(1, C, N, 3)
            else:
                from gsplat.rendering import _maybe_evaluate_sh
                v_means = means[vis_ids]
                v_quats = quats[vis_ids]
                v_scales = scales[vis_ids]
                v_opacities = opacities[vis_ids]
                v_colors = colors[vis_ids]
                N = vis_ids.numel()
                radii, means2d, depths, conics, _ = fully_fused_projection(
                    means=v_means.unsqueeze(0).contiguous(), covars=None,
                    quats=v_quats.unsqueeze(0).contiguous(),
                    scales=v_scales.unsqueeze(0).contiguous(),
                    viewmats=vm.contiguous(), Ks=K.contiguous(),
                    width=width, height=height, eps2d=self._eps2d,
                    near_plane=0.01, far_plane=1e10, radius_clip=0.0,
                    packed=False, calc_compensations=False,
                    camera_model="pinhole")
                op_bc = torch.broadcast_to(
                    v_opacities.unsqueeze(0).unsqueeze(0), (1, C, N)
                ).contiguous()
                colors_eval = _maybe_evaluate_sh(
                    sh_degree, v_colors,
                    v_means.unsqueeze(0).contiguous(), radii, vm, (1,),
                    C, N, True).contiguous()
            _, isect_ids, flatten_ids = torch.ops.gsplat.intersect_tile(
                means2d, radii, depths, conics, op_bc, None, None, C,
                16, tw, th, True, False)
            offsets = isect_offset_encode(isect_ids, C, tw, th).reshape(
                (1, C, th, tw))
            render_colors, _, _, _ = _make_lazy_cuda_func(
                "rasterize_to_pixels_3dgs")(
                means2d.contiguous(), conics.contiguous(),
                colors_eval.contiguous(), op_bc.contiguous(),
                None, None, width, height, 16, offsets.contiguous(),
                flatten_ids.contiguous(), False, False)
            frame = render_colors.squeeze(0).squeeze(0)
        return (frame, vis_ids, radii, means2d, depths, conics, op_bc,
                colors_eval, isect_ids, flatten_ids, tw, th)

    def _diag_render(self, means, quats, scales, opacities, colors, cam,
                     sh_degree):
        """Reference render for probes: raw frame [H,W,3] + visible ids."""
        frame, vis_ids, *_ = self._forward_ops(
            means, quats, scales, opacities, colors, cam, sh_degree)
        return frame, vis_ids

    def _collect_anchor(self, a):
        cam, _gt = self._anchor_data[a]
        m = self._model
        means, quats, scales, opacities, colors = self._render_inputs_detached()
        (frame, vis_ids, radii, means2d, depths, conics, op_bc, colors_eval,
         isect_ids, flatten_ids, tw, th) = self._forward_ops(
            means, quats, scales, opacities, colors, cam,
            m.active_sh_degree)
        width, height = int(cam.image_width), int(cam.image_height)
        if vis_ids.numel() == 0:
            return {"anchor": a, "empty": True, "width": width,
                    "height": height}
        vis = vis_ids.cpu().numpy().astype(np.int64)
        flat = flatten_ids.cpu().numpy().astype(np.int64)
        master_flat = vis[flat]
        iids = isect_ids.cpu().numpy().astype(np.int64)
        n_tiles = th * tw
        tile_bits = max(1, (n_tiles - 1).bit_length())
        tile_of_isect = (iids >> 32) & ((1 << tile_bits) - 1)
        img_id = iids >> (32 + tile_bits)
        assert int(img_id.max()) == 0, "unexpected camera id in isect_ids"
        C, N = 1, vis_ids.numel()
        attrs = torch.cat([
            means2d.reshape(C, N, 2)[0], conics.reshape(C, N, 3)[0],
            op_bc.reshape(C, N, 1)[0], colors_eval.reshape(C, N, 3)[0],
            depths.reshape(C, N, 1)[0], radii.reshape(C, N, 2)[0],
        ], dim=1).float().cpu().numpy().astype(np.float32)
        return {
            "anchor": a, "empty": False, "width": width, "height": height,
            "th": th, "tw": tw, "n_tiles": n_tiles,
            "vis_ids": vis, "master_flat": master_flat,
            "tile_of_isect": tile_of_isect.astype(np.int64),
            "attrs": attrs,
            "frame_raw": frame.detach().float().cpu().numpy(),
            "n_isects": int(master_flat.shape[0]),
            "n_visible": int(vis.shape[0]),
        }

    # ------------------------------------------------------------ X4
    def _x4_stats(self, g):
        H, W, _ = g.shape
        tiles, TH, TW = _tiles_of(g)
        if tiles.shape[0] == 0:
            return {"error": "image smaller than one tile"}
        D = _get_dct(g.device)
        coeffs = torch.matmul(torch.matmul(D, tiles), D.transpose(0, 1))
        orig = tiles.reshape(tiles.shape[0], -1)
        e_orig = orig.norm(dim=1)
        out = {"n_tiles": int(tiles.shape[0])}
        for K in X4_K_LIST:
            masked = coeffs * _DCT_MASKS[K].unsqueeze(0).unsqueeze(0)
            rec = torch.matmul(torch.matmul(D.transpose(0, 1), masked), D)
            rec = rec.reshape(rec.shape[0], -1)
            e_rec = rec.norm(dim=1)
            dot = (orig * rec).sum(dim=1)
            cos = dot / (e_orig * e_rec + 1e-30)
            rel = (rec - orig).norm(dim=1) / (e_orig + 1e-30)
            ret = (e_rec / (e_orig + 1e-30)) ** 2
            out[f"K{K}"] = {
                "energy_retention_mean": float(ret.mean()),
                "energy_retention_median": float(ret.median()),
                "rel_l2_mean": float(rel.mean()),
                "rel_l2_median": float(rel.median()),
                "rel_l2_p90": float(torch.quantile(rel, 0.90)),
                "cosine_mean": float(cos.mean()),
                "cosine_median": float(cos.median()),
                "frac_tiles_rel_l2_le": {
                    str(t): float((rel <= t).float().mean())
                    for t in X4_REL_L2_THRESHOLDS},
                "frac_tiles_cosine_ge": {
                    str(t): float((cos >= t).float().mean())
                    for t in X4_COSINE_THRESHOLDS},
            }
        return out

    # ------------------------------------------------------------ X2 helpers
    @staticmethod
    def _tile_slices(snap):
        tids = snap["tile_of_isect"]
        T = snap["n_tiles"]
        starts = np.searchsorted(tids, np.arange(T), side="left")
        ends = np.searchsorted(tids, np.arange(T), side="right")
        if tids.shape[0]:
            recon = np.repeat(np.arange(T), ends - starts)
            assert np.array_equal(recon, tids), "tile boundary check failed"
        return starts, ends

    @staticmethod
    def _per_tile_hashes(snap):
        starts, ends = P13XDiag._tile_slices(snap)
        mf = snap["master_flat"]
        T = snap["n_tiles"]
        mem_h = np.empty(T, dtype=np.int64)
        ord_h = np.empty(T, dtype=np.int64)
        counts = np.empty(T, dtype=np.int64)
        for t in range(T):
            seg = mf[starts[t]:ends[t]]
            counts[t] = seg.shape[0]
            mem_h[t] = _crc32(np.sort(seg))
            ord_h[t] = _crc32(seg)
        return mem_h, ord_h, counts

    @staticmethod
    def _inversion_rates(snapA, snapB, startsA, endsA, startsB, endsB):
        """Normalized inversion rate among common ids per tile (exact order)."""
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        rates = np.zeros(snapA["n_tiles"], dtype=np.float64)
        mfA, mfB = snapA["master_flat"], snapB["master_flat"]
        for t in range(snapA["n_tiles"]):
            a = mfA[startsA[t]:endsA[t]]
            b = mfB[startsB[t]:endsB[t]]
            common = np.intersect1d(a, b)
            n = int(common.shape[0])
            if n < 2:
                continue
            # position of each common id within A's / B's in-tile sequence
            jA = np.argsort(a)[np.searchsorted(np.sort(a), common)]
            jB = np.argsort(b)[np.searchsorted(np.sort(b), common)]
            W = torch.as_tensor(jB[np.argsort(jA)], device=dev)
            if n <= 8192:
                # inversions: pairs (i, j), i < j in A-order, W[i] > W[j]
                cmp = W.unsqueeze(1) > W.unsqueeze(0)
                inv = float(torch.triu(cmp, diagonal=1).sum())
            else:
                inv = 0.0
                CH = 8192
                for s0 in range(0, n, CH):
                    e0 = min(n, s0 + CH)
                    # cmp[i, c] = W[i] > W[c + s0]   (shape [n, chunk])
                    cmp = W.unsqueeze(1) > W[s0:e0].unsqueeze(0)
                    ii = torch.arange(n, device=dev).unsqueeze(1)
                    jj = torch.arange(s0, e0, device=dev).unsqueeze(0)
                    inv += float((cmp & (ii < jj)).sum())
            rates[t] = inv / (n * (n - 1) / 2.0)
        return rates

    # ------------------------------------------------------------ pair metrics
    def _pair_metrics(self, a, itA, itB, lag):
        sA, sB = self._snaps[a][itA], self._snaps[a][itB]
        rec = {"anchor": a, "iter_a": itA, "iter_b": itB, "lag": lag}
        # ---- X1
        A, B = sA["vis_ids"], sB["vis_ids"]
        inter = np.intersect1d(A, B)
        union = A.shape[0] + B.shape[0] - inter.shape[0]
        x1 = {
            "n_visible_a": int(A.shape[0]), "n_visible_b": int(B.shape[0]),
            "jaccard": float(inter.shape[0] / union) if union else 1.0,
            "retained": float(inter.shape[0] / A.shape[0]) if A.shape[0] else 1.0,
            "entered": float((B.shape[0] - inter.shape[0]) / B.shape[0])
            if B.shape[0] else 0.0,
        }
        # ---- X2
        Nmax = int(max(A.max(), B.max())) + 1
        keysA = sA["master_flat"] + sA["tile_of_isect"] * Nmax
        keysB = sB["master_flat"] + sB["tile_of_isect"] * Nmax
        uA, uB = np.unique(keysA), np.unique(keysB)
        ui = int(np.intersect1d(uA, uB).shape[0])
        uu = int(uA.shape[0] + uB.shape[0] - ui)
        mem_hA, ord_hA, cntA = self._per_tile_hashes(sA)
        mem_hB, ord_hB, cntB = self._per_tile_hashes(sB)
        startsA, endsA = self._tile_slices(sA)
        startsB, endsB = self._tile_slices(sB)
        active = (cntA > 0) | (cntB > 0)
        n_active = int(active.sum())
        inv = self._inversion_rates(sA, sB, startsA, endsA, startsB, endsB)
        x2 = {
            "n_isects_a": int(sA["n_isects"]), "n_isects_b": int(sB["n_isects"]),
            "n_unique_memberships_a": int(uA.shape[0]),
            "n_unique_memberships_b": int(uB.shape[0]),
            "membership_jaccard": float(ui / uu) if uu else 1.0,
            "n_active_tiles": n_active,
            "frac_tiles_membership_hash_unchanged":
                float((mem_hA[active] == mem_hB[active]).mean())
                if n_active else 1.0,
            "frac_tiles_order_hash_unchanged":
                float((ord_hA[active] == ord_hB[active]).mean())
                if n_active else 1.0,
            "inversion_rate_median_over_tiles":
                float(np.median(inv[active])) if n_active else 0.0,
            "inversion_rate_mean_over_tiles":
                float(inv[active].mean()) if n_active else 0.0,
            "inversion_rate_p90_over_tiles":
                float(np.quantile(inv[active], 0.90)) if n_active else 0.0,
            "members_per_tile_median_b":
                float(np.median(cntB[cntB > 0])) if (cntB > 0).any() else 0.0,
            "members_per_tile_max_b": int(cntB.max()) if cntB.size else 0,
        }
        # ---- X3
        common_ids, iA_idx, iB_idx = np.intersect1d(A, B, return_indices=True)
        atA, atB = sA["attrs"][iA_idx], sB["attrs"][iB_idx]
        group_rel = {}
        for (gname, sl) in ATTR_GROUPS:
            ga, gb = atA[:, sl], atB[:, sl]
            d = np.maximum(np.linalg.norm(ga, axis=1), 1e-30)
            group_rel[gname] = np.linalg.norm(gb - ga, axis=1) / d
        order_changed = ord_hA != ord_hB   # covers membership changes too
        mfB = sB["master_flat"]
        fA, fB = sA["frame_raw"], sB["frame_raw"]
        H, W = fB.shape[:2]
        TH, TW = sB["th"], sB["tw"]
        Hp, Wp = TH * 16, TW * 16
        fAp = np.zeros((Hp, Wp, 3), np.float32); fAp[:H, :W] = fA
        fBp = np.zeros((Hp, Wp, 3), np.float32); fBp[:H, :W] = fB
        tilesA = fAp.reshape(TH, 16, TW, 16, 3).transpose(0, 2, 1, 3, 4)
        tilesB = fBp.reshape(TH, 16, TW, 16, 3).transpose(0, 2, 1, 3, 4)
        denom_img = float(np.linalg.norm(fB))
        x3 = {"n_common_visible": int(common_ids.shape[0]),
              "attr_group_rel_l2_median": {
                  g: float(np.median(group_rel[g])) for g in group_rel}}
        for eps in EPS_LIST:
            ch = np.zeros(common_ids.shape[0], dtype=bool)
            for gname in group_rel:
                ch |= group_rel[gname] > eps
            lut = np.zeros(Nmax, dtype=bool)
            lut[common_ids] = ch
            any_changed_tile = np.zeros(sB["n_tiles"], dtype=bool)
            if mfB.size:
                pic = lut[mfB].astype(np.int32)
                ne = endsB > startsB
                if ne.any() and pic.size:
                    any_changed_tile[ne] = np.maximum.reduceat(pic, startsB[ne]) > 0
            dirty = order_changed | any_changed_tile
            dirty_active = dirty & active
            cnts = endsB - startsB
            frac_dirty_tiles = (float(dirty_active.sum() / n_active)
                                if n_active else 0.0)
            frac_dirty_isects = (float(cnts[dirty].sum() / mfB.size)
                                 if mfB.size else 0.0)
            clean2d = (~dirty).reshape(TH, TW)
            hyb = tilesB.copy()
            hyb[clean2d] = tilesA[clean2d]
            hyb = hyb.transpose(0, 2, 1, 3, 4).reshape(Hp, Wp, 3)[:H, :W]
            replay_rel = float(np.linalg.norm(hyb - fB) / max(denom_img, 1e-30))
            hybC, fBC = np.clip(hyb, 0, 1), np.clip(fB, 0, 1)
            replay_rel_c = float(np.linalg.norm(hybC - fBC)
                                 / max(float(np.linalg.norm(fBC)), 1e-30))
            x3[f"eps_{eps:g}"] = {
                "dirty_tile_fraction": frac_dirty_tiles,
                "dirty_intersection_fraction": frac_dirty_isects,
                "replay_rel_l2_raw": replay_rel,
                "replay_rel_l2_clamped": replay_rel_c,
                "n_dirty_tiles": int(dirty_active.sum()),
            }
        return {"x1": x1, "x2": x2, "x3": x3, "meta": rec}

    # ------------------------------------------------------------ X5
    def _x5_probe(self, a):
        from gsplat.experimental.render.functional.gaussian_inference import (
            _higs_dynamic_forward)
        cam, gt = self._anchor_data[a]
        model = self._model
        width, height = int(cam.image_width), int(cam.image_height)
        vm, K = self._cam_tensors(cam)
        # differentiable probe forward on LIVE params via the model getters
        # (the exact training render inputs, incl. activation chains)
        means = model.get_xyz
        quats = model.get_rotation
        scales = model.get_scaling
        opacities = model.get_opacity
        colors = model.get_features
        raw = [model._xyz, model._rotation, model._scaling, model._opacity,
               model._shs]
        names = ["means", "quats", "scales", "opacity", "sh"]
        out = _higs_dynamic_forward(
            means, quats, scales, opacities, colors,
            viewmats=vm, Ks=K, width=width, height=height,
            sh_degree=model.active_sh_degree,
            backward_mode="higs_native", enable_culling=True,
            camera_model="pinhole", render_mode="RGB", eps2d=self._eps2d)
        frame = out["frame"].reshape(height, width, 3)
        image = frame.clamp(0, 1)
        loss = self._loss_fn(image, gt)
        dLd = torch.autograd.grad(loss, image, retain_graph=True)[0]
        g_exact = torch.autograd.grad(image, raw, grad_outputs=dLd,
                                      retain_graph=True, allow_unused=True)
        g_exact = [g if g is not None else torch.zeros_like(p)
                   for g, p in zip(g_exact, raw)]
        gK = {}
        for Kk in X5_K_LIST:
            adj = _dct_reconstruct(dLd, Kk)
            gk = torch.autograd.grad(image, raw, grad_outputs=adj,
                                     retain_graph=(Kk != X5_K_LIST[-1]),
                                     allow_unused=True)
            gK[Kk] = [g if g is not None else torch.zeros_like(p)
                      for g, p in zip(gk, raw)]
        probe_norm0 = float(dLd.norm())
        probe_loss = float(loss.item())
        # internal validation: full-basis reconstruction must reproduce dLd
        _adj256 = _dct_reconstruct(dLd, 256)
        rec256_rel = float((_adj256 - dLd).norm()
                           / max(probe_norm0, 1e-30))
        del _adj256
        del out, frame, image, loss, dLd
        rec = {"anchor": a, "dLdRGB_norm": probe_norm0,
               "probe_loss": probe_loss,
               "dct256_reconstruction_rel_l2": rec256_rel, "K": {}}
        for Kk in X5_K_LIST:
            row = {}
            for gi, gname in enumerate(names):
                ge, gk = g_exact[gi], gK[Kk][gi]
                ne = float(ge.norm())
                rel = float((gk - ge).norm() / max(ne, 1e-30))
                cos = float((ge * gk).sum()
                            / max(float(ge.norm() * gk.norm()), 1e-30))
                k = max(1, int(ge.numel() * 0.10))
                topv, topi = torch.topk(ge.abs().reshape(-1), k)
                agree = float((torch.sign(gk.reshape(-1)[topi])
                               == torch.sign(ge.reshape(-1)[topi])).float().mean())
                row[gname] = {"rel_l2": rel, "cosine": cos,
                              "top10_sign_agreement": agree,
                              "grad_norm_exact": ne}
            rec["K"][str(Kk)] = row
        # shadow one-step loss delta (cloned optimizer state; no contamination)
        if getattr(model, "optimizer", None) is not None:
            shadow = self._shadow_base(raw)
            losses_next = {}
            variants = [("exact", g_exact)] + [(f"K{k}", gK[k])
                                               for k in X5_K_LIST]
            for vname, grads in variants:
                p_new = self._adam_step(shadow, grads)
                mi, qi, si, oi, ci = self._activated_from_raw(*p_new)
                with torch.no_grad():
                    fnext, _ = self._diag_render(
                        mi, qi, si, oi, ci, cam, model.active_sh_degree)
                    lnext = float(self._loss_fn(fnext.clamp(0, 1), gt).item())
                losses_next[vname] = lnext
            lex = losses_next["exact"]
            rec["shadow_one_step"] = {
                v: {"loss_next": lv,
                    "delta_rel": float(abs(lv - lex) / max(abs(lex), 1e-30))}
                for v, lv in losses_next.items()}
            del shadow
        del gK, g_exact
        return rec

    def _shadow_base(self, raw):
        model = self._model
        base = []
        for p in raw:
            grp = None
            for g in model.optimizer.param_groups:
                if any(pp is p for pp in g["params"]):
                    grp = g
                    break
            st = model.optimizer.state.get(p, {})
            base.append({
                "p0": p.detach().clone(),
                "m0": st["exp_avg"].clone() if "exp_avg" in st else None,
                "v0": st["exp_avg_sq"].clone() if "exp_avg_sq" in st else None,
                "step": (int(st["step"].item()) if "step" in st
                         and torch.is_tensor(st["step"])
                         else int(st.get("step", 0))),
                "lr": float(grp["lr"]) if grp else 0.0,
                "betas": tuple(grp["betas"]) if grp else (0.9, 0.999),
                "eps": float(grp["eps"]) if grp else 1e-15,
                "wd": float(grp.get("weight_decay", 0.0)) if grp else 0.0,
            })
        return base

    @staticmethod
    def _adam_step(shadow, grads):
        out = []
        for b, g in zip(shadow, grads):
            p0, m0, v0 = b["p0"], b["m0"], b["v0"]
            b1, b2, eps, lr, wd = (b["betas"][0], b["betas"][1], b["eps"],
                                   b["lr"], b["wd"])
            t = b["step"] + 1
            gg = g if wd == 0.0 else g + wd * p0
            if m0 is None:
                m0 = torch.zeros_like(p0)
                v0 = torch.zeros_like(p0)
            m = b1 * m0 + (1 - b1) * gg
            v = b2 * v0 + (1 - b2) * gg * gg
            mh = m / (1 - b1 ** t)
            vh = v / (1 - b2 ** t)
            out.append((p0 - lr * mh / (vh.sqrt() + eps)).detach())
        return out

    # ------------------------------------------------------------ entry
    def on_iteration(self, iteration, image):
        if iteration not in self.enabled_iters:
            return
        t0 = time.perf_counter()
        rng_cpu = torch.get_rng_state()
        rng_cuda = torch.cuda.get_rng_state()
        try:
            if iteration in self.x4_set:
                if image is not None and image.grad is not None:
                    self.records["x4"].append(
                        {"iter": iteration,
                         **self._x4_stats(image.grad.detach())})
                else:
                    self.records["x4"].append(
                        {"iter": iteration, "error": "image.grad missing"})
            if iteration in self._diag_set:
                for a in self.anchors:
                    snap = self._collect_anchor(a)
                    self._snaps[a][iteration] = snap
                    self.records["snapshots"].append({
                        "iter": iteration, "anchor": a,
                        "n_visible": snap.get("n_visible", 0),
                        "n_isects": snap.get("n_isects", 0),
                        "empty": bool(snap.get("empty", False))})
                for a in self.anchors:
                    for lag in self.lags:
                        itA = iteration - lag
                        if itA in self._snaps[a]:
                            r = self._pair_metrics(a, itA, iteration, lag)
                            for probe in ("x1", "x2", "x3"):
                                self.records[probe].append(
                                    dict(r[probe], **r["meta"]))
                max_lag = max((L for L in self.lags
                               if any(i - L in self._diag_set
                                      for i in self.diag_iters)), default=0)
                for a in self.anchors:
                    for it in list(self._snaps[a].keys()):
                        if iteration - it > max_lag:
                            del self._snaps[a][it]
                if self.x5_enabled:
                    for a in self.anchors[: self.x5_anchors]:
                        try:
                            self.records["x5"].append(
                                {"iter": iteration, **self._x5_probe(a)})
                        except Exception as e:  # noqa: BLE001
                            self.records["x5"].append(
                                {"iter": iteration, "anchor": a,
                                 "error": repr(e)})
        finally:
            torch.set_rng_state(rng_cpu)
            torch.cuda.set_rng_state(rng_cuda)
        self.diag_wall_s += time.perf_counter() - t0

    # ------------------------------------------------------------ finalize
    def finalize(self):
        os.makedirs(os.path.join(self.outdir, "raw"), exist_ok=True)
        summary = {
            "scene": self.scene, "arm": self.arm,
            "diag_wall_s_excluded_from_phase_timing": self.diag_wall_s,
            "n_x1_pairs": len(self.records["x1"]),
            "n_x2_pairs": len(self.records["x2"]),
            "n_x3_pairs": len(self.records["x3"]),
            "n_x4_samples": len(self.records["x4"]),
            "n_x5_probes": len(self.records["x5"]),
            "n_snapshots": len(self.records["snapshots"]),
            "x5_errors": sum(1 for r in self.records["x5"] if "error" in r),
        }
        with open(os.path.join(self.outdir, "raw", "P13X_RAW_SUMMARY.json"),
                  "w") as f:
            json.dump(summary, f, indent=2)
        for key in ("x1", "x2", "x3", "x4", "x5", "snapshots"):
            with open(os.path.join(self.outdir, "raw",
                                   f"P13X_RAW_{key.upper()}.json"), "w") as f:
                json.dump(self.records[key], f, indent=1)
        print(f"  [P13X] finalize: {json.dumps(summary)}", flush=True)
        return summary
