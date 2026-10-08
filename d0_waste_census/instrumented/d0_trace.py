"""D0 trace instrumentation (metadata-only) for the frozen P12 C0 recipe.

All state here is derived metadata: no training tensor is read-modify-written,
no RNG is consumed, no densification/prune decision is influenced. Definitions
are frozen in D0_INSTRUMENTATION_SPEC.md.

Layout: per-uid arrays indexed by uid (creation order); rows map to uids through
model._gaussian_ids (maintained by the frozen GaussianModel).
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

FEATURE_NAMES = [
    # parent state (10)
    "parent_age", "parent_opacity", "parent_xyz_norm", "depth_last_visible",
    "scale_x", "scale_y", "scale_z", "scale_volume", "radius_last_px",
    "sh_degree_active",
    # densification signals (7)
    "grad_accum_row", "grad_denom_row", "grad_mean_norm", "threshold_margin",
    "max_radii2D_row", "last_absgrad_norm_px", "branch",
    # historical visibility (6)
    "vis_count_32", "vis_count_128", "vis_frac_32", "vis_frac_128",
    "n_distinct_cameras", "view_dir_spread_32",
    # historical reconstruction (7)
    "res_mean_current_tile", "res_p90_current_tile", "res_roll_mean_32",
    "res_roll_var_32", "grad_dir_stability_32", "footprint_tiles_last_obs",
    "footprint_tiles_last32_mean",
]
N_FEATURES = len(FEATURE_NAMES)

HIST_RING = 32          # residual / direction ring length (observations)
VIS_RING = 128          # visibility stamp ring length (iterations)
STAMP_INIT = -(1 << 20)  # sentinel: age = iter - stamp >> 128 => never visible

TRAJ_FIELDS = [
    "iter", "f9", "N_t", "n_visible", "I_t", "I_t_sumchecked", "tiles_sample_uid_sum",
    "births_clone", "births_split", "prunes_total", "prunes_opacity",
    "prunes_screen", "prunes_world", "split_parent_removed", "opacity_reset",
    "loss", "isfinite", "iter_ms",
    "ph_forward_ms", "ph_loss_ms", "ph_backward_ms", "ph_densify_ms",
    "ph_optimizer_ms",
]


class D0Tracer:
    def __init__(self, model, scene: str, outdir: str, n_cameras: int,
                 grad_threshold: float, sample_offset: int,
                 cost_samples: bool = True, device: str = "cuda"):
        self.model = model
        self.scene = scene
        self.outdir = outdir
        self.n_cameras = int(n_cameras)
        self.grad_threshold = float(grad_threshold)
        self.sample_offset = int(sample_offset)
        self.cost_samples = bool(cost_samples)
        self.dev = torch.device(device)
        self.cam_words = (self.n_cameras + 63) // 64
        self.samples_dir = os.path.join(outdir, "d0_cost_samples")

        # per-uid arrays ------------------------------------------------------
        self.cap = 0
        self._alloc(262144)
        self.next_uid = 0

        # trajectory / events -------------------------------------------------
        self.traj = {k: [] for k in TRAJ_FIELDS}
        self.events = []          # densify events with D0 counters
        self.checkpoints = []
        self.sample_summary = []   # per sampled iteration scalars
        self.reset_iters = []
        self.cur = None
        self.event_counter = 0
        self.f9_last = None
        self.f9_first_iter = None
        self.f9_transitions = []   # [iter, flag] on change (first iter not recorded)
        self.n_iters_f9 = 0
        self.n_iters_base = 0
        self.m2d_mode = {}         # per-branch means2d space: {f9_flag: str}
        self._space_checked = False
        self._branch_checked = set()
        self.grad_missing_iters = 0
        self.sumcheck_mismatch = []
        self.wall_start = time.time()
        self.notes = []
        # current-iteration captures (set by note_render; None before the first
        # visible iteration so capture_birth_features is safe in isolation)
        self._cur_absgrad_norm_px = None
        self._cur_res_mean_full = None
        self._cur_res_p90_full = None

    # ------------------------------------------------------------------ alloc
    def _alloc(self, cap: int):
        dev = self.dev
        self.cap = int(cap)
        self.u_parent = torch.full((cap,), -1, dtype=torch.int64, device=dev)
        self.u_birth_iter = torch.zeros((cap,), dtype=torch.int32, device=dev)
        self.u_birth_type = torch.zeros((cap,), dtype=torch.uint8, device=dev)
        self.u_birth_event = torch.full((cap,), -1, dtype=torch.int32, device=dev)
        self.u_death_iter = torch.full((cap,), -1, dtype=torch.int32, device=dev)
        self.u_reason = torch.zeros((cap,), dtype=torch.uint8, device=dev)
        self.u_feat = torch.full((N_FEATURES, cap), float("nan"),
                                 dtype=torch.float32, device=dev)
        self.vis_stamp = torch.full((cap, VIS_RING), STAMP_INIT,
                                    dtype=torch.int32, device=dev)
        self.cam_bits = torch.zeros((cap, self.cam_words), dtype=torch.int64,
                                    device=dev)
        self.cam_count = torch.zeros((cap,), dtype=torch.int32, device=dev)
        self.hist_n = torch.zeros((cap,), dtype=torch.int32, device=dev)
        self.res_mean_ring = torch.full((cap, HIST_RING), float("nan"),
                                        dtype=torch.float32, device=dev)
        self.res_p90_ring = torch.full((cap, HIST_RING), float("nan"),
                                       dtype=torch.float32, device=dev)
        self.tiles_ring = torch.full((cap, HIST_RING), float("nan"),
                                     dtype=torch.float32, device=dev)
        self.dir_ring = torch.full((cap, HIST_RING, 2), float("nan"),
                                   dtype=torch.float32, device=dev)
        self.last_tiles = torch.full((cap,), float("nan"),
                                     dtype=torch.float32, device=dev)
        self.last_radius = torch.full((cap,), float("nan"),
                                      dtype=torch.float32, device=dev)
        self.last_depth = torch.full((cap,), float("nan"),
                                     dtype=torch.float32, device=dev)
        self.view_dir_sum = torch.zeros((cap, 3), dtype=torch.float32, device=dev)
        self.tiles_cum = torch.zeros((cap,), dtype=torch.int64, device=dev)

    def _ensure(self, need: int):
        if need <= self.cap:
            return
        new_cap = self.cap
        while new_cap < need:
            new_cap *= 2
        old = {k: getattr(self, k) for k in (
            "u_parent", "u_birth_iter", "u_birth_type", "u_birth_event",
            "u_death_iter", "u_reason", "u_feat", "vis_stamp", "cam_bits",
            "cam_count", "hist_n", "res_mean_ring", "res_p90_ring",
            "tiles_ring", "dir_ring", "last_tiles", "last_radius",
            "last_depth", "view_dir_sum", "tiles_cum")}
        self._alloc(new_cap)
        fill = {"u_parent": -1, "u_birth_iter": 0, "u_birth_type": 0,
                "u_birth_event": -1, "u_death_iter": -1, "u_reason": 0,
                "u_feat": float("nan"), "vis_stamp": STAMP_INIT,
                "cam_bits": 0, "cam_count": 0, "hist_n": 0,
                "res_mean_ring": float("nan"), "res_p90_ring": float("nan"),
                "tiles_ring": float("nan"),
                "dir_ring": float("nan"), "last_tiles": float("nan"),
                "last_radius": float("nan"), "last_depth": float("nan"),
                "view_dir_sum": 0, "tiles_cum": 0}
        for k in list(old):
            v = old.pop(k)
            newt = getattr(self, k)
            newt[tuple(slice(0, s) for s in v.shape)] = v
        del old

    # ------------------------------------------------------------- life cycle
    def register_init(self, n_init: int):
        self._ensure(int(n_init))
        self.next_uid = int(n_init)
        self.u_birth_event[:n_init] = -1
        self.u_birth_type[:n_init] = 0
        self.u_parent[:n_init] = -1

    def register_births(self, new_ids: torch.Tensor, parent_uids: torch.Tensor,
                        birth_type: int, iteration: int, feat: torch.Tensor | None):
        new_ids = new_ids.to(torch.int64)
        if new_ids.numel() == 0:
            return
        self._ensure(int(new_ids.max().item()) + 1)
        self.next_uid = max(self.next_uid, int(new_ids.max().item()) + 1)
        self.u_parent[new_ids] = parent_uids.to(torch.int64)
        self.u_birth_iter[new_ids] = int(iteration)
        self.u_birth_type[new_ids] = int(birth_type)
        self.u_birth_event[new_ids] = self.event_counter
        if feat is not None:
            assert feat.shape == (N_FEATURES, new_ids.numel()), feat.shape
            self.u_feat[:, new_ids] = feat
        if self.cur is not None:
            if birth_type == 1:
                self.cur["births_clone"] += int(new_ids.numel())
            elif birth_type == 2:
                self.cur["births_split"] += int(new_ids.numel())

    def register_deaths(self, uids: torch.Tensor, reasons, iteration: int):
        uids = uids.to(torch.int64)
        if uids.numel() == 0:
            return
        self.u_death_iter[uids] = int(iteration)
        if torch.is_tensor(reasons):
            self.u_reason[uids] = reasons.to(torch.uint8)
        else:
            self.u_reason[uids] = int(reasons)
        if self.cur is None:
            return
        # normalize to one code per death (int callers pass a single code);
        # prunes_total mirrors the trainer's ev["pruned_total"]: every pruned
        # row except split-parent removals (reason 4), which are counted only
        # in split_parent_removed.
        rc = (reasons.to(torch.int64) if torch.is_tensor(reasons)
              else torch.full_like(uids, int(reasons)))
        self.cur["prunes_opacity"] += int((rc == 1).sum().item())
        self.cur["prunes_screen"] += int((rc == 2).sum().item())
        self.cur["prunes_world"] += int((rc == 3).sum().item())
        self.cur["split_parent_removed"] += int((rc == 4).sum().item())
        self.cur["prunes_total"] += int((rc != 4).sum().item())

    def register_opacity_reset(self, iteration: int):
        self.reset_iters.append(int(iteration))
        if self.cur is not None:
            self.cur["opacity_reset"] = 1

    def begin_event(self, iteration: int):
        self.event_counter += 1

    # ------------------------------------------------------------- render hook
    def note_render(self, iteration: int, cam_idx: int, capture: dict,
                    image: torch.Tensor, gt_image: torch.Tensor):
        """Per-iteration per-uid metadata update (runs after backward)."""
        self.cur = {k: 0 for k in TRAJ_FIELDS}
        self.cur["iter"] = int(iteration)
        dev = self.dev
        m = self.model

        vis = capture.get("visible_ids")
        tiles = capture.get("tiles_per_gauss")
        m2d = capture.get("means2d_px_visible")
        proxy = capture.get("means2d_proxy")
        n_isects = capture.get("n_isects")
        space = capture.get("isect_row_space")

        N_t = int(m._xyz.shape[0])
        self.cur["N_t"] = N_t
        self.cur["n_visible"] = int(vis.numel()) if vis is not None else 0
        self.cur["I_t"] = int(n_isects or 0)
        if not self._space_checked:
            self._space_checked = True
            assert space == "visible", f"unexpected isect row space: {space!r}"
        # Frozen-recipe branch note: the gatherless F9 producer is enabled only
        # for sh_degree==3 renders (iterations >= 3000; the earlier SH-degree
        # ramp runs the baseline-native branch).  Both branches share the same
        # capture site, so the exposed metadata (visible-row tiles / depths /
        # pixel-space means2d) and the sumcheck semantics are identical; the
        # branch is recorded per iteration instead of asserted.
        f9 = bool(capture.get("f9_enabled"))
        self.cur["f9"] = int(f9)
        if self.f9_last is not None and f9 != self.f9_last:
            self.f9_transitions.append([int(iteration), f9])
        self.f9_last = f9
        if f9:
            self.n_iters_f9 += 1
            if self.f9_first_iter is None:
                self.f9_first_iter = int(iteration)
        else:
            self.n_iters_base += 1

        # current-iteration densification signal: the exact per-row value
        # add_densification_stats accumulates (absgrad scaled [W/2, H/2]).
        if proxy is not None and hasattr(proxy, "absgrad") and proxy.absgrad is not None:
            g = proxy.absgrad
            if g.dim() == 3:
                g = g[0]
            W = float(capture.get("width") or 0.0)
            H = float(capture.get("height") or 0.0)
            gpx = g * g.new_tensor([W / 2.0, H / 2.0])[None, :]
            self._cur_absgrad_norm_px = gpx.norm(dim=-1).detach()
        else:
            self._cur_absgrad_norm_px = None

        # residual tile grid for the current camera
        with torch.no_grad():
            resid = (image - gt_image).abs().mean(dim=-1)         # [H, W]
            Hh, Ww = int(resid.shape[0]), int(resid.shape[1])
            tsz = 16
            th = math.ceil(Hh / tsz)
            tw = math.ceil(Ww / tsz)
            pad_h = th * tsz - Hh
            pad_w = tw * tsz - Ww
            rp = F.pad(resid[None, None], (0, pad_w, 0, pad_h), mode="replicate")
            tmean = F.avg_pool2d(rp, tsz).reshape(th, tw)
            patches = rp.reshape(th, tsz, tw, tsz).permute(0, 2, 1, 3).reshape(-1, tsz * tsz)
            k = int(math.ceil(0.9 * tsz * tsz))
            tp90 = torch.kthvalue(patches, k, dim=1).values.reshape(th, tw)

        if vis is not None and vis.numel() > 0:
            vis = vis.to(torch.int64)
            uid = m._gaussian_ids[vis]
            V = int(vis.numel())

            # presence checks per renderer branch (base / F9): a missing hook
            # must fail fast instead of silently under-accounting
            if f9 not in self._branch_checked:
                self._branch_checked.add(f9)
                assert tiles is not None, \
                    ("renderer metadata delivered no tiles_per_gauss "
                     f"(f9={f9}; R1/R5 hooks missing)")
                assert m2d is not None, \
                    ("renderer metadata delivered no means2d_px "
                     f"(f9={f9}; R2/R5 hooks missing)")

            if space == "visible" and tiles is not None:
                tv = tiles.reshape(-1).to(torch.float32)
                assert tv.numel() == V, ("tiles_per_gauss not visible-row space",
                                         tv.numel(), V)
                sum_tiles = int(tv.sum().item())
                self.cur["I_t_sumchecked"] = sum_tiles
                if sum_tiles != self.cur["I_t"]:
                    self.sumcheck_mismatch.append(
                        {"iter": int(iteration), "sum": sum_tiles,
                         "n_isects": self.cur["I_t"]})
                self.tiles_cum.index_add_(0, uid, tv.to(torch.int64))
                self.last_tiles[uid] = tv
            else:
                tv = None

            # per-visible depth / radius (visible-row space, aligned to vis)
            dv_vis = capture.get("depths_vis")
            if dv_vis is not None:
                dv_vis = dv_vis.reshape(-1).to(torch.float32)
                assert dv_vis.numel() == V, (dv_vis.numel(), V)
                self.last_depth[uid] = dv_vis
            rfull = capture.get("radii_full")
            if rfull is not None:
                rfull = rfull.reshape(-1, 2)
                assert rfull.shape[0] == N_t, (rfull.shape, N_t)
                self.last_radius[uid] = rfull[vis].max(dim=-1).values.to(torch.float32)

            # pixel-space coords -> center tile residual lookup
            if m2d is not None:
                m2dv = m2d.reshape(-1, 2).to(torch.float32)
                assert m2dv.shape[0] == V, (m2dv.shape, V)
                if f9 not in self.m2d_mode:
                    med = m2dv.abs().median(dim=0).values
                    mode = "pixel" if float(med.max()) > 10.0 else "ndc"
                    self.m2d_mode[f9] = mode
                    assert mode == "pixel", \
                        f"means2d not pixel-space (f9={f9}): " \
                        f"median|x|={med.tolist()}"
                cx = (m2dv[:, 0] / tsz).floor().clamp(0, tw - 1).to(torch.int64)
                cy = (m2dv[:, 1] / tsz).floor().clamp(0, th - 1).to(torch.int64)
                self._cur_res_mean_full = torch.full((N_t,), float("nan"),
                                                     dtype=torch.float32, device=dev)
                self._cur_res_p90_full = torch.full((N_t,), float("nan"),
                                                    dtype=torch.float32, device=dev)
                idx = cy * tw + cx
                self._cur_res_mean_full[vis] = tmean.reshape(-1)[idx]
                self._cur_res_p90_full[vis] = tp90.reshape(-1)[idx]
                # residual ring
                vm = tmean.reshape(-1)[idx]
                vp = tp90.reshape(-1)[idx]
            else:
                self._cur_res_mean_full = None
                self._cur_res_p90_full = None
                vm = vp = None

            # signed gradient direction
            grad = None
            if proxy is not None and getattr(proxy, "grad", None) is not None:
                grad = proxy.grad[0][vis].to(torch.float32)          # [V,2]
            if grad is None:
                self.grad_missing_iters += 1
                dv = None
            else:
                nrm = grad.norm(dim=-1, keepdim=True)
                dv = grad / nrm.clamp_min(1e-12)

            # radius / depth at this observation
            radii = capture.get("radii")
            if radii is not None:
                rv = radii.reshape(-1, 2)[vis].to(torch.float32).max(dim=-1).values \
                    if radii.reshape(-1, 2).shape[0] == V else None
                if rv is not None:
                    self.last_radius[uid] = rv

            # camera bitmap / distinct camera count
            w = int(cam_idx) // 64
            bit = 1 << (int(cam_idx) % 64)
            old = self.cam_bits[uid, w]
            newly = (old & bit) == 0
            self.cam_bits[uid, w] = old | bit
            if newly.any():
                self.cam_count[uid[newly]] += 1

            # stamp ring
            slot = int(iteration) % VIS_RING
            self.vis_stamp[uid, slot] = int(iteration)

            # view-direction sum (camera center -> gaussian, unit)
            cam_center = capture.get("cam_center")
            if cam_center is not None:
                d = m.get_xyz[vis].detach() - cam_center[None, :]
                d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                self.view_dir_sum.index_add_(0, uid, d.to(torch.float32))

            # observation ring (slot shared by res / p90 / tiles / dir);
            # missing values write NaN so stale slots never leak backwards
            cnt = self.hist_n[uid]
            rslot = (cnt % HIST_RING).to(torch.int64)
            if vm is not None:
                self.res_mean_ring[uid, rslot] = vm
                self.res_p90_ring[uid, rslot] = vp
            else:
                self.res_mean_ring[uid, rslot] = float("nan")
                self.res_p90_ring[uid, rslot] = float("nan")
            if tv is not None:
                self.tiles_ring[uid, rslot] = tv
            else:
                self.tiles_ring[uid, rslot] = float("nan")
            if dv is not None:
                self.dir_ring[uid, rslot, 0] = dv[:, 0]
                self.dir_ring[uid, rslot, 1] = dv[:, 1]
            else:
                self.dir_ring[uid, rslot, 0] = float("nan")
                self.dir_ring[uid, rslot, 1] = float("nan")
            self.hist_n[uid] = cnt + 1

            # sampled cost dump
            if self.cost_samples and (int(iteration) % 20) == self.sample_offset and tv is not None:
                keep = tv > 0
                # int32/uint16: provably safe (uid < 2^31, tiles <= n_tiles = 8160)
                assert int(uid.max()) < 2 ** 31 and (int(tv.max()) if tv.numel() else 0) < 65536
                uid_np = uid[keep].cpu().numpy().astype(np.int32)
                tv_np = tv[keep].cpu().numpy().astype(np.uint16)
                os.makedirs(self.samples_dir, exist_ok=True)
                np.savez(os.path.join(self.samples_dir, f"sample_{iteration:05d}.npz"),
                         uid=uid_np, tiles=tv_np)
                self.sample_summary.append({
                    "iter": int(iteration), "cam_idx": int(cam_idx),
                    "I_t": self.cur["I_t"], "sum_tiles": int(sum_tiles),
                    "N_t": N_t, "n_visible": int(V),
                    "n_nonzero": int(keep.sum().item()),
                    "tiles_uid_sum": float(tv_np.sum()),
                })
        else:
            self._cur_res_mean_full = None
            self._cur_res_p90_full = None

    def note_iteration(self, iteration: int, loss: float, phase_ms: dict,
                       iter_ms: float):
        if self.cur is None:
            return
        c = self.cur
        c["loss"] = float(loss)
        c["isfinite"] = int(math.isfinite(loss))
        c["iter_ms"] = float(iter_ms)
        for k_src, k_dst in (("forward", "ph_forward_ms"), ("loss", "ph_loss_ms"),
                             ("backward", "ph_backward_ms"),
                             ("densify", "ph_densify_ms"),
                             ("optimizer", "ph_optimizer_ms")):
            vals = phase_ms.get(k_src)
            c[k_dst] = float(vals[-1]) if vals else float("nan")
        for k in TRAJ_FIELDS:
            self.traj[k].append(c[k])
        self.cur = None

    # ------------------------------------------------------ birth-feature hook
    def capture_birth_features(self, rows: torch.Tensor, branch: int,
                               iteration: int) -> torch.Tensor:
        m = self.model
        rows = rows.to(torch.int64)
        n = int(rows.numel())
        dev = self.dev
        feat = torch.full((N_FEATURES, n), float("nan"), dtype=torch.float32,
                          device=dev)
        if n == 0:
            return feat
        uid = m._gaussian_ids[rows]
        xyz = m.get_xyz[rows].detach()
        scaling = m.get_scaling[rows].detach()
        op = m.get_opacity[rows].detach()
        feat[0] = float(iteration) - self.u_birth_iter[uid].to(torch.float32)
        feat[1] = op
        feat[2] = xyz.norm(dim=1)
        feat[3] = self.last_depth[uid]
        feat[4] = scaling[:, 0]
        feat[5] = scaling[:, 1]
        feat[6] = scaling[:, 2]
        feat[7] = scaling.prod(dim=1)
        feat[8] = self.last_radius[uid]
        feat[9] = float(m.active_sh_degree)
        accum = m.xyz_gradient_accum[rows, 0].detach()
        denom = m.denom[rows, 0].detach()
        feat[10] = accum
        feat[11] = denom
        mean_norm = torch.nan_to_num(accum / denom.clamp_min(1e-12), nan=0.0)
        feat[12] = mean_norm
        feat[13] = mean_norm - self.grad_threshold
        feat[14] = m.max_radii2D[rows].detach()
        if self._cur_absgrad_norm_px is not None:
            feat[15] = self._cur_absgrad_norm_px[rows]
        feat[16] = float(branch)

        stamps = self.vis_stamp[uid]
        age = int(iteration) - stamps
        c32 = (age < 32).sum(dim=1).to(torch.float32)
        c128 = (age < 128).sum(dim=1).to(torch.float32)
        feat[17] = c32
        feat[18] = c128
        feat[19] = c32 / 32.0
        feat[20] = c128 / 128.0
        feat[21] = self.cam_count[uid].to(torch.float32)
        cnt = self.hist_n[uid].to(torch.float32)
        has = cnt > 0
        vsum = self.view_dir_sum[uid]
        spread = 1.0 - (vsum.norm(dim=1) / cnt.clamp_min(1.0)).clamp(max=1.0)
        feat[22] = torch.where(has, spread, torch.full_like(spread, float("nan")))

        if self._cur_res_mean_full is not None:
            feat[23] = self._cur_res_mean_full[rows]
            feat[24] = self._cur_res_p90_full[rows]
        rm = self.res_mean_ring[uid]                     # [n, 32]
        feat[25] = torch.nanmean(rm, dim=1)
        mu = feat[25].unsqueeze(1)
        feat[26] = torch.nanmean((rm - mu) ** 2, dim=1)
        dr = self.dir_ring[uid]                          # [n, 32, 2]
        dmean = torch.nanmean(dr, dim=1)                 # [n,2]
        stability = dmean.norm(dim=1)
        feat[27] = torch.where(has, stability, torch.full_like(stability, float("nan")))
        feat[28] = self.last_tiles[uid]
        feat[29] = torch.nanmean(self.tiles_ring[uid], dim=1)
        return feat

    # ------------------------------------------------ final-state audit (S5)
    def run_final_audit(self, dataset, render_fn, model, iterations):
        """D0 Stage 5 (descriptive): 16 deterministic audit cameras on the
        final state.  Records per-row contribution proxies only; makes no
        removal/selector claim.  The RGB contribution proxy is not cheaply
        exposed by the C0 renderer and is recorded as unavailable.
        """
        n_cam = int(self.n_cameras)
        audit = sorted({int(round(k * (n_cam - 1) / 15.0)) for k in range(16)})
        assert len(audit) == 16, ("audit cameras not distinct", n_cam, audit)
        N = int(model._xyz.shape[0])
        row_uid = model._gaussian_ids.detach().cpu().numpy().astype(np.int64)
        assert row_uid.size == N, (row_uid.size, N)
        # get_opacity already applies sigmoid (activated alpha in [0,1]);
        # the spec's sigma(opacity) is this activated value.
        op = model.get_opacity.detach().reshape(-1).cpu().numpy().astype(np.float64)
        vis_cnt = np.zeros(N, dtype=np.int32)
        vis_mask = np.zeros(N, dtype=np.uint16)   # bit j = visible in audit[j]
        tiles_tot = np.zeros(N, dtype=np.int64)
        fp = np.zeros(N, dtype=np.float64)
        cam_rows = []
        for j, ci in enumerate(audit):
            cam, _gt = dataset.get_item(int(ci))
            with torch.no_grad():
                _img, radii_full, _m2d, cap = render_fn(
                    model, cam, model.active_sh_degree)
            W, H = int(cam.image_width), int(cam.image_height)
            wh = float(W) * float(H)
            rmax = radii_full[0].max(dim=-1).values
            rmax = rmax.detach().cpu().numpy().astype(np.float64)
            vis_ids = cap["visible_ids"].detach().cpu().numpy().astype(np.int64)
            tiles_t = cap.get("tiles_per_gauss")
            tiles = (np.zeros(0, dtype=np.int64) if tiles_t is None
                     else tiles_t.reshape(-1).detach().cpu().numpy().astype(np.int64))
            assert tiles.size == vis_ids.size, (tiles.size, vis_ids.size)
            vis_cnt[vis_ids] += 1
            vis_mask[vis_ids] |= np.uint16(1 << j)
            tiles_tot[vis_ids] += tiles
            area = np.minimum(np.pi * rmax * rmax, wh)
            fp += op * area
            cam_rows.append({"cam_idx": int(ci), "width": W, "height": H,
                             "n_visible": int(vis_ids.size),
                             "sum_tiles": int(tiles.sum()),
                             "n_isects": int(cap.get("n_isects") or 0),
                             "sum_footprint": float((op * area).sum())})
            print("   [D0] audit cam %d/%d idx=%d vis=%d tiles=%d" %
                  (len(cam_rows), len(audit), ci, vis_ids.size,
                   int(tiles.sum())), flush=True)
        keys = sorted(cam_rows[0].keys())
        np.savez(os.path.join(self.outdir, "d0_final_audit.npz"),
                 audit_cameras=np.asarray(audit, dtype=np.int64),
                 row_uid=row_uid,
                 visibility_count=vis_cnt,
                 visibility_mask=vis_mask,
                 tiles_total=tiles_tot,
                 alpha_footprint=fp,
                 per_camera=np.asarray([[c[k] for k in keys] for c in cam_rows],
                                       dtype=np.float64),
                 per_camera_keys=np.array(keys, dtype=object),
                 iterations=np.int64(iterations),
                 n_rows=np.int64(N),
                 opacity_sigma=op.astype(np.float32))
        return {"n_rows": N, "n_audit_cameras": len(audit)}

    # ------------------------------------------------------------- finalize
    def finalize(self, run_meta: dict):
        outdir = self.outdir
        os.makedirs(outdir, exist_ok=True)
        U = self.next_uid
        dev = self.dev

        def np_cpu(t):
            return t.detach().cpu().numpy()

        uid_table = {
            "uid": np.arange(U, dtype=np.int64),
            "parent_uid": np_cpu(self.u_parent[:U]),
            "birth_iter": np_cpu(self.u_birth_iter[:U]),
            "birth_type": np_cpu(self.u_birth_type[:U]),
            "birth_event": np_cpu(self.u_birth_event[:U]),
            "death_iter": np_cpu(self.u_death_iter[:U]),
            "terminal_reason": np_cpu(self.u_reason[:U]),
            "tiles_cum": np_cpu(self.tiles_cum[:U]),
            "cam_count": np_cpu(self.cam_count[:U]),
            "hist_n": np_cpu(self.hist_n[:U]),
        }
        np.savez(os.path.join(outdir, "d0_uid_table.npz"), **uid_table)

        post = np_cpu(self.u_birth_type[:U]) > 0
        feat = np_cpu(self.u_feat[:, :U]).astype(np.float32)
        np.savez(os.path.join(outdir, "d0_birth_records.npz"),
                 uid=np.where(post)[0].astype(np.int64),
                 feat=feat[:, post],
                 feature_names=np.array(FEATURE_NAMES, dtype=object))

        traj = {k: np.asarray(v) for k, v in self.traj.items()}
        np.savez(os.path.join(outdir, "d0_population_trajectory.npz"), **traj)

        if self.sample_summary:
            keys = sorted(self.sample_summary[0].keys())
            np.savez(os.path.join(outdir, "d0_sample_summary.npz"),
                     **{k: np.asarray([s[k] for s in self.sample_summary]) for k in keys})

        meta = {
            "scene": self.scene,
            "n_features": N_FEATURES,
            "feature_names": FEATURE_NAMES,
            "sample_offset": self.sample_offset,
            "cost_samples": self.cost_samples,
            "f9_enabled": self.f9_last,
            "f9_first_iter": self.f9_first_iter,
            "f9_transitions": self.f9_transitions,
            "n_iters_f9": self.n_iters_f9,
            "n_iters_base": self.n_iters_base,
            "means2d_mode": self.m2d_mode.get(False),
            "means2d_mode_f9": self.m2d_mode.get(True),
            "branch_note": ("frozen recipe: baseline-native branch for "
                            "sh_degree<3 iterations, gatherless F9 from the "
                            "iteration the SH ramp reaches degree 3 (3000); "
                            "metadata identical at the shared capture site"),
            "n_samples": len(self.sample_summary),
            "n_uids": int(U),
            "n_births_post_init": int(post.sum()),
            "n_events": int(self.event_counter),
            "grad_missing_iters": int(self.grad_missing_iters),
            "sumcheck_mismatches": self.sumcheck_mismatch[:20],
            "n_sumcheck_mismatches": len(self.sumcheck_mismatch),
            "reset_iters": self.reset_iters,
            "checkpoints": self.checkpoints,
            "wall_s": time.time() - self.wall_start,
            "run_meta": run_meta,
        }
        with open(os.path.join(outdir, "d0_trace_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        return meta
