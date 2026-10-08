#!/usr/bin/env python3
"""P13-X X6 oracle timing (after X1-X5 are collected; no implementation claim).

Measures current C0 component times with CUDA events on the frozen 16K state
and constructs conservative upper bounds from measured components ONLY:
  - Delta oracle: skip work classified reusable by X2/X3 (dirty fraction at
    eps=1e-5); charge measured unchanged overheads (culling, producer,
    attribute comparison, full backward, loss, optimizer).
  - Adjoint oracle: replace pixel-adjoint interactions by the measured
    K-basis interaction count (rho = K/256 per tile) applied to the
    per-intersection part of the backward (split via a measured
    T_bwd ~ alpha + beta1*N_vis + beta2*M regression); charge the measured
    basis transform cost.
  - Composed conservative oracle: dirty fraction applies to forward
    isect+raster AND the per-isect backward part; adjoint ratio applies only
    to the dirty portion (no multiplicative independence assumption).

State is kept frozen across repetitions (params + optimizer state restored
in-place after every timed rep). Output: P13X_X6_ORACLE_RAW.json.
"""
import argparse
import importlib.util
import json
import math
import os
import statistics
import sys

import numpy as np
import torch
import torch.nn.functional as F

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


def median_ms(samples):
    return float(statistics.median(samples))


class EvTimer:
    def __init__(self):
        self.ev = [(torch.cuda.Event(True), torch.cuda.Event(True))
                   for _ in range(64)]

    def time_fn(self, fn, reps=30, warmup=5):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        out = []
        for i in range(reps):
            s, e = self.ev[i % len(self.ev)]
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            out.append(s.elapsed_time(e))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="room")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--p13x-raw-dir", required=True,
                    help="the diag run's p13x/raw directory (for X2/X3 d)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--extra-cameras", type=int, default=16)
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)

    bootstrap_c0()
    from dataset import GTDataset
    from gaussian_model import GaussianModel
    from config import ReferenceV1Config
    from gsplat.experimental.render.functional.gaussian_inference import (
        _cull_gaussians_batched, _higs_dynamic_forward)
    from gsplat.cuda._wrapper import (isect_offset_encode,
                                      _make_lazy_cuda_func)
    from gsplat.experimental.render.kernels import _backend as _ib
    import fused_ssim as fused_ssim_mod

    backend = _ib._C
    config = ReferenceV1Config()
    config.scene = args.scene
    dataset = GTDataset(scene=args.scene, repo_root=REPO, resolution="1080p",
                        device="cuda", background="black")
    n_cameras = len(dataset)

    model = GaussianModel(max_sh_degree=config.sh_degree)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
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
    model.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"  restored N={model._xyz.shape[0]:,} from {args.ckpt}")

    def loss_fn(pred_hwc, target_hwc):
        b = pred_hwc.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
        g = target_hwc.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
        L1 = F.l1_loss(b, g)
        ssim = fused_ssim_mod.fused_ssim(b, g, padding="same", train=True)
        return (1.0 - config.lambda_dssim) * L1 \
            + config.lambda_dssim * (1.0 - ssim)

    # camera set: 8 anchor rule + evenly spaced extras (deterministic)
    anchors = sorted(set(int(i) for i in np.round(
        np.linspace(0, n_cameras - 1, 8)).astype(int)))
    extras = sorted(set(int(i) for i in np.round(
        np.linspace(0, n_cameras - 1, args.extra_cameras)).astype(int)))
    cams = sorted(set(anchors) | set(extras))
    eps2d = 0.3
    T = EvTimer()
    rows = []
    K_LIST = [4, 8, 16, 32, 64, 128, 256]

    for ci in cams:
        cam, gt = dataset.get_item(ci)
        width, height = int(cam.image_width), int(cam.image_height)
        tw, th = math.ceil(width / 16), math.ceil(height / 16)
        vm = cam.viewmatrix.unsqueeze(0).unsqueeze(0).contiguous()
        K = cam.K.unsqueeze(0).unsqueeze(0).contiguous()
        means = model._xyz
        quats = F.normalize(model._rotation, dim=-1)
        scales = torch.exp(model._scaling)
        opacities = torch.sigmoid(model._opacity).squeeze(-1)
        colors = model.get_features

        # ---- diagnostic pieces (frozen state) ----
        with torch.no_grad():
            def _cull():
                return _cull_gaussians_batched(
                    means, quats, scales, vm, K, width, height,
                    eps2d=eps2d, near_plane=0.01, far_plane=1e10,
                    radius_clip=0.0, camera_model="pinhole")
            vis_ids, _, _ = _cull()

            def _producer():
                cp = backend.higs_camera_positions_from_viewmats(vm[0].contiguous())
                r, m2, d, c, ob, ce = backend.higs_gatherless_projected_producer(
                    vis_ids.contiguous(), means.contiguous(), quats.contiguous(),
                    scales.contiguous(), opacities.contiguous(), colors.contiguous(),
                    vm[0].contiguous(), K[0].contiguous(), cp,
                    width, height, eps2d, 0.01, 1e10, 0.0)
                return r, m2, d, c, ob, ce
            r0, m20, d0, c0_, ob0, ce0 = _producer()
            Cc, N = 1, vis_ids.numel()
            r0 = r0.reshape(1, Cc, N, 2); m20 = m20.reshape(1, Cc, N, 2)
            d0 = d0.reshape(1, Cc, N); c0_ = c0_.reshape(1, Cc, N, 3)
            ob0 = ob0.reshape(1, Cc, N); ce0 = ce0.reshape(1, Cc, N, 3)

            def _isect():
                _, ii, ff = torch.ops.gsplat.intersect_tile(
                    m20, r0, d0, c0_, ob0, None, None, Cc, 16, tw, th,
                    True, False)
                off = isect_offset_encode(ii, Cc, tw, th)
                return ii, ff, off
            ii0, ff0, off0 = _isect()
            M_isect = int(ff0.shape[0])

            def _raster():
                rc, ra, ab, li = _make_lazy_cuda_func(
                    "rasterize_to_pixels_3dgs")(
                    m20.contiguous(), c0_.contiguous(), ce0.contiguous(),
                    ob0.contiguous(), None, None, width, height, 16,
                    off0.reshape(1, Cc, th, tw).contiguous(),
                    ff0.contiguous(), False, False)
                return rc
        n_vis = int(vis_ids.numel())
        cull_ms = median_ms(T.time_fn(_cull, args.reps))
        prod_ms = median_ms(T.time_fn(_producer, args.reps))
        isect_ms = median_ms(T.time_fn(_isect, args.reps))
        raster_ms = median_ms(T.time_fn(_raster, args.reps))

        # ---- attribute comparison cost (delta-execution overhead) ----
        def _compare():
            ch = False
            for sl in (slice(0, 2), slice(2, 5), slice(5, 6),
                       slice(6, 9), slice(9, 10), slice(10, 12)):
                a = torch.cat([m20.reshape(Cc, N, 2)[0], c0_.reshape(Cc, N, 3)[0],
                               ob0.reshape(Cc, N, 1)[0], ce0.reshape(Cc, N, 3)[0],
                               d0.reshape(Cc, N, 1)[0], r0.reshape(Cc, N, 2)[0]],
                              dim=1)[:, sl]
                ch = ch or bool((a - a).abs().max() > 0)  # same-cost dummy
            return ch
        compare_ms = median_ms(T.time_fn(_compare, args.reps))

        # ---- E2E training-iteration structure (frozen state) ----
        p0 = [p.detach().clone() for p in
              (model._xyz, model._rotation, model._scaling, model._opacity,
               model._shs)]
        o_state = {k: {kk: (vv.clone() if torch.is_tensor(vv) else vv)
                       for kk, vv in v.items()}
                   for k, v in model.optimizer.state.items()}

        def _e2e():
            torch.cuda.synchronize()
            s = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
            e2 = torch.cuda.Event(True); e3 = torch.cuda.Event(True)
            e4 = torch.cuda.Event(True)
            s.record()
            out = _higs_dynamic_forward(
                model._xyz, model.get_rotation, model.get_scaling,
                model.get_opacity, model.get_features,
                viewmats=vm, Ks=K, width=width, height=height,
                sh_degree=model.active_sh_degree,
                backward_mode="higs_native", enable_culling=True,
                camera_model="pinhole", render_mode="RGB", eps2d=eps2d)
            image = out["frame"].reshape(height, width, 3).clamp(0, 1)
            e1.record()
            loss = loss_fn(image, gt)
            e2.record()
            loss.backward()
            e3.record()
            model.optimizer.step()
            model.optimizer.zero_grad(set_to_none=True)
            e4.record()
            torch.cuda.synchronize()
            # restore frozen state in-place
            for p, q in zip((model._xyz, model._rotation, model._scaling,
                             model._opacity, model._shs), p0):
                p.data.copy_(q)
            for k, v in model.optimizer.state.items():
                for kk, vv in v.items():
                    if torch.is_tensor(vv) and kk in o_state.get(k, {}):
                        vv.copy_(o_state[k][kk])
            return (s.elapsed_time(e1), e1.elapsed_time(e2),
                    e2.elapsed_time(e3), e3.elapsed_time(e4))
        for _ in range(3):
            _e2e()
        fwd_l, loss_l, bwd_l, opt_l = [], [], [], []
        for _ in range(args.reps):
            f, l, b, o = _e2e()
            fwd_l.append(f); loss_l.append(l); bwd_l.append(b); opt_l.append(o)

        rows.append({
            "camera": ci, "n_visible": n_vis, "n_isects": M_isect,
            "cull_ms": cull_ms, "producer_ms": prod_ms,
            "isect_ms": isect_ms, "raster_ms": raster_ms,
            "compare_ms": compare_ms,
            "fwd_ms": median_ms(fwd_l), "loss_ms": median_ms(loss_l),
            "bwd_ms": median_ms(bwd_l), "opt_ms": median_ms(opt_l),
        })
        print(f"  cam {ci}: N_vis={n_vis:,} M={M_isect:,} "
              f"fwd={rows[-1]['fwd_ms']:.2f} bwd={rows[-1]['bwd_ms']:.2f} "
              f"opt={rows[-1]['opt_ms']:.2f}", flush=True)

    # ---- DCT transform cost (actual basis transform; data-independent) ----
    cam0, _ = dataset.get_item(cams[0])
    H0, W0 = int(cam0.image_height), int(cam0.image_width)
    adj = torch.linspace(-1, 1, H0 * W0 * 3, device="cuda").reshape(
        H0, W0, 3).contiguous()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import p13x_diag
    dct_ms = {}
    for Kk in K_LIST:
        dct_ms[str(Kk)] = median_ms(
            T.time_fn(lambda: p13x_diag._dct_reconstruct(adj, Kk), args.reps))

    # ---- backward split regression: bwd ~ alpha + b1*N_vis + b2*M ----
    X = np.array([[1.0, r["n_visible"], r["n_isects"]] for r in rows])
    y = np.array([r["bwd_ms"] for r in rows])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else 0.0

    # ---- dirty fractions from X2/X3 raw (eps=1e-5, lag=250) ----
    x3 = json.load(open(os.path.join(args.p13x_raw_dir,
                                     "P13X_RAW_X3.json")))
    x2 = json.load(open(os.path.join(args.p13x_raw_dir,
                                     "P13X_RAW_X2.json")))
    d_isect = [r["eps_1e-05"]["dirty_intersection_fraction"]
               for r in x3 if r.get("lag") == 250]
    d_tile = [r["eps_1e-05"]["dirty_tile_fraction"]
              for r in x3 if r.get("lag") == 250]
    d_med = float(np.median(d_isect)) if d_isect else None
    dt_med = float(np.median(d_tile)) if d_tile else None

    # ---- oracle construction (medians over cameras) ----
    med = {k: float(np.median([r[k] for r in rows])) for k in
           ("cull_ms", "producer_ms", "isect_ms", "raster_ms", "compare_ms",
            "fwd_ms", "loss_ms", "bwd_ms", "opt_ms")}
    e2e_base = med["fwd_ms"] + med["loss_ms"] + med["bwd_ms"] + med["opt_ms"]
    n_vis_med = float(np.median([r["n_visible"] for r in rows]))
    m_med = float(np.median([r["n_isects"] for r in rows]))
    alpha, b1, b2 = float(beta[0]), float(beta[1]), float(beta[2])

    out = {
        "scene": args.scene, "ckpt": args.ckpt,
        "n_cameras_measured": len(rows), "reps": args.reps,
        "component_ms_median": med,
        "e2e_baseline_ms": e2e_base,
        "p12_reference_mean_iter_ms_room": None,
        "backward_regression": {"alpha": alpha, "beta_nvis": b1,
                                "beta_isect": b2, "r2": r2,
                                "n": len(rows)},
        "dct_transform_ms": dct_ms,
        "dirty_fraction_eps1e5_lag250": {
            "median_dirty_isect": d_med, "median_dirty_tile": dt_med,
            "n_pairs": len(d_isect)},
        "oracles": {},
    }
    if d_med is not None:
        # Delta oracle (conservative): forward-side reuse only; full backward
        T_delta = (med["cull_ms"] + med["producer_ms"] + med["compare_ms"]
                   + d_med * (med["isect_ms"] + med["raster_ms"])
                   + med["bwd_ms"] + med["loss_ms"] + med["opt_ms"])
        # Delta oracle (optimistic): backward per-isect part also scaled by d
        T_delta_opt = (med["cull_ms"] + med["producer_ms"] + med["compare_ms"]
                       + d_med * (med["isect_ms"] + med["raster_ms"])
                       + alpha + b1 * n_vis_med + d_med * b2 * m_med
                       + med["loss_ms"] + med["opt_ms"])
        out["oracles"]["delta"] = {
            "T_ms": T_delta, "speedup_vs_e2e": e2e_base / T_delta,
            "T_ms_optimistic_bwd_reuse": T_delta_opt,
            "speedup_optimistic": e2e_base / T_delta_opt,
            "dirty_isect_median": d_med,
        }
        for Kk in K_LIST:
            rho = Kk / 256.0
            # Adjoint oracle: basis interactions replace pixel-adjoint
            # interactions in the rasterizer backward (per-isect part only)
            T_adj = (med["cull_ms"] + med["producer_ms"] + med["isect_ms"]
                     + med["raster_ms"] + med["loss_ms"]
                     + dct_ms[str(Kk)]
                     + alpha + b1 * n_vis_med + rho * b2 * m_med
                     + med["opt_ms"])
            # Composed conservative: adjoint ratio applies only to the DIRTY
            # portion of the per-isect backward (no independence assumption)
            T_comp = (med["cull_ms"] + med["producer_ms"] + med["compare_ms"]
                      + dct_ms[str(Kk)]
                      + d_med * (med["isect_ms"] + med["raster_ms"])
                      + med["loss_ms"]
                      + alpha + b1 * n_vis_med + d_med * rho * b2 * m_med
                      + med["opt_ms"])
            out["oracles"][f"adjoint_K{Kk}"] = {
                "T_ms": T_adj, "speedup_vs_e2e": e2e_base / T_adj,
                "rho": rho}
            out["oracles"][f"composed_K{Kk}"] = {
                "T_ms": T_comp, "speedup_vs_e2e": e2e_base / T_comp,
                "rho": rho, "dirty_isect_median": d_med}
    out["per_camera"] = rows

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "P13X_X6_ORACLE_RAW.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "per_camera"},
                     indent=2))


if __name__ == "__main__":
    main()
