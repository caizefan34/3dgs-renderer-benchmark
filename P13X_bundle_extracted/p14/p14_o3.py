"""P14 O3: conservative E2E oracle (Sections 16-19). NO production CUDA.

Per scene:
  1. Re-measure same-state C0 components on the union of cluster cameras
     (cull / producer / isect / raster / fwd / loss / bwd / opt dense).
  2. Backward regression bwd ~ alpha + b1*N_vis + b2*M.
  3. Benchmark the masked optimizer (frozen-env SelectiveAdam fallback:
     fused adam kernel absent from the frozen core build) for
     q in {0.10,0.20,0.25}, with bitwise freeze verification AND an
     active-row bitwise comparison against torch.optim.Adam on the
     sliced active rows (same step count, lr, betas, eps).
  4. Conservative compiled-renderer cost model per (cell, q, H):
       fixed floors: cull + producer + isect + loss + bwd fixed part
                     (kernel-launch/sort/producer floors preserved)
       variable parts scaled by R_stream: raster, bwd per-isect part
       cache build: full forward + segment write, per camera, per
                    micro-epoch (H=32 -> 4 visits, H=128 -> 16)
       cache read: segment bytes per step
       invalidated tiles: full charge over the micro-epoch
  5. --finalize merges scenes and evaluates the frozen final gate.
"""
import argparse
import glob
import json
import math
import os
import statistics
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p14_common as C


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


def median_ms(samples):
    return float(statistics.median(samples))


def measure_components(ctx, cam_ids, reps):
    from gsplat.experimental.render.functional.gaussian_inference \
        import _cull_gaussians_batched
    from gsplat.cuda._wrapper import isect_offset_encode, \
        _make_lazy_cuda_func
    from gsplat.experimental.render.kernels import _backend as _ib
    backend = _ib._C
    T = EvTimer()
    rows = []
    m = ctx.model
    for ci in cam_ids:
        cam, gt = ctx.cam_gt(ci)
        width, height = int(cam.image_width), int(cam.image_height)
        tw, th = math.ceil(width / 16), math.ceil(height / 16)
        vm = cam.viewmatrix.unsqueeze(0).unsqueeze(0).contiguous()
        K = cam.K.unsqueeze(0).unsqueeze(0).contiguous()
        means = m._xyz.detach()
        quats = F.normalize(m._rotation.detach(), dim=-1)
        scales = torch.exp(m._scaling.detach())
        opacities = torch.sigmoid(m._opacity.detach()).squeeze(-1)
        colors = m.get_features.detach().contiguous()

        with torch.no_grad():
            def _cull():
                return _cull_gaussians_batched(
                    means, quats, scales, vm, K, width, height,
                    eps2d=C.EPS2D, near_plane=0.01, far_plane=1e10,
                    radius_clip=0.0, camera_model="pinhole")
            vis_ids, _, _ = _cull()

            def _producer():
                cp = backend.higs_camera_positions_from_viewmats(
                    vm[0].contiguous())
                r, m2, d, c_, ob, ce = \
                    backend.higs_gatherless_projected_producer(
                        vis_ids.contiguous(), means.contiguous(),
                        quats.contiguous(), scales.contiguous(),
                        opacities.contiguous(), colors.contiguous(),
                        vm[0].contiguous(), K[0].contiguous(), cp,
                        width, height, C.EPS2D, 0.01, 1e10, 0.0)
                return r, m2, d, c_, ob, ce
            r0, m20, d0, c0_, ob0, ce0 = _producer()
            Cc, N = 1, vis_ids.numel()
            r0 = r0.reshape(1, Cc, N, 2)
            m20 = m20.reshape(1, Cc, N, 2)
            d0 = d0.reshape(1, Cc, N)
            c0_ = c0_.reshape(1, Cc, N, 3)
            ob0 = ob0.reshape(1, Cc, N)
            ce0 = ce0.reshape(1, Cc, N, 3)

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
                        m20.contiguous(), c0_.contiguous(),
                        ce0.contiguous(), ob0.contiguous(), None, None,
                        width, height, 16,
                        off0.reshape(1, Cc, th, tw).contiguous(),
                        ff0.contiguous(), False, False)
                return rc

        n_vis = int(vis_ids.numel())

        # full training step timing (fwd/loss/bwd/opt) with state restore
        p0 = [p.detach().clone() for p in (m._xyz, m._rotation, m._scaling,
                                           m._opacity, m._shs)]
        o_state = {k: {kk: (vv.clone() if torch.is_tensor(vv) else vv)
                       for kk, vv in v.items()}
                   for k, v in m.optimizer.state.items()}

        def _step():
            ev = [torch.cuda.Event(True) for _ in range(5)]
            ev[0].record()
            image = ctx.render_train(cam)
            ev[1].record()
            loss = ctx.fused_loss(image, gt)
            ev[2].record()
            loss.backward()
            ev[3].record()
            m.optimizer.step()
            m.optimizer.zero_grad(set_to_none=True)
            ev[4].record()
            torch.cuda.synchronize()
            for p, q_ in zip((m._xyz, m._rotation, m._scaling,
                              m._opacity, m._shs), p0):
                p.data.copy_(q_)
            for k, v in m.optimizer.state.items():
                for kk, vv in v.items():
                    if torch.is_tensor(vv) and kk in o_state.get(k, {}):
                        vv.copy_(o_state[k][kk])
            return [ev[i].elapsed_time(ev[i + 1]) for i in range(4)]
        for _ in range(3):
            _step()
        fwd_l, loss_l, bwd_l, opt_l = [], [], [], []
        for _ in range(reps):
            f_, l_, b_, o_ = _step()
            fwd_l.append(f_); loss_l.append(l_)
            bwd_l.append(b_); opt_l.append(o_)

        rows.append({
            "camera": int(ci), "n_visible": n_vis, "n_isects": M_isect,
            "cull_ms": median_ms(T.time_fn(_cull, reps)),
            "producer_ms": median_ms(T.time_fn(_producer, reps)),
            "isect_ms": median_ms(T.time_fn(_isect, reps)),
            "raster_ms": median_ms(T.time_fn(_raster, reps)),
            "fwd_ms": median_ms(fwd_l), "loss_ms": median_ms(loss_l),
            "bwd_ms": median_ms(bwd_l), "opt_ms": median_ms(opt_l),
        })
        print(f"  cam {ci}: N_vis={n_vis:,} M={M_isect:,} "
              f"fwd={rows[-1]['fwd_ms']:.2f} bwd={rows[-1]['bwd_ms']:.2f} "
              f"opt={rows[-1]['opt_ms']:.2f}", flush=True)
    return rows


def bench_masked_optimizer(ctx, cam_id, reps):
    """Masked Adam timing + bitwise freeze verification + active-row
    equivalence check vs torch.optim.Adam on the sliced active rows."""
    m = ctx.model
    cam, gt = ctx.cam_gt(cam_id)
    image = ctx.render_train(cam)
    loss = ctx.fused_loss(image, gt)
    loss.backward()
    grads = {n: ctx._param(n).grad.detach().clone()
             for n in C.PARAM_NAMES}
    m.optimizer.zero_grad(set_to_none=True)
    snap = {}
    for n in C.PARAM_NAMES:
        p = ctx._param(n)
        st = m.optimizer.state[p]
        snap[n] = (p.data.clone(), st["exp_avg"].clone(),
                   st["exp_avg_sq"].clone(), st["step"].clone())
    T = EvTimer()
    out = {"reference_camera": int(cam_id), "reps": reps, "by_q": {}}
    npz = np.load(os.path.join(C.P14_ROOT, "raw",
                               f"P14_ACTIVE_{ctx.scene}.npz"))
    for q in C.Q_LIST_LE25:
        act = npz[f"cell0_q{int(round(q*100)):02d}"].astype(np.int64)
        act_idx = torch.as_tensor(act, dtype=torch.long, device="cuda")
        frozen = np.setdiff1d(np.arange(ctx.N, dtype=np.int64), act)
        fr_idx = torch.as_tensor(frozen, dtype=torch.long, device="cuda")
        # restore + attach grads
        def _prep():
            for n in C.PARAM_NAMES:
                p = ctx._param(n)
                st = m.optimizer.state[p]
                p0_, m0_, v0_, s0_ = snap[n]
                p.data.copy_(p0_)
                st["exp_avg"].copy_(m0_)
                st["exp_avg_sq"].copy_(v0_)
                st["step"].copy_(s0_)
                p.grad = grads[n].clone()
        _prep()
        lrs = {g["name"]: float(g["lr"])
               for g in m.optimizer.param_groups}
        it = ctx.it0 + 1
        times = T.time_fn(
            lambda: C.masked_adam_step(ctx, act_idx, it, lrs), reps)
        # ---- verification pass (fresh state, single step) ----
        _prep()
        fr_ref = {n: (ctx._param(n).data[fr_idx].clone(),
                      m.optimizer.state[ctx._param(n)]["exp_avg"]
                      [fr_idx].clone(),
                      m.optimizer.state[ctx._param(n)]["exp_avg_sq"]
                      [fr_idx].clone()) for n in C.PARAM_NAMES}
        C.masked_adam_step(ctx, act_idx, it, lrs)
        frozen_ok = True
        for n in C.PARAM_NAMES:
            p = ctx._param(n)
            st = m.optimizer.state[p]
            pr, mr, vr = fr_ref[n]
            frozen_ok &= bool(torch.equal(p.data[fr_idx], pr)
                              and torch.equal(st["exp_avg"][fr_idx], mr)
                              and torch.equal(st["exp_avg_sq"][fr_idx], vr))
        # ---- active-row equivalence vs torch.optim.Adam (sliced) ----
        _prep()
        C.masked_adam_step(ctx, act_idx, it, lrs)
        mini_ok, mini_maxdiff = True, 0.0
        for n in C.PARAM_NAMES:
            p = ctx._param(n)
            st = m.optimizer.state[p]
            p0_, m0_, v0_, s0_ = snap[n]
            mini_p = p0_[act_idx].clone().requires_grad_(False)
            mini_p.grad = grads[n][act_idx].clone()
            opt = torch.optim.Adam([mini_p], lr=lrs[n], eps=1e-15)
            opt.state[mini_p] = {
                "step": s0_.clone(),
                "exp_avg": m0_[act_idx].clone(),
                "exp_avg_sq": v0_[act_idx].clone()}
            opt.step()
            got = p.data[act_idx]
            d = float((got - mini_p).abs().max())
            mini_maxdiff = max(mini_maxdiff, d)
            mini_ok &= bool(torch.equal(got, mini_p))
        out["by_q"][str(q)] = {
            "n_active": int(len(act)),
            "masked_opt_ms_median": median_ms(times),
            "masked_opt_ms_mean": float(np.mean(times)),
            "frozen_bitwise_unchanged": bool(frozen_ok),
            "active_rows_bitwise_equal_torch_adam": bool(mini_ok),
            "active_rows_max_absdiff_vs_torch_adam": mini_maxdiff,
        }
        print(f"  q={q}: masked_opt={median_ms(times):.3f}ms "
              f"frozen_ok={frozen_ok} active_bitwise={mini_ok} "
              f"(maxdiff {mini_maxdiff:.2e})", flush=True)
    # restore pristine state
    for n in C.PARAM_NAMES:
        p = ctx._param(n)
        st = m.optimizer.state[p]
        p0_, m0_, v0_, s0_ = snap[n]
        p.data.copy_(p0_)
        st["exp_avg"].copy_(m0_)
        st["exp_avg_sq"].copy_(v0_)
        st["step"].copy_(s0_)
        p.grad = None
    return out


def run_scene(scene, gpu, reps):
    t0 = time.time()
    ctx = C.SceneCtx(scene, gpu=gpu)
    ctx.restore_initial()
    clusters = json.load(open(os.path.join(
        C.P14_ROOT, "artifacts", "P14_CAMERA_CLUSTERS.json")))
    cells = clusters["scenes"][scene]["cells"]
    cam_ids = sorted({c for cell in cells
                      for c in cell["cluster_cameras"]})
    print(f"[p14:{scene}] O3 components on {len(cam_ids)} cluster "
          f"cameras", flush=True)
    rows = measure_components(ctx, cam_ids, reps)

    X = np.array([[1.0, r["n_visible"], r["n_isects"]] for r in rows])
    y = np.array([r["bwd_ms"] for r in rows])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else 0.0
    med = {k: float(np.median([r[k] for r in rows])) for k in
           ("cull_ms", "producer_ms", "isect_ms", "raster_ms",
            "fwd_ms", "loss_ms", "bwd_ms", "opt_ms")}
    e2e_base = med["fwd_ms"] + med["loss_ms"] + med["bwd_ms"] \
        + med["opt_ms"]
    comp = {
        "scene": scene, "reps": reps,
        "cameras": [r["camera"] for r in rows],
        "component_ms_median": med,
        "e2e_baseline_ms": e2e_base,
        "n_vis_median": float(np.median([r["n_visible"] for r in rows])),
        "n_isect_median": float(np.median([r["n_isects"] for r in rows])),
        "backward_regression": {
            "alpha": float(beta[0]), "beta_nvis": float(beta[1]),
            "beta_isect": float(beta[2]), "r2": r2},
        "per_camera_rows": rows,
        "gpu_snapshot_start_end": C.gpu_snapshot(),
    }
    print(f"[p14:{scene}] components done ({time.time()-t0:.0f}s): "
          f"e2e={e2e_base:.2f}ms bwd~{beta[0]:.3f}+"
          f"{beta[1]:.2e}*Nvis+{beta[2]:.2e}*M (r2={r2:.3f})", flush=True)

    optbench = bench_masked_optimizer(ctx, cells[0]["cluster_cameras"][0],
                                      reps)
    part = {"scene": scene, "components": comp,
            "optimizer_benchmark": optbench}
    os.makedirs(os.path.join(C.P14_ROOT, "artifacts", "p14_o3_parts"),
                exist_ok=True)
    C.jwrite(os.path.join(C.P14_ROOT, "artifacts", "p14_o3_parts",
                          f"{scene}.json"), part)
    print(f"[p14:{scene}] O3 part written ({time.time()-t0:.0f}s)",
          flush=True)


def finalize():
    o1 = json.load(open(os.path.join(C.P14_ROOT, "artifacts",
                                     "P14_O1_AGG.json")))
    o2 = json.load(open(os.path.join(C.P14_ROOT, "artifacts",
                                     "P14_O2_AGG.json")))
    parts = {}
    for scene in C.SCENES:
        p = os.path.join(C.P14_ROOT, "artifacts", "p14_o3_parts",
                         f"{scene}.json")
        parts[scene] = json.load(open(p))
    o2units = {}
    for path in sorted(glob.glob(os.path.join(C.P14_ROOT,
                                              "P14_O2_UNITS", "*.json"))):
        u = json.load(open(path))
        o2units[(u["scene"], u["cell"], u["q"], u["H"])] = u

    common_q = sorted(set(o1.get("surviving_q", []))
                      & set(o2.get("O2_PASS_q", [])))
    out = {
        "phase": "P14", "scenes": {
            s: parts[s] for s in C.SCENES},
        "common_q_O1_O2": common_q,
        "cost_model": {
            "fixed_floors_per_step": ["cull (full)", "producer (full)",
                                      "isect (full: kernel-launch + sort "
                                      "floor preserved)",
                                      "loss (full)",
                                      "bwd fixed part alpha + beta_nvis "
                                      "* N_vis_median (full)"],
            "variable_scaled_by_R_stream": ["raster fwd", "bwd per-isect "
                                            "part beta_isect * M"],
            "cache_build_per_camera_per_microepoch":
                "cull+producer+isect+raster + n_seg_per_cam * 4112B / BW",
            "cache_read_per_step":
                "n_seg_per_cam * 4112B / BW",
            "invalidated_tiles_full_charge_per_microepoch":
                "R_rebuild * (isect + raster + beta_isect*M) + R_rebuild "
                "* n_seg_per_cam * 4112B / BW",
            "bandwidth_bytes_per_s": C.BW_BYTES_PER_S,
            "visits_per_camera": "H/8 (H=32 -> 4, H=128 -> 16); no "
                                 "extrapolation beyond measured H",
            "selector_cost": "excluded per prereg (oracle-only; NOT a "
                             "speedup source and NOT charged)",
        },
        "per_cell_speedups": {},
        "final_gate": {},
    }

    def seg_per_cam(u):
        return u["n_frozen_segments_max"] / len(u["cameras"])

    gate_detail = {}
    for q in common_q:
        qrec = {"H32": {}, "H128": {}}
        all_ok = True
        p25_cells = {"H32": [], "H128": []}
        for scene in C.SCENES:
            comp = parts[scene]["components"]
            med = comp["component_ms_median"]
            reg = comp["backward_regression"]
            ob = parts[scene]["optimizer_benchmark"]["by_q"][str(q)]
            opt_masked = ob["masked_opt_ms_median"]
            for H in (32, 128):
                s_cells = []
                for cell in range(3):
                    u = o2units.get((scene, cell, q, H))
                    if u is None:
                        s_cells.append(None)
                        continue
                    n_seg_cam = seg_per_cam(u)
                    rs = u["R_stream_max"]
                    rb = u["rebuild"]["R_rebuild"]
                    cache_build = (med["cull_ms"] + med["producer_ms"]
                                   + med["isect_ms"] + med["raster_ms"]
                                   + n_seg_cam * C.CACHE_SEG_BYTES_FP32
                                   / C.BW_BYTES_PER_S * 1e3)
                    cache_read = n_seg_cam * C.CACHE_SEG_BYTES_FP32 \
                        / C.BW_BYTES_PER_S * 1e3
                    bwd_fixed = reg["alpha"] + reg["beta_nvis"] \
                        * comp["n_vis_median"]
                    bwd_var_full = reg["beta_isect"] \
                        * comp["n_isect_median"]
                    step = (med["cull_ms"] + med["producer_ms"]
                            + med["isect_ms"]
                            + rs * med["raster_ms"] + med["loss_ms"]
                            + bwd_fixed + rs * bwd_var_full
                            + opt_masked + cache_read)
                    invalid = rb * (med["isect_ms"] + med["raster_ms"]
                                    + bwd_var_full) \
                        + rb * cache_read
                    micro = 8 * cache_build + H * step + invalid
                    e2e = comp["e2e_baseline_ms"]
                    S = H * e2e / micro
                    s_cells.append(S)
                    out["per_cell_speedups"][
                        f"{scene}/c{cell}/q{q}/H{H}"] = {
                        "R_stream_max": rs, "R_rebuild": rb,
                        "n_seg_per_cam": n_seg_cam,
                        "cache_build_ms": cache_build,
                        "step_ms": step, "invalid_ms_epoch": invalid,
                        "micro_epoch_ms": micro, "e2e_baseline_ms": e2e,
                        "S": S,
                        "opt_masked_ms": opt_masked,
                        "bwd_fixed_ms": bwd_fixed,
                        "bwd_var_full_ms": bwd_var_full}
                valid = [x for x in s_cells if x is not None]
                qrec[f"H{H}"][scene] = {
                    "cells_S": s_cells,
                    "median_S": float(np.median(valid))
                    if valid else None}
                p25_cells[f"H{H}"].extend(
                    x for x in s_cells if x is not None)
        for H in ("H32", "H128"):
            for scene in C.SCENES:
                ms = qrec[H][scene]["median_S"]
                if ms is None or ms < 4.0:
                    all_ok = False
            p25 = float(np.percentile(p25_cells[H], 25)) \
                if p25_cells[H] else None
            qrec[H]["P25_S_all_cells"] = p25
            if p25 is None or p25 < 3.5:
                all_ok = False
        qrec["optimizer_verification"] = {
            s: parts[s]["optimizer_benchmark"]["by_q"][str(q)]
            for s in C.SCENES}
        for s in C.SCENES:
            v = qrec["optimizer_verification"][s]
            if not v["frozen_bitwise_unchanged"]:
                all_ok = False
            if not (v["active_rows_bitwise_equal_torch_adam"]
                    or v["active_rows_max_absdiff_vs_torch_adam"] <= 1e-6):
                all_ok = False
        qrec["FINAL_GATE_PASS"] = bool(all_ok)
        gate_detail[str(q)] = qrec
    out["final_gate"]["by_q"] = gate_detail
    go_q = [q for q in common_q if gate_detail[str(q)]["FINAL_GATE_PASS"]]
    out["final_gate"]["GO_q"] = go_q
    out["final_gate"]["verdict"] = (
        "GO_MINIMAL_COMPILED_RENDER_PROTOTYPE" if go_q else "NO_GO")
    out["final_gate"]["reminder_even_on_go"] = (
        "NO full training; next phase must first solve practical "
        "selector, block rotation, optimizer-state reactivation, global "
        "camera coverage, thousands-step convergence.")
    C.jwrite(os.path.join(C.P14_ROOT, "artifacts", "P14_O3_TIMING.json"),
             out)
    print(json.dumps({"GO_q": go_q,
                      "verdict": out["final_gate"]["verdict"]},
                     indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", choices=C.SCENES)
    ap.add_argument("--gpu", default="4")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--finalize", action="store_true")
    args = ap.parse_args()
    if args.finalize:
        finalize()
        return
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    print(f"[p14] O3 {args.scene} on GPU {args.gpu}\n{C.gpu_snapshot()}",
          flush=True)
    run_scene(args.scene, args.gpu, args.reps)
    print(f"[p14] final gpu snapshot:\n{C.gpu_snapshot()}", flush=True)


if __name__ == "__main__":
    main()
