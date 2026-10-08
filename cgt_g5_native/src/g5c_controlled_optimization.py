#!/usr/bin/env python
"""g5c_controlled_optimization.py - G5-C controlled native optimization (room).

Goal sec.10-13: controlled real-scene optimization, main metric = quality vs WALL-CLOCK.
Three arms on the SAME room tile segment (fixed grouping, R=4 frozen, same init):
  canonical_3dgs : frozen 3DGS teacher field ref_teacher_f64 (quality reference vs GT)
  route_A        : scipy L-BFGS-B + derivative-matching loss, CPU (production fit path)
  native_teacher : GPU Adam + differentiable cgt_student_render, SAME target + SAME loss
  native_GT      : GPU Adam, PURE reconstruction to GT E_rgb (no teacher; G5 candidate)

route_A and native_teacher share target (teacher field) AND loss (recon + dweight*deriv),
so the speedup isolates optimizer + device + renderer. CLEAN TIMING: timed region contains
ONLY optimization steps; PSNR/SSIM trajectories computed post-hoc from untimed coeff
snapshots. Amdahl: native per-iter render/loss/backward/optimizer via CUDA events.
Gate (sec.12): native reaches route-A quality (|dPSNR|<=0.15 vs teacher field at paired
point) AND >=1.25x time-to-quality or total wall speedup.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize

sys.path.insert(0, "/home/liaoyuanjun/cgt_realscene/final/SCRIPTS")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g1_compressibility import predict_and_vjp, initialize, tile_teacher, teacher_jac, TILE  # noqa: E402
from g5_cgt_renderer import cgt_student_render              # noqa: E402

WORKLOAD = "/home/liaoyuanjun/cgt_f0_fused_cuda/results/workload.npz"
GTPATCHES = "/home/liaoyuanjun/cgt_g5_native/results/g5a_gt_patches.npz"
OUTDIR = "/home/liaoyuanjun/cgt_g5_native/results"
RANK = 4
DWEIGHT = 0.02
H = 1e-3
BOUNDS = [(-3, 3), (-3, 3), (-3, 4), (-15, 15), (-3, 4), (-12, 12), (-12, 12), (-12, 12), (-12, 12), (-12, 12)] * RANK


def psnr(a, b):
    return float(-10.0 * np.log10(max(np.mean((a - b) ** 2), 1e-30)))


def ssim_np(pred, true):
    P = pred.shape[0]; Hh = Ww = int(round(P ** 0.5))
    a = torch.tensor(pred.reshape(Hh, Ww, 3), dtype=torch.float32).permute(2, 0, 1)[None]
    b = torch.tensor(true.reshape(Hh, Ww, 3), dtype=torch.float32).permute(2, 0, 1)[None]
    coords = torch.arange(11, dtype=torch.float32) - 5
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2)); g = g / g.sum()
    w = (g[:, None] * g[None, :]).expand(3, 1, 11, 11).contiguous()
    C1, C2 = 0.0001, 0.0009
    mu_a = F.conv2d(a, w, padding=5, groups=3); mu_b = F.conv2d(b, w, padding=5, groups=3)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sig_a = F.conv2d(a * a, w, padding=5, groups=3) - mu_a2
    sig_b = F.conv2d(b * b, w, padding=5, groups=3) - mu_b2
    sig_ab = F.conv2d(a * b, w, padding=5, groups=3) - mu_ab
    s = ((2 * mu_ab + C1) * (2 * sig_ab + C2)) / ((mu_a2 + mu_b2 + C1) * (sig_a + sig_b + C2))
    return float(s.mean())


def eval_curve(coeffs_snaps, px, tgt_te):
    """coeffs_snaps: list of (R,10). Returns (psnr_te list, ssim list) vs tgt_te (P,4)."""
    ps = []; ss = []
    for c in coeffs_snaps:
        pp = np.clip(predict_and_vjp(c, px)[:, :3], 0, 1)
        ps.append(psnr(pp[te], tgt_te[te, :3]))
        ss.append(ssim_np(pp, tgt_te[:, :3]))
    return ps, ss


# ---- route_A objective (numpy, mirrors g1.objective with deriv match) ----
def route_A_obj(flat, k, q, y, tj):
    p = flat.reshape(k, 10)
    pred = predict_and_vjp(p, q)
    err = pred - y; loss = float(np.mean(err ** 2))
    _, grad = predict_and_vjp(p, q, 2 * err / err.size)
    for dd in range(2):
        dq = np.zeros(2); dq[dd] = H
        ed = (predict_and_vjp(p, q + dq) - predict_and_vjp(p, q - dq)) / (2 * H) - tj[:, :, dd]
        loss += DWEIGHT * float(np.mean(ed ** 2)) / 2
        gp = DWEIGHT * ed / (ed.size * 2 * H)
        _, ga = predict_and_vjp(p, q + dq, gp); _, gb = predict_and_vjp(p, q - dq, -gp)
        grad = grad + ga + gb
    return loss, grad.ravel()


def run_route_A(t, iters, px_tr, y_tr, tj_tr, tgt_te, seed):
    p0 = initialize(RANK, px_tr, y_tr, seed=seed)
    snaps = []; times = []
    t0 = time.time()
    def cb(xk):
        snaps.append(xk.reshape(RANK, 10).copy()); times.append(time.time() - t0)
    res = minimize(route_A_obj, p0.ravel(), args=(RANK, px_tr, y_tr, tj_tr), method="L-BFGS-B", jac=True,
                   bounds=BOUNDS, options={"maxiter": iters, "ftol": 1e-13, "gtol": 1e-8, "maxls": 30}, callback=cb)
    wall = time.time() - t0
    # ensure final coeffs included
    final = res.x.reshape(RANK, 10)
    if not snaps or np.max(np.abs(snaps[-1] - final)) > 1e-9:
        snaps.append(final.copy()); times.append(wall)
    return final, wall, times, snaps


def run_native(t, iters, mode, lr, px_tr, y_tr, tj_tr, tgt_te, seed, dev, amdahl=False, snap_every=20):
    if mode == "teacher":
        y_np = y_tr
    else:
        y_np = np.concatenate([y_tr, np.zeros((len(y_tr), 1))], axis=1)
    p0 = initialize(RANK, px_tr, y_np if mode == "teacher" else np.concatenate([y_tr, np.zeros((len(y_tr), 1))], axis=1), seed=seed)
    c = torch.tensor(p0, dtype=torch.float32, device=dev, requires_grad=True)
    opt = torch.optim.Adam([c], lr=lr)
    px_t = torch.tensor(px_tr, dtype=torch.float32, device=dev)
    y_t = torch.tensor(y_tr, dtype=torch.float32, device=dev)
    tj_t = torch.tensor(tj_tr, dtype=torch.float32, device=dev) if (mode == "teacher" and tj_tr is not None) else None
    snaps = []; times = []; acc = {"render": 0.0, "loss": 0.0, "bwd": 0.0, "opt": 0.0}
    use_cuda = "cuda" in dev
    if use_cuda: torch.cuda.synchronize()
    t0 = time.time()
    for it in range(iters):
        opt.zero_grad()
        # --- render (all forward renders incl. derivative FD) ---
        tr0 = time.time()
        pred = cgt_student_render(c, px_t)
        if mode == "teacher":
            dx = torch.zeros_like(px_t); dx[:, 0] = H
            dy = torch.zeros_like(px_t); dy[:, 1] = H
            dpx = (cgt_student_render(c, px_t + dx) - cgt_student_render(c, px_t - dx)) / (2 * H)
            dpy = (cgt_student_render(c, px_t + dy) - cgt_student_render(c, px_t - dy)) / (2 * H)
            jnp = torch.stack([dpx, dpy], dim=2)
        if use_cuda: torch.cuda.synchronize()
        tr1 = time.time()
        # --- loss ---
        tl0 = time.time()
        if mode == "teacher":
            ed = jnp - tj_t
            loss = torch.mean((pred - y_t) ** 2) + DWEIGHT * torch.mean(ed ** 2) / 2.0
        else:
            loss = torch.mean((pred[:, :3] - y_t) ** 2)
        if use_cuda: torch.cuda.synchronize()
        tl1 = time.time()
        # --- backward ---
        tb0 = time.time(); loss.backward()
        if use_cuda: torch.cuda.synchronize()
        tb1 = time.time()
        # --- optimizer ---
        to0 = time.time(); opt.step()
        if use_cuda: torch.cuda.synchronize()
        to1 = time.time()
        acc["render"] += tr1 - tr0; acc["loss"] += tl1 - tl0; acc["bwd"] += tb1 - tb0; acc["opt"] += to1 - to0
        if it % snap_every == 0 or it == iters - 1:
            snaps.append(c.detach().cpu().numpy().copy()); times.append(time.time() - t0)
    wall = time.time() - t0
    # post-hoc curve (untimed)
    ps = []; ss = []
    for cc in snaps:
        with torch.no_grad():
            pp = np.clip(cgt_student_render(torch.tensor(cc, dtype=torch.float32, device=dev),
                              torch.tensor(px_all, dtype=torch.float32, device=dev)).detach().cpu().numpy()[:, :3], 0, 1)
        ps.append(psnr(pp[te], tgt_te[te, :3])); ss.append(ssim_np(pp, tgt_te[:, :3]))
    amd = None
    if amdahl:
        tot = sum(acc.values()) + 1e-30
        amd = {"ms_per_iter": {k: 1000 * v / iters for k, v in acc.items()}, "frac": {k: v / tot for k, v in acc.items()}}
    return c.detach().cpu().numpy(), wall, times, ps, ss, amd


def main():
    global px_all, te
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUTDIR)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--native-iters", type=int, default=400)
    ap.add_argument("--route-iters", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--scene", default="room")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = args.device if (torch.cuda.is_available() or "cpu" in args.device) else "cpu"
    use_cuda = "cuda" in dev
    if use_cuda: torch.cuda.set_device(dev)
    if args.smoke: args.repeats = 1; args.native_iters = 60; args.route_iters = 60

    d = np.load(WORKLOAD)
    coeffs0 = d["coeffs"].astype(np.float64); px = d["px"].astype(np.float64)
    scene = d["scene"]; ref_tch = d["ref_teacher_f64"]
    means2d = d["means2d"]; conics = d["conics"]; opac = d["opac"]; colors = d["colors"]; off = d["tile_off"]
    g = np.load(GTPATCHES); gt_all = g["gt"].astype(np.float64)
    P = px.shape[0]; te = np.arange(1, P, 2); px_all = px
    tr = np.arange(0, P, 2); px_tr = px[tr]
    idx = np.where(scene == args.scene)[0]
    n = len(idx)

    maxd = 0.0
    for t in idx:
        s, e = int(off[t]), int(off[t + 1]); m = e - s; gids = np.arange(m)
        E, T = tile_teacher(px[:, 0:1], px[:, 1:2], gids, means2d[s:e], conics[s:e], opac[s:e], colors[s:e])
        maxd = max(maxd, float(np.max(np.abs(np.concatenate([E, T], axis=1) - ref_tch[t]))))
    slice_ok = maxd < 1e-6
    print(f"[verify] tile slice reproduces ref_teacher_f64: max_abs={maxd:.2e} ok={slice_ok}", flush=True)

    tj0 = time.time(); tj_train = {}
    for t in idx:
        s, e = int(off[t]), int(off[t + 1]); gids = np.arange(e - s)
        tj_train[int(t)] = teacher_jac(px_tr[:, 0:1], px_tr[:, 1:2], gids, means2d[s:e], conics[s:e], opac[s:e], colors[s:e], h=H)
    tj_time = time.time() - tj0

    if use_cuda:  # warmup
        run_native(int(idx[0]), 5, "teacher", args.lr, px_tr, ref_tch[int(idx[0])][tr], tj_train[int(idx[0])], ref_tch[int(idx[0])], int(idx[0]) % (2**31), dev)
        torch.cuda.synchronize()

    # canonical teacher reference vs GT
    canon = {"psnr_heldout_median_vs_GT": float(np.median([psnr(np.clip(ref_tch[t][te, :3], 0, 1), gt_all[t][te]) for t in idx])),
             "ssim_mean_vs_GT": float(np.mean([ssim_np(np.clip(ref_tch[t][:, :3], 0, 1), gt_all[t]) for t in idx]))}

    # route_A
    print(f"[route_A] {n} tiles x {args.route_iters} iters (CPU L-BFGS-B)...", flush=True)
    rA = {"wall": [], "psnr_curve": [], "times": [], "final_psnr": []}
    for t in idx:
        t = int(t)
        final, wall, times, snaps = run_route_A(t, args.route_iters, px_tr, ref_tch[t][tr], tj_train[t], ref_tch[t], t % (2**31))
        ps, ss = eval_curve(snaps, px, ref_tch[t])
        rA["wall"].append(wall); rA["times"].append(times); rA["psnr_curve"].append(ps); rA["final_psnr"].append(ps[-1])
    rA_wall_med = float(np.median(rA["wall"])); rA_psnr_med = float(np.median(rA["final_psnr"]))
    print(f"[route_A] wall/tile med={rA_wall_med*1000:.1f}ms psnr_te med={rA_psnr_med:.3f}", flush=True)

    # native_teacher (repeats)
    print(f"[native_teacher] {args.repeats}x {n} tiles x {args.native_iters} iters (GPU Adam)...", flush=True)
    nT = {"wall": [], "psnr_curve": [], "times": [], "ssim_curve": [], "final_psnr": [], "final_ssim": [], "amdahl": []}
    for rep in range(args.repeats):
        for t in idx:
            t = int(t)
            amd = (rep == 0 and t == int(idx[0]))
            final, wall, times, ps, ss, amd = run_native(t, args.native_iters, "teacher", args.lr, px_tr, ref_tch[t][tr], tj_train[t], ref_tch[t], t % (2**31), dev, amdahl=amd)
            nT["wall"].append(wall); nT["times"].append(times); nT["psnr_curve"].append(ps); nT["ssim_curve"].append(ss)
            nT["final_psnr"].append(ps[-1]); nT["final_ssim"].append(ss[-1])
            if amd is not None: nT["amdahl"].append(amd)
        if use_cuda: torch.cuda.synchronize()
    nT_wall_med = float(np.median(nT["wall"])); nT_psnr_med = float(np.median(nT["final_psnr"])); nT_ssim_mean = float(np.mean(nT["final_ssim"]))
    print(f"[native_teacher] wall/tile med={nT_wall_med*1000:.1f}ms psnr_te med={nT_psnr_med:.3f} ssim={nT_ssim_mean:.4f}", flush=True)

    # native_GT (pure recon to GT)
    print(f"[native_GT] {n} tiles x {args.native_iters} iters (GPU Adam, pure recon to GT)...", flush=True)
    nG = {"wall": [], "psnr_curve": [], "times": [], "final_psnr": [], "final_ssim": []}
    for t in idx:
        t = int(t)
        final, wall, times, ps, ss, amd = run_native(t, args.native_iters, "gt", args.lr, px_tr, gt_all[t][tr], None, np.concatenate([gt_all[t], np.zeros((P, 1))], axis=1), t % (2**31), dev)
        nG["wall"].append(wall); nG["times"].append(times); nG["psnr_curve"].append(ps); nG["final_ssim"].append(ss[-1]); nG["final_psnr"].append(ps[-1])
    nG_wall_med = float(np.median(nG["wall"])); nG_psnr_med = float(np.median(nG["final_psnr"])); nG_ssim_mean = float(np.mean(nG["final_ssim"]))
    print(f"[native_GT] wall/tile med={nG_wall_med*1000:.1f}ms psnr_te(vsGT) med={nG_psnr_med:.3f} ssim={nG_ssim_mean:.4f}", flush=True)

    def time_to_q(times, psnr_curve, thr):
        tt = []
        for tm, pv in zip(times, psnr_curve):
            for t_, p_ in zip(tm, pv):
                if p_ >= thr: tt.append(t_); break
        return float(np.median(tt)) if tt else None

    thresholds = [30, 35, 40, 45]
    ttq = {}
    for thr in thresholds:
        a = time_to_q(rA["times"], rA["psnr_curve"], thr)
        b = time_to_q(nT["times"], nT["psnr_curve"], thr)
        ttq[str(thr)] = {"route_A_s": a, "native_teacher_s": b, "speedup": (a / b) if (a and b) else None}

    speedup_total = (rA_wall_med / nT_wall_med) if nT_wall_med > 0 else None
    dpsnr_qual = abs(nT_psnr_med - rA_psnr_med)
    qual_pass = dpsnr_qual <= 0.15
    ttq_speedups = [v["speedup"] for v in ttq.values() if v["speedup"]]
    ttq_max = max(ttq_speedups) if ttq_speedups else 0.0
    speedup_pass = (speedup_total is not None and speedup_total >= 1.25) or (ttq_max >= 1.25)
    verdict = "PASS_NATIVE_OPTIMIZATION" if (qual_pass and speedup_pass) else "DROP_NATIVE_OPTIMIZATION"

    out = {
        "stage": "G5-C", "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "scene": args.scene, "n_tiles": n, "rank": RANK,
        "device": dev, "native_iters": args.native_iters, "route_iters": args.route_iters, "lr": args.lr, "repeats": args.repeats,
        "smoke": bool(args.smoke),
        "slice_verify": {"max_abs": maxd, "ok": bool(slice_ok)},
        "teacher_jac_precompute_s": tj_time,
        "canonical_teacher_ref_vs_GT": canon,
        "route_A": {"wall_per_tile_median_s": rA_wall_med, "psnr_heldout_median": rA_psnr_med, "wall_per_tile": rA["wall"], "psnr_heldout_per_tile": rA["final_psnr"]},
        "native_teacher": {"wall_per_tile_median_s": nT_wall_med, "psnr_heldout_median": nT_psnr_med, "ssim_mean": nT_ssim_mean,
                           "wall_per_tile_repeats": nT["wall"], "psnr_heldout_per_tile": nT["final_psnr"], "ssim_per_tile": nT["final_ssim"],
                           "amdahl_per_iter": nT["amdahl"][0] if nT["amdahl"] else None},
        "native_GT": {"wall_per_tile_median_s": nG_wall_med, "psnr_vs_GT_median": nG_psnr_med, "ssim_mean": nG_ssim_mean,
                      "wall_per_tile": nG["wall"], "psnr_vs_GT_per_tile": nG["final_psnr"], "ssim_per_tile": nG["final_ssim"]},
        "time_to_quality_s": ttq,
        "total_wall_speedup": speedup_total,
        "gate": {"quality_dPSNR_vs_routeA": dpsnr_qual, "quality_pass": bool(qual_pass),
                 "ttq_max_speedup": ttq_max, "total_speedup": speedup_total, "speedup_pass": bool(speedup_pass), "threshold": 1.25},
        "verdict": verdict,
        "notes": "route_A & native_teacher share target (teacher field) + loss (recon+dweight*deriv-match) + init; speedup isolates optimizer+device+renderer. native_GT = pure recon to GT (no teacher). route_A CPU L-BFGS-B; native GPU Adam. Clean timing: timed region = optimization steps only; PSNR/SSIM curves post-hoc from untimed snapshots. Amdahl = native per-iter render/loss/backward/optimizer.",
    }
    with open(os.path.join(args.out, "G5C_CONTROLLED_OPTIMIZATION.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({"verdict": verdict, "qual_pass": qual_pass, "dPSNR": dpsnr_qual,
                      "total_speedup": speedup_total, "ttq_max": ttq_max,
                      "route_A_wall_ms": rA_wall_med * 1000, "native_teacher_wall_ms": nT_wall_med * 1000,
                      "native_GT_psnr_vsGT": nG_psnr_med, "slice_ok": slice_ok}, indent=2))
    print("WROTE G5C_CONTROLLED_OPTIMIZATION.json")


if __name__ == "__main__":
    main()
