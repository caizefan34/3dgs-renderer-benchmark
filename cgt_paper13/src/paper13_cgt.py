#!/usr/bin/env python
"""CGT-PAPER13 CGT mixed-forward benchmark (P2 core). EXTENDS F0 3-scene evidence to 13 scenes.

FROZEN semantics (inherit F0 verbatim, NO retuning):
  R=4, GATE_MAX_ABS=0.02 (rigorous cert), kernel-correctness thresh 2e-5, TILE=16,
  centered-px teacher (g1.tile_teacher), certify_tile (g2), single fused kernel (f0_kernels.so),
  seed 20261004, work-decile-stratified tile sampling.
Per scene: extract tiles -> fit_cgt(R4) + certify_tile -> refs (teacher/student, f32+f64)
  -> time teacher vs fused kernel (CUDA events) -> correctness.
Reports: S_mixed (kernel-only, build excluded), certificate stats, kernel-correctness (2e-5),
  CGT-vs-canonical tile error (max_abs/PSNR).
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

SCRIPTS = "/home/liaoyuanjun/cgt_realscene/final/SCRIPTS"
F0_SRC = "/home/liaoyuanjun/cgt_f0_fused_cuda/src"
sys.path.insert(0, SCRIPTS); sys.path.insert(0, F0_SRC)
import ctypes
import torch
import gsplat
from g1_compressibility import (read_cameras_bin, read_images_bin, q2R,
                                tile_teacher, fit_cgt, predict_and_vjp, TILE, POWER_CLIP, GATE_MAX_ABS)
from g2_rigorous_certificate import certify_tile, extract_tiles

R = 4
GATE = GATE_MAX_ABS
CORRECT_THRESH = 2e-5
NPX = TILE * TILE
SEED = 20261004

def shared_px():
    xs = np.arange(TILE) + 0.5; ys = np.arange(TILE) + 0.5
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    px = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float64)
    cx0, cy0 = px.mean(0); px[:, 0] -= cx0; px[:, 1] -= cy0
    return np.ascontiguousarray(px, dtype=np.float64)

def refs_for_tile(px, gids, m2, con, op, cl, coeffs):
    out = {}
    for dt in (np.float64, np.float32):
        pxd = px.astype(dt); m2d = m2.astype(dt); cod = con.astype(dt); od = op.astype(dt); cd = cl.astype(dt); gid = gids.astype(np.int64)
        E, T = tile_teacher(pxd[:, 0:1], pxd[:, 1:2], gid, m2d, cod, od, cd)
        ref_teacher = np.concatenate([E, T], axis=1).astype(dt)
        ref_student = predict_and_vjp(coeffs.astype(dt), pxd).astype(dt)
        out["teacher_" + dt.__name__] = np.ascontiguousarray(ref_teacher)
        out["student_" + dt.__name__] = np.ascontiguousarray(ref_student)
    return out

def build_scene_workload(scene, ckpt, data_root, views, tiles_per_view, seed, rank=R):
    """F0-STYLE view selection + build. Inherits f0_prepare semantics exactly.
    Per-scene rng_views=default_rng(seed); view_idx=sorted(choice(n_views,views));
    per-view seed=seed+vi*101; per-tile fit_seed=seed+tc*1000+R (tc per-scene counter).
    (For room, the first F0 scene, this reproduces F0 tiles exactly.)"""
    px = shared_px()
    imgs = read_images_bin(f"{data_root}/datasets/mipnerf360/{scene}/sparse/0/images.bin")
    n_views = len(imgs)
    rng_views = np.random.default_rng(seed)
    view_idx = sorted(rng_views.choice(n_views, size=min(views, n_views), replace=False).tolist())
    print(f"  [{scene}] view_idx={view_idx}", flush=True)
    all_tiles = []
    for vi in view_idx:
        vseed = seed + int(vi) * 101
        try:
            all_tiles.extend(extract_tiles(scene, ckpt, data_root, int(vi), tiles_per_view, vseed))
        except Exception as ex:
            print(f"  [{scene}] view {vi} extract failed: {ex}", flush=True)
    if not all_tiles:
        return None
    m2l, col_, opl, cll, cfl = [], [], [], [], []
    safel, Ml, rigl, leavesl = [], [], [], []
    rt_f32, rs_f32, rt_f64, rs_f64 = [], [], [], []
    tile_off = [0]; tc = 0
    for t in all_tiles:
        gids = t["gids"]; m2 = t["means2d"]; con = t["conics"]; op = t["opac"]; cl = t["colors"]
        M = int(gids.shape[0])
        E, T = tile_teacher(px[:, 0:1], px[:, 1:2], gids, m2, con, op, cl)
        y = np.concatenate([E, T], axis=1)
        P = int(y.shape[0])
        tr = np.arange(0, P, 2); yt = y[tr]
        qt = np.column_stack([px[tr, 0], px[tr, 1]]).astype(np.float64)
        fit_seed = seed + tc * 1000 + R
        p = fit_cgt(qt, yt, rank, fit_seed, 200)
        res = certify_tile(t, p, 1.0, 10, np.random.default_rng(fit_seed + 7), True)
        safel.append(int(res["passed"])); rigl.append(float(res["rigorous_max_abs"]))
        leavesl.append(int(res.get("n_leaves", 0))); tc += 1
        p_r = np.ascontiguousarray(p, dtype=np.float64).reshape(rank, 10)
        cfl.append(p_r)
        refs = refs_for_tile(px, gids, m2, con, op, cl, p_r)
        rt_f32.append(refs["teacher_float32"]); rs_f32.append(refs["student_float32"])
        rt_f64.append(refs["teacher_float64"]); rs_f64.append(refs["student_float64"])
        m2l.append(np.ascontiguousarray(m2[gids], dtype=np.float64))
        col_.append(np.ascontiguousarray(con[gids], dtype=np.float64))
        opl.append(np.ascontiguousarray(op[gids], dtype=np.float64))
        cll.append(np.ascontiguousarray(cl[gids], dtype=np.float64))
        Ml.append(M); tile_off.append(tile_off[-1] + M)
    Mt = tile_off[-1]; n_tiles = len(all_tiles)
    def cat(lst, shp): return np.ascontiguousarray(np.concatenate(lst, axis=0), dtype=np.float64) if Mt else np.zeros(shp)
    def stack(lst): return np.ascontiguousarray(np.stack(lst, axis=0)) if n_tiles else np.zeros((0, NPX, 4))
    return {
        "scene": scene, "n_tiles": n_tiles, "Mt": int(Mt),
        "means2d": cat(m2l, (0, 2)), "conics": cat(col_, (0, 3)),
        "opac": cat(opl, (0,)), "colors": cat(cll, (0, 3)),
        "coeffs": np.ascontiguousarray(np.stack(cfl, axis=0), dtype=np.float64).reshape(n_tiles, rank, 10),
        "safe": np.asarray(safel, dtype=np.int32),
        "px": np.ascontiguousarray(px, dtype=np.float64),
        "tile_off": np.asarray(tile_off, dtype=np.int32),
        "M": np.asarray(Ml, dtype=np.int64),
        "rig_max": np.asarray(rigl), "leaves": np.asarray(leavesl), "R": R,
        "ref_teacher_f32": stack(rt_f32), "ref_student_f32": stack(rs_f32),
        "ref_teacher_f64": stack(rt_f64), "ref_student_f64": stack(rs_f64),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", default="/tmp/3dgs-renderer-benchmark/data")
    ap.add_argument("--so", default="/home/liaoyuanjun/cgt_f0_fused_cuda/src/f0_kernels.so")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--views", type=int, default=5)
    ap.add_argument("--tiles-per-view", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--rank", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    np.seterr(divide="ignore", invalid="ignore")
    torch.cuda.set_device(args.device)
    t0 = time.perf_counter()
    wl = build_scene_workload(args.scene, args.ckpt, args.data_root, args.views, args.tiles_per_view, args.seed, args.rank)
    t_build = (time.perf_counter() - t0) * 1e3
    if wl is None:
        open(os.path.join(args.out, f"cgt_{args.scene}.json"), "w").write(json.dumps({"scene": args.scene, "status": "NO_TILES"}))
        print(f"[{args.scene}] NO_TILES"); return
    n_tiles = wl["n_tiles"]; R_ = wl["R"]

    lib = ctypes.CDLL(args.so)
    P = ctypes.c_void_p; I = ctypes.c_int; S = ctypes.c_size_t; Fl = ctypes.c_float
    lib.f0_init.argtypes = [I]; lib.f0_init.restype = None
    lib.f0_malloc.argtypes = [S]; lib.f0_malloc.restype = P
    lib.f0_h2d.argtypes = [P, P, S]; lib.f0_d2h.argtypes = [P, P, S]; lib.f0_free.argtypes = [P]
    lib.f0_sync.argtypes = []; lib.f0_stream_sync.argtypes = []
    lib.f0_ev_create.argtypes = []; lib.f0_ev_create.restype = P
    lib.f0_ev_record.argtypes = [P]; lib.f0_ev_elapsed.argtypes = [P, P]; lib.f0_ev_elapsed.restype = Fl
    lib.f0_ev_destroy.argtypes = [P]
    LP = [P, P, P, P, P, P, P, I]; LS = [P, I, P, P, I]; LF = [P, P, P, P, P, P, I, P, P, P, I]
    lib.f0_teacher_f32.argtypes = LP; lib.f0_teacher_f64.argtypes = LP
    lib.f0_student_f32.argtypes = LS; lib.f0_student_f64.argtypes = LS
    lib.f0_fused_f32.argtypes = LF; lib.f0_fused_f64.argtypes = LF
    bufs = []
    def up(arr):
        a = np.ascontiguousarray(arr); n = a.nbytes
        d = lib.f0_malloc(S(n)); lib.f0_h2d(P(d), a.ctypes.data_as(P), S(n)); bufs.append(d); return int(d)
    def upcast(arr, dt): return up(np.ascontiguousarray(arr, dtype=dt))
    d_m2 = {64: upcast(wl["means2d"], np.float64), 32: upcast(wl["means2d"], np.float32)}
    d_co = {64: upcast(wl["conics"], np.float64), 32: upcast(wl["conics"], np.float32)}
    d_op = {64: upcast(wl["opac"], np.float64), 32: upcast(wl["opac"], np.float32)}
    d_cl = {64: upcast(wl["colors"], np.float64), 32: upcast(wl["colors"], np.float32)}
    d_px = {64: upcast(wl["px"], np.float64), 32: upcast(wl["px"], np.float32)}
    d_cf = {64: upcast(wl["coeffs"], np.float64), 32: upcast(wl["coeffs"], np.float32)}
    d_off = up(wl["tile_off"]); d_safe = up(wl["safe"])
    def out_buf(dt): return up(np.zeros((n_tiles, NPX, 4), dtype=dt))
    d_out = {64: out_buf(np.float64), 32: out_buf(np.float32)}
    def L_teacher(p, m2, co, op, cl, off, pxa, out, nt):
        (lib.f0_teacher_f32 if p == 32 else lib.f0_teacher_f64)(P(m2), P(co), P(op), P(cl), P(off), P(pxa), P(out), I(nt))
    def L_student(p, cf, pxa, out, nt):
        (lib.f0_student_f32 if p == 32 else lib.f0_student_f64)(P(cf), I(R_), P(pxa), P(out), I(nt))
    def L_fused(p, m2, co, op, cl, off, cf, sf, pxa, out, nt):
        (lib.f0_fused_f32 if p == 32 else lib.f0_fused_f64)(P(m2), P(co), P(op), P(cl), P(off), P(cf), I(R_), P(sf), P(pxa), P(out), I(nt))
    def d2h(dptr, dt):
        h = np.zeros((n_tiles, NPX, 4), dtype=dt); n = h.nbytes
        lib.f0_d2h(h.ctypes.data_as(P), P(dptr), S(n)); return h
    def time_kernel(launch, warmup=None, repeats=None):
        wu = args.warmup if warmup is None else warmup
        rp = args.repeats if repeats is None else repeats
        for _ in range(wu): launch()
        lib.f0_stream_sync()
        e0 = lib.f0_ev_create(); e1 = lib.f0_ev_create(); samples = []
        for _ in range(rp):
            lib.f0_ev_record(e0); launch(); lib.f0_ev_record(e1)
            samples.append(float(lib.f0_ev_elapsed(e0, e1)))
        lib.f0_ev_destroy(e0); lib.f0_ev_destroy(e1); lib.f0_stream_sync()
        return np.asarray(samples, dtype=np.float64)

    T_teacher = time_kernel(lambda: L_teacher(32, d_m2[32], d_co[32], d_op[32], d_cl[32], d_off, d_px[32], d_out[32], n_tiles))
    T_fused = time_kernel(lambda: L_fused(32, d_m2[32], d_co[32], d_op[32], d_cl[32], d_off, d_cf[32], d_safe, d_px[32], d_out[32], n_tiles))
    T_student = time_kernel(lambda: L_student(32, d_cf[32], d_px[32], d_out[32], n_tiles))

    # correctness f32: kernel vs ref_mix (where safe student else teacher)
    L_teacher(32, d_m2[32], d_co[32], d_op[32], d_cl[32], d_off, d_px[32], d_out[32], n_tiles); lib.f0_stream_sync(); out_t = d2h(d_out[32], np.float32)
    L_fused(32, d_m2[32], d_co[32], d_op[32], d_cl[32], d_off, d_cf[32], d_safe, d_px[32], d_out[32], n_tiles); lib.f0_stream_sync(); out_f = d2h(d_out[32], np.float32)
    ref_mix_f32 = np.where(wl["safe"][:, None, None] == 1, wl["ref_student_f32"], wl["ref_teacher_f32"])
    ref_mix_f64 = np.where(wl["safe"][:, None, None] == 1, wl["ref_student_f64"], wl["ref_teacher_f64"])
    kernel_max_abs = float(np.abs(out_f - ref_mix_f32).max())
    per_tile_kernel = np.array([np.abs(out_f[i] - ref_mix_f32[i]).max() for i in range(n_tiles)])
    n_kernel_fail = int((per_tile_kernel > CORRECT_THRESH).sum())
    # CGT-vs-canonical (fused vs teacher), f64 authoritative
    cgt_max_abs = float(np.abs(out_f.astype(np.float64) - wl["ref_teacher_f64"]).max())
    def psnr_t(a, b):
        mse = np.mean((a[..., :3] - b[..., :3]) ** 2)
        return 10 * np.log10(1.0 / mse) if mse > 0 else 99.0
    cgt_psnrs = np.array([psnr_t(out_f[i].astype(np.float64), wl["ref_teacher_f64"][i]) for i in range(n_tiles)])

    safe = wl["safe"]; Mt = wl["Mt"]
    weighted_safe = float((safe * wl["M"]).sum() / max(Mt, 1))
    result = {
        "scene": args.scene, "status": "OK", "rank": R, "n_tiles": n_tiles, "Mt": Mt,
        "M_median": int(np.median(wl["M"])), "M_mean": float(np.mean(wl["M"])),
        "T_build_ms": t_build,
        "T_teacher_all_ms": float(np.median(T_teacher)),
        "T_fused_mixed_ms": float(np.median(T_fused)),
        "T_student_all_ms": float(np.median(T_student)),
        "S_mixed": float(np.median(T_teacher) / np.median(T_fused)) if np.median(T_fused) > 0 else None,
        "raw_T_teacher": T_teacher.tolist(), "raw_T_fused": T_fused.tolist(),
        "certificate": {
            "safe_fraction": float(safe.mean()), "weighted_safe_fraction": weighted_safe,
            "fallback_fraction": float(1 - safe.mean()), "weighted_fallback_fraction": float(1 - weighted_safe),
            "rigorous_max_abs_median": float(np.median(wl["rig_max"])),
            "rigorous_max_abs_max": float(wl["rig_max"].max()),
            "rigorous_pass_rate": float((wl["rig_max"] <= GATE).mean()),
            "leaves_median": float(np.median(wl["leaves"])), "leaves_max": int(wl["leaves"].max()),
            "gate_max_abs": GATE,
        },
        "correctness": {
            "threshold_max_abs_RGB": CORRECT_THRESH,
            "kernel_max_abs": kernel_max_abs, "n_tiles_over_thresh": n_kernel_fail,
            "kernel_all_passed": bool(kernel_max_abs <= CORRECT_THRESH),
            "cgt_vs_canonical_max_abs": cgt_max_abs,
            "cgt_vs_canonical_psnr_median": float(np.median(cgt_psnrs)),
            "cgt_vs_canonical_psnr_min": float(cgt_psnrs.min()),
        },
        "device": args.device, "gsplat": gsplat.__version__, "torch": torch.__version__,
        "views": args.views, "tiles_per_view": args.tiles_per_view, "seed": args.seed,
    }
    outp = os.path.join(args.out, f"cgt_{args.scene}.json")
    open(outp, "w").write(json.dumps(result, indent=2))
    print(f"[{args.scene}] S_mixed={result['S_mixed']:.4f} safe={result['certificate']['safe_fraction']:.3f} "
          f"kernel_max_abs={kernel_max_abs:.2e} pass={result['correctness']['kernel_all_passed']} "
          f"cgt_vs_canon_max_abs={cgt_max_abs:.2e} T_build={t_build:.0f}ms", flush=True)

if __name__ == "__main__":
    main()