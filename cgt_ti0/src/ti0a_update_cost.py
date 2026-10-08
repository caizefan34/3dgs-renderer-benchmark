#!/usr/bin/env python
"""CGT-TI0-A: CGT representation update-cost oracle (room, topology-stable).

Measures the REAL cost of restoring a valid CGT representation after a canonical
3DGS optimizer step, on a real topology-stable room checkpoint. Determines whether
the ~3x fused-forward benefit survives representation update cost.

FROZEN upstream (NOT modified): CGT-F0 f0_kernels.so, g1 teacher/fit, g2 certificate.
This harness only loads the room checkpoint into a canonical GaussianModel, applies
ONE real reference_v1 Adam optimizer step (densification OFF at iter 15000), and times:
  T_canonical_forward (gsplat), T_update (full CGT rebuild), T_CGT_forward (fused kernel).

Update strategies:
  A. full rebuild/refit   -> MEASURED (only mechanically available path; CGT is stateless)
  B. affected tiles only  -> NOT_IMPLEMENTED (no stable tile->gaussian dependency map;
                             extract_tiles uses per-view stratified random sampling)
  C. incremental coeffs   -> NOT_IMPLEMENTED (fit_cgt is a cold 200-iter least-squares,
                             no warm-start; certificate must be recomputed regardless)
"""
from __future__ import annotations
import argparse, json, os, sys, time, math
import numpy as np

SCRIPTS = "/home/liaoyuanjun/cgt_realscene/final/SCRIPTS"
F0_SRC  = "/home/liaoyuanjun/cgt_f0_fused_cuda/src"
P13     = "/mnt/storage_pool/liaoyuanjun/cgt_paper13/src"
REFV1   = "/home/liaoyuanjun/3dgs-renderer-benchmark/baseline/reference_v1"
for p in (SCRIPTS, F0_SRC, P13, REFV1):
    sys.path.insert(0, p)

import ctypes
import torch
import torch.nn.functional as F
import gsplat
from g1_compressibility import read_cameras_bin, read_images_bin, q2R, tile_teacher, fit_cgt, TILE
from g2_rigorous_certificate import certify_tile, extract_tiles
from paper13_cgt import shared_px, R as CGT_R, GATE as CGT_GATE, CORRECT_THRESH, NPX
from gaussian_model import GaussianModel
from config import ReferenceV1Config

P = ctypes.c_void_p; I = ctypes.c_int; S = ctypes.c_size_t; Fl = ctypes.c_float


def gauss_window(size, sigma, channels, device):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2)); g = g / g.sum()
    w = g[:, None] * g[None, :]
    return w.expand(channels, 1, size, size).contiguous()


def ssim_mean(pred, gt, window, size=11):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    p = pred.permute(2, 0, 1)[None]; g = gt.permute(2, 0, 1)[None]
    pad = size // 2
    mu_p = F.conv2d(p, window, padding=pad, groups=3)
    mu_g = F.conv2d(g, window, padding=pad, groups=3)
    mu_p2, mu_g2, mu_pg = mu_p * mu_p, mu_g * mu_g, mu_p * mu_g
    sig_p = F.conv2d(p * p, window, padding=pad, groups=3) - mu_p2
    sig_g = F.conv2d(g * g, window, padding=pad, groups=3) - mu_g2
    sig_pg = F.conv2d(p * g, window, padding=pad, groups=3) - mu_pg
    s = ((2 * mu_pg + C1) * (2 * sig_pg + C2)) / ((mu_p2 + mu_g2 + C1) * (sig_p + sig_g + C2))
    return s.mean()


def build_cam(data_root, scene, view_idx):
    cams = read_cameras_bin(f"{data_root}/datasets/mipnerf360/{scene}/sparse/0/cameras.bin")
    imgs = read_images_bin(f"{data_root}/datasets/mipnerf360/{scene}/sparse/0/images.bin")
    iid, name, cam, qv, tv = imgs[view_idx]; mid, W, H, par = cams[cam]; fx, fy, cx, cy = par
    R_w2c = q2R(qv); t_w2c = np.array(tv); t_c2w = -R_w2c.T @ t_w2c
    viewmat = torch.eye(4, dtype=torch.float32, device="cuda")[None]
    viewmat[0, :3, :3] = torch.as_tensor(R_w2c, dtype=torch.float32, device="cuda")
    viewmat[0, :3, 3] = torch.as_tensor(t_w2c, dtype=torch.float32, device="cuda")
    K = torch.as_tensor([[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]], dtype=torch.float32, device="cuda")
    return dict(name=name, W=int(W), H=int(H), viewmat=viewmat, K=K)


def load_gt(data_root, scene, name, W, H):
    from PIL import Image
    p = f"{data_root}/datasets/mipnerf360/{scene}/images/{name}"
    im = Image.open(p).convert("RGB")
    if im.size != (W, H):
        im = im.resize((W, H))
    return torch.from_numpy(np.asarray(im)).float().cuda() / 255.0


def model_from_ckpt(ck, config):
    m = GaussianModel(max_sh_degree=int(ck["active_sh_degree"]))
    m._xyz = torch.nn.Parameter(ck["xyz"].cuda().float().clone())
    m._scaling = torch.nn.Parameter(ck["scaling"].cuda().float().clone())
    m._rotation = torch.nn.Parameter(ck["rotation"].cuda().float().clone())
    m._opacity = torch.nn.Parameter(ck["opacity"].cuda().float().clone())
    m._shs = torch.nn.Parameter(ck["shs"].cuda().float().clone())
    m.active_sh_degree = int(ck["active_sh_degree"])
    n = m._xyz.shape[0]
    for attr in ("max_radii2D", "xyz_gradient_accum", "denom"):
        if getattr(m, attr, None) is None:
            setattr(m, attr, torch.zeros(n, device="cuda"))
    m.spatial_lr_scale = 1.0
    m.training_setup({
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
    return m


def render(model, cam, sh_degree):
    rc, ra, meta = gsplat.rasterization(
        means=model.get_xyz, quats=model.get_rotation, scales=model.get_scaling,
        opacities=model.get_opacity, colors=model.get_features,
        viewmats=cam["viewmat"], Ks=cam["K"], width=cam["W"], height=cam["H"],
        sh_degree=sh_degree, render_mode="RGB", packed=True, tile_size=16, radius_clip=0.0, eps2d=0.1)
    return rc[0].clamp(0, 1)


def extract_all_tiles(scene, ckpt, data, view_idx):
    """Extract ALL tiles of one view (no decile sampling). Replicates extract_tiles exactly but iterates every tile."""
    cams = read_cameras_bin(f"{data}/datasets/mipnerf360/{scene}/sparse/0/cameras.bin")
    imgs = read_images_bin(f"{data}/datasets/mipnerf360/{scene}/sparse/0/images.bin")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    means = ck["xyz"].cuda().float(); scales = torch.exp(ck["scaling"].cuda().float())
    quats = ck["rotation"].cuda().float(); opacities = torch.sigmoid(ck["opacity"].cuda().float()); shs = ck["shs"].cuda().float()
    deg = int(ck["active_sh_degree"])
    iid, name, cam, qv, tv = imgs[view_idx]; mid, W, H, par = cams[cam]; fx, fy, cx, cy = par
    R_w2c = q2R(qv); t_w2c = np.array(tv); t_c2w = -R_w2c.T @ t_w2c
    viewmat = torch.eye(4, dtype=torch.float32, device="cuda")[None]
    viewmat[0, :3, :3] = torch.as_tensor(R_w2c, dtype=torch.float32, device="cuda")
    viewmat[0, :3, 3] = torch.as_tensor(t_w2c, dtype=torch.float32, device="cuda")
    K = torch.as_tensor([[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]], dtype=torch.float32, device="cuda")
    campos = torch.as_tensor(t_c2w, dtype=torch.float32, device="cuda")[None]
    rc, ra, meta = gsplat.rasterization(means, quats, scales, opacities, shs, viewmat, K, width=W, height=H,
                                        sh_degree=deg, render_mode="RGB", packed=True, tile_size=TILE)
    dirs = torch.nn.functional.normalize(means - campos, dim=-1)
    colors = (gsplat.spherical_harmonics(deg, dirs, shs) + 0.5).clamp(0, 1).detach().cpu().numpy()[meta["gaussian_ids"].detach().cpu().numpy()]
    def flat2(x):
        x = meta[x].detach().cpu().numpy(); return x[0] if x.ndim == 3 else x
    means2d = flat2("means2d"); conics = flat2("conics"); opac = flat2("opacities")
    io = meta["isect_offsets"].detach().cpu().numpy().reshape(-1); fi = meta["flatten_ids"].detach().cpu().numpy()
    th, tw = int(meta["tile_height"]), int(meta["tile_width"]); total = len(fi)
    starts = io.astype(np.int64)
    def end_of(ti, tj):
        if tj + 1 < tw: return int(starts[ti*tw + tj + 1])
        if ti + 1 < th: return int(starts[(ti + 1)*tw + 0])
        return total
    out = []
    for ti in range(th):
        for tj in range(tw):
            s = int(starts[ti*tw + tj]); e = end_of(ti, tj)
            if e - s <= 0: continue
            gids = fi[s:e]
            if len(gids) < 1: continue
            ox = tj*TILE; oy = ti*TILE
            xs = ox + np.arange(TILE) + 0.5; ys = oy + np.arange(TILE) + 0.5
            gx, gy = np.meshgrid(xs, ys, indexing="xy")
            px = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float64)
            cx0, cy0 = px.mean(0); px[:, 0] -= cx0; px[:, 1] -= cy0
            out.append(dict(px=px, gids=gids, means2d=means2d, conics=conics, opac=opac, colors=colors, M=int(e - s)))
    return out
def build_workload_from_tiles(tiles, px, R, seed):
    m2l, col_, opl, cll, cfl, Ml = [], [], [], [], [], []
    safel, tile_off = [], [0]
    for t in tiles:
        gids = t["gids"]; m2 = t["means2d"]; con = t["conics"]; op = t["opac"]; cl = t["colors"]
        M = int(gids.shape[0])
        E, T = tile_teacher(px[:, 0:1], px[:, 1:2], gids, m2, con, op, cl)
        y = np.concatenate([E, T], axis=1); tr = np.arange(0, y.shape[0], 2); yt = y[tr]
        qt = np.column_stack([px[tr, 0], px[tr, 1]]).astype(np.float64)
        p = fit_cgt(qt, yt, R, seed, 200)
        res = certify_tile(t, p, 1.0, 10, np.random.default_rng(seed + 7), True)
        safel.append(int(res["passed"]))
        cfl.append(np.ascontiguousarray(p, dtype=np.float64).reshape(R, 10))
        m2l.append(np.ascontiguousarray(m2[gids], dtype=np.float64))
        col_.append(np.ascontiguousarray(con[gids], dtype=np.float64))
        opl.append(np.ascontiguousarray(op[gids], dtype=np.float64))
        cll.append(np.ascontiguousarray(cl[gids], dtype=np.float64))
        Ml.append(M); tile_off.append(tile_off[-1] + M)
    Mt = tile_off[-1]; n_tiles = len(tiles)
    def cat(lst, shp): return np.ascontiguousarray(np.concatenate(lst, axis=0), dtype=np.float64) if Mt else np.zeros(shp)
    return {"n_tiles": n_tiles, "Mt": int(Mt), "means2d": cat(m2l, (0, 2)), "conics": cat(col_, (0, 3)),
            "opac": cat(opl, (0,)), "colors": cat(cll, (0, 3)),
            "coeffs": np.ascontiguousarray(np.stack(cfl, axis=0), dtype=np.float64).reshape(n_tiles, R, 10),
            "safe": np.asarray(safel, dtype=np.int32), "px": np.ascontiguousarray(px, dtype=np.float64),
            "tile_off": np.asarray(tile_off, dtype=np.int32), "M": np.asarray(Ml, dtype=np.int64), "R": R}


def time_fused_kernel(wl, so_path, repeats, warmup):
    n_tiles = wl["n_tiles"]; R_ = wl["R"]
    lib = ctypes.CDLL(so_path)
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
    def L_fused(p, m2, co, op, cl, off, cf, sf, pxa, out, nt):
        (lib.f0_fused_f32 if p == 32 else lib.f0_fused_f64)(P(m2), P(co), P(op), P(cl), P(off), P(cf), I(R_), P(sf), P(pxa), P(out), I(nt))
    def time_kernel(launch, wu, rp):
        for _ in range(wu): launch()
        lib.f0_stream_sync()
        e0 = lib.f0_ev_create(); e1 = lib.f0_ev_create(); samples = []
        for _ in range(rp):
            lib.f0_ev_record(e0); launch(); lib.f0_ev_record(e1)
            samples.append(float(lib.f0_ev_elapsed(e0, e1)))
        lib.f0_ev_destroy(e0); lib.f0_ev_destroy(e1); lib.f0_stream_sync()
        return np.asarray(samples, dtype=np.float64)
    return float(np.median(time_kernel(lambda: L_fused(32, d_m2[32], d_co[32], d_op[32], d_cl[32], d_off, d_cf[32], d_safe, d_px[32], d_out[32], n_tiles), warmup, repeats)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="room")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", default="/tmp/3dgs-renderer-benchmark/data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--so", default="/home/liaoyuanjun/cgt_f0_fused_cuda/src/f0_kernels.so")
    ap.add_argument("--tmp-ckpt", required=True)
    ap.add_argument("--device", type=int, default=1)
    ap.add_argument("--view-idx", type=int, default=0)
    ap.add_argument("--iter", type=int, default=15000)
    ap.add_argument("--sample-tiles", type=int, default=256)
    ap.add_argument("--full-build", action="store_true")
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.cuda.set_device(args.device)
    np.seterr(divide="ignore", invalid="ignore")
    config = ReferenceV1Config()
    result = {"stage": "CGT-TI0-A", "scene": args.scene, "iter": args.iter, "view_idx": args.view_idx,
              "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "device": args.device, "gsplat": gsplat.__version__, "torch": torch.__version__,
              "frozen": {"R": CGT_R, "gate_max_abs": CGT_GATE, "correct_thresh": CORRECT_THRESH, "tile": TILE}}

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    result["N_gaussians"] = int(ck["xyz"].shape[0])
    result["sh_degree"] = int(ck["active_sh_degree"])
    model = model_from_ckpt(ck, config)
    cam = build_cam(args.data_root, args.scene, args.view_idx)
    cam_name = cam["name"]
    gt = load_gt(args.data_root, args.scene, cam_name, cam["W"], cam["H"])
    result["view"] = {"name": cam_name, "W": cam["W"], "H": cam["H"]}
    window = gauss_window(11, 1.5, 3, "cuda")
    model.update_learning_rate(args.iter)
    img = render(model, cam, model.active_sh_degree)
    l1 = F.l1_loss(img, gt); dssim = 1.0 - ssim_mean(img, gt, window)
    loss = 0.8 * l1 + 0.2 * dssim
    model.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    t0 = time.perf_counter(); model.optimizer.step(); t1 = time.perf_counter()
    T_optimizer_ms = (t1 - t0) * 1e3
    result["optimizer"] = {"loss": float(loss.item()), "l1": float(l1.item()), "dssim": float(dssim),
                           "T_optimizer_step_ms": T_optimizer_ms,
                           "note": "one real reference_v1 Adam step; densification OFF (iter>=densify_until_iter=15000) => topology-stable"}
    with torch.no_grad():
        xyz0 = ck["xyz"].cuda().float(); scl0 = torch.exp(ck["scaling"].cuda().float())
        rot0 = ck["rotation"].cuda().float(); opa0 = torch.sigmoid(ck["opacity"].cuda().float()); shs0 = ck["shs"].cuda().float()
        rel_xyz = (model.get_xyz - xyz0).norm(dim=-1) / (xyz0.norm(dim=-1) + 1e-12)
        d_scl = (model.get_scaling - scl0).norm(dim=-1)
        d_rot = (model.get_rotation - rot0).norm(dim=-1)
        d_opa = (model.get_opacity - opa0).norm(dim=-1)
        d_shs = (model.get_features - shs0).norm(dim=-1)
    result["dirty_work"] = {
        "note": "CGT is STATELESS across optimizer steps (extract_tiles->fit_cgt->certify refit from scratch every frame). Reusability=0; all tiles rebuilt every step.",
        "n_gaussians": int(ck["xyz"].shape[0]),
        "rel_xyz_change_median": float(rel_xyz.median()), "rel_xyz_change_p90": float(rel_xyz.quantile(0.9)),
        "frac_xyz_changed_1e-4": float((rel_xyz > 1e-4).float().mean()),
        "frac_xyz_changed_1e-3": float((rel_xyz > 1e-3).float().mean()),
        "frac_xyz_changed_1e-2": float((rel_xyz > 1e-2).float().mean()),
        "d_scaling_median": float(d_scl.median()), "d_rotation_median": float(d_rot.median()),
        "d_opacity_median": float(d_opa.median()), "d_shs_median": float(d_shs.median()),
        "tiles_rebuilt_fraction": 1.0, "certificate_invalidated_fraction": 1.0,
        "threshold_note": "relative per-gaussian parameter-change thresholds (1e-4/1e-3/1e-2). CGT-validity output threshold (GATE_MAX_ABS=0.02) not mechanically mapped to param delta without render Jacobian; reusability is 0% regardless (stateless).",
    }

    for _ in range(args.warmup): _ = render(model, cam, model.active_sh_degree)
    torch.cuda.synchronize()
    t_can = []
    for _ in range(args.repeats):
        t0 = time.perf_counter(); _ = render(model, cam, model.active_sh_degree); torch.cuda.synchronize()
        t_can.append((time.perf_counter() - t0) * 1e3)
    T_canon = float(np.median(t_can))
    result["T_canonical_forward_ms"] = T_canon
    result["T_canonical_forward_raw"] = t_can

    ck_new = {"xyz": model.get_xyz.detach().cpu(), "scaling": model._scaling.detach().cpu(),
              "rotation": model._rotation.detach().cpu(), "opacity": model._opacity.detach().cpu(),
              "shs": model.get_features.detach().cpu(), "active_sh_degree": model.active_sh_degree}
    torch.save(ck_new, args.tmp_ckpt)

    px = shared_px()
    t0 = time.perf_counter()
    sample_tiles = extract_tiles(args.scene, args.tmp_ckpt, args.data_root, args.view_idx, args.sample_tiles, args.iter + 1)
    T_extract_sample_ms = (time.perf_counter() - t0) * 1e3
    if not sample_tiles:
        result["status"] = "NO_TILES"; open(os.path.join(args.out, "TI0A_UPDATE_COST.json"), "w").write(json.dumps(result, indent=2)); print("NO_TILES"); return
    per_tile_ms = []
    for t in sample_tiles[:min(len(sample_tiles), 64)]:
        gids = t["gids"]; m2 = t["means2d"]; con = t["conics"]; op = t["opac"]; cl = t["colors"]
        E, T = tile_teacher(px[:, 0:1], px[:, 1:2], gids, m2, con, op, cl)
        y = np.concatenate([E, T], axis=1); tr = np.arange(0, y.shape[0], 2); yt = y[tr]
        qt = np.column_stack([px[tr, 0], px[tr, 1]]).astype(np.float64)
        for _ in range(3):
            ts = time.perf_counter()
            p = fit_cgt(qt, yt, CGT_R, args.iter, 200)
            certify_tile(t, p, 1.0, 10, np.random.default_rng(args.iter), True)
            per_tile_ms.append((time.perf_counter() - ts) * 1e3)
    per_tile_ms = np.asarray(per_tile_ms)
    t_per_tile_median = float(np.median(per_tile_ms))
    th = math.ceil(cam["H"] / TILE); tw = math.ceil(cam["W"] / TILE); n_tiles_full = th * tw
    result["update"] = {
        "strategy_A_full_rebuild": "MEASURED", "strategy_B_affected_tiles": "NOT_IMPLEMENTED",
        "strategy_C_incremental_coeffs": "NOT_IMPLEMENTED",
        "T_extract_sample_ms": T_extract_sample_ms,
        "per_tile_fit_certify_ms_median": t_per_tile_median,
        "per_tile_fit_certify_ms_p90": float(np.percentile(per_tile_ms, 90)),
        "n_tiles_sample": len(sample_tiles), "n_tiles_full_frame": int(n_tiles_full),
        "T_update_full_extrapolated_ms": T_extract_sample_ms + n_tiles_full * t_per_tile_median,
    }

    full_tiles = None
    if args.full_build:
        t0 = time.perf_counter()
        full_tiles = extract_all_tiles(args.scene, args.tmp_ckpt, args.data_root, args.view_idx)
        T_extract_full_ms = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        for t in full_tiles:
            gids = t["gids"]; m2 = t["means2d"]; con = t["conics"]; op = t["opac"]; cl = t["colors"]
            E, T = tile_teacher(px[:, 0:1], px[:, 1:2], gids, m2, con, op, cl)
            y = np.concatenate([E, T], axis=1); tr = np.arange(0, y.shape[0], 2); yt = y[tr]
            qt = np.column_stack([px[tr, 0], px[tr, 1]]).astype(np.float64)
            p = fit_cgt(qt, yt, CGT_R, args.iter, 200)
            certify_tile(t, p, 1.0, 10, np.random.default_rng(args.iter), True)
        T_fit_certify_full_ms = (time.perf_counter() - t0) * 1e3
        result["update"]["T_extract_full_ms"] = T_extract_full_ms
        result["update"]["n_tiles_full_extracted"] = int(len(full_tiles))
        result["update"]["T_fit_certify_full_ms"] = T_fit_certify_full_ms
        result["update"]["T_update_full_measured_ms"] = T_extract_full_ms + T_fit_certify_full_ms

    T_update = result["update"].get("T_update_full_measured_ms", result["update"]["T_update_full_extrapolated_ms"])
    result["update"]["T_update_used_ms"] = T_update

    src_tiles = full_tiles if args.full_build else sample_tiles
    wl = build_workload_from_tiles(src_tiles, px, CGT_R, args.iter)
    T_fused = time_fused_kernel(wl, args.so, args.repeats, args.warmup)
    result["T_CGT_forward_ms"] = T_fused
    result["T_CGT_forward_n_tiles"] = wl["n_tiles"]

    T_eff = T_update + T_fused
    S = T_canon / T_eff if T_eff > 0 else None
    result["ratios"] = {"R_update": (T_update / T_canon) if T_canon > 0 else None,
                        "T_effective_CGT_ms": T_eff, "S_effective_forward": S}
    if S is None or S <= 1.0: v = "DROP_UPDATE_COST"
    elif S < 1.15: v = "MARGINAL_UPDATE_HEADROOM"
    else: v = "PROCEED_TI0_B"
    result["verdict_TI0_A"] = v
    open(os.path.join(args.out, "TI0A_UPDATE_COST.json"), "w").write(json.dumps(result, indent=2))
    open(os.path.join(args.out, "TI0A_DIRTY_WORK.json"), "w").write(json.dumps(result["dirty_work"], indent=2))
    print("[TI0-A] T_canon=%.3fms T_update=%.1fms T_fused=%.3fms R_update=%.1f S_eff=%.3e verdict=%s" % (T_canon, T_update, T_fused, result["ratios"]["R_update"], S, v))


if __name__ == "__main__":
    main()