#!/usr/bin/env python3
"""G3 GPU local microbenchmark for CGT-Splat (MX A100).

Produces the FIRST real measured GPU latency evidence for gate G3 ("does fewer
components actually run faster?"), replacing the assumed gamma=2-4 in the G1/G2
Amdahl bound with a measured per-component cost ratio.

Method (honest, bounded):
  * Reuse G1's EXACT gsplat packed tile extraction to obtain real per-tile
    Gaussian intersection sets (M = tile_work) and 16x16 pixel grids.
  * Port the teacher (front-to-back alpha composite of M Gaussians) and the
    student (rank-R CGT Gaussian-transfer mixture eval) to torch GPU,
    numerically faithful to g1_compressibility.py / cgt_reference.py.
  * Time with CUDA events + warmup + median, two regimes:
      - per-tile: one tile per call (isolates launch/traversal overhead h)
      - batched : many tiles concatenated (amortizes launch -> intrinsic math)
  * gamma_measured(R) = t_teacher(M) / t_student(R)  (per-tile, same tile)
  * h_overhead   = per-tile dispatch overhead as a fraction of baseline teacher
  * S_measured   = work-weighted Amdahl S using measured gamma and measured h

SCOPE / HONESTY:
  * This times a torch REFERENCE implementation of the math, NOT a fused CGT
    CUDA kernel and NOT gsplat's fused rasterizer. gamma_measured is the
    intrinsic A100 math-cost ratio of student-vs-teacher for this tile work.
  * It does NOT implement the spatial tree / cost selector / adaptive control
    (plan sec.6). h_overhead here is a LOWER BOUND (torch dispatch only); a real
    renderer adds tree traversal on top.
  * No speedup is claimed; measured gamma and h are reported against the G2
    budget (h_max 26.6% at gamma=2) and the assumed gamma=2-4.
"""
from __future__ import annotations
import argparse, json, struct, time
from pathlib import Path
import numpy as np
import torch

TILE = 16
POWER_CLIP = -4.5
DEV = "cuda"

# ---------------- COLMAP readers (verbatim from g1_compressibility.py) ----------------
def read_cameras_bin(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; out = {}
        for _ in range(n):
            cid = struct.unpack("<i", f.read(4))[0]; mid = struct.unpack("<i", f.read(4))[0]
            w = struct.unpack("<Q", f.read(8))[0]; h = struct.unpack("<Q", f.read(8))[0]
            npar = {0:3,1:4,2:3,3:4,4:4,5:5,6:6,7:8}.get(mid,4)
            p = struct.unpack(f"<{npar}d", f.read(npar*8)); out[cid] = (mid,w,h,p)
    return out

def read_images_bin(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; out = []
        for _ in range(n):
            iid = struct.unpack("<i", f.read(4))[0]; qv = struct.unpack("<4d", f.read(32)); tv = struct.unpack("<3d", f.read(24)); cam = struct.unpack("<i", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00": break
                name += c
            np_ = struct.unpack("<Q", f.read(8))[0]; f.read(np_*24)
            out.append((iid, name.decode(), cam, qv, tv))
    return out

def q2R(q):
    x,y,z,w = q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

# ---------------- torch GPU ports (numerically faithful) ----------------
def tile_teacher_torch(px, py, means2d, conics, opac, colors):
    # px,py: (P,1); means2d:(M,2); conics:(M,3); opac:(M,); colors:(M,3)
    mu = means2d; co = conics; op = opac; col = colors
    dx = torch.cat([px[:, None, :], py[:, None, :]], dim=2) - mu[None, :, :]   # (P,M,2)
    a = co[:, 0]; b = co[:, 1]; c = co[:, 2]
    Ax = dx[:, :, 0]; Ay = dx[:, :, 1]
    power = -0.5 * (a[None, :] * Ax * Ax + 2 * b[None, :] * Ax * Ay + c[None, :] * Ay * Ay)
    clip = torch.clamp(power, max=0.0)
    alpha = torch.where(power > POWER_CLIP, op[None, :] * torch.exp(clip), torch.zeros_like(power))
    alpha = torch.clamp(alpha, 0.0, 0.999)
    one_minus = 1.0 - alpha
    Tprefix = torch.cumprod(one_minus, dim=1)          # (P,M)
    Tbefore = torch.cat([torch.ones_like(Tprefix[:, :1]), Tprefix[:, :-1]], dim=1)
    E = torch.einsum("pi,pi,ic->pc", Tbefore, alpha, col)
    T = Tprefix[:, -1:]
    return E, T

def expert_values_torch(p):
    t = torch.sigmoid(p[:, 6:7]); color = torch.sigmoid(p[:, 7:10])
    return torch.cat([(1 - t) * color, t], dim=1)      # (R,4) premult RGB,T

def predict_torch(p, q):
    # p:(R,10) q:(P,2)
    dx = q[:, 0:1] - p[None, :, 0]
    dy = q[:, 1:2] - p[None, :, 1]
    l00 = torch.exp(p[:, 2]); l10 = p[:, 3]; l11 = torch.exp(p[:, 4])
    z0 = dx * l00[None, :] + dy * l10[None, :]
    z1 = dy * l11[None, :]
    logits = p[None, :, 5] - 0.5 * (z0 * z0 + z1 * z1)
    weights = torch.softmax(logits, dim=1)
    expert = expert_values_torch(p)
    return weights @ expert                            # (P,4)

def predict_vjp_torch(p, q, cotangent):
    # forward with grads then vjp; returns pred, grad_p
    p_ = p.detach().clone().requires_grad_(True)
    pred = predict_torch(p_, q)
    (grad_p,) = torch.autograd.grad(pred, p_, grad_outputs=cotangent, retain_graph=False)
    return pred.detach(), grad_p.detach()

def init_params(R, P, rng):
    # representative valid params (values irrelevant to timing; shapes matter)
    p = torch.zeros(R, 10, device=DEV)
    p[:, 0] = torch.as_tensor(rng.uniform(-1, 1, (R,)), device=DEV, dtype=torch.float32)
    p[:, 1] = torch.as_tensor(rng.uniform(-1, 1, (R,)), device=DEV, dtype=torch.float32)
    p[:, 2] = p[:, 4] = np.log(np.sqrt(R) / 1.4)
    p[:, 5] = 0.0
    p[:, 6] = -1.0; p[:, 7:10] = 0.5
    return p

# ---------------- timing helpers ----------------
def bench(fn, iters=50, warmup=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return float(np.median(times)), float(np.percentile(times, 90))

# ---------------- large-P intrinsic (launch-free) math ratio ----------------
def bench_intrinsic(gtiles, ranks, rng, n_pixels=65536):
    """Launch-free intrinsic per-element cost: feed a large pixel batch through a
    SINGLE teacher call (M gaussians) and a SINGLE student call (R components).
    With n_pixels huge, CUDA launch overhead is amortized to ~0, so
    t ~ elements * per_element_cost. intrinsic_gamma(R) = t_teacher / t_student.
    Uses a representative tile (median M); params created ONCE (no H2D in loop)."""
    Meds = sorted(gtiles, key=lambda t: t["M"])
    rep = Meds[len(Meds) // 2]
    M = rep["M"]
    px_big = torch.as_tensor(rep["px"][:, :2], device=DEV)              # (256,2)
    reps = n_pixels // px_big.shape[0] + 1
    qbig = px_big.repeat(reps, 1)[:n_pixels].contiguous()               # (N,2)
    means2d = rep["means2d"]; conics = rep["conics"]; opac = rep["opac"]; colors = rep["colors"]
    px1 = qbig[:, 0:1]; py1 = qbig[:, 1:2]
    tt, _ = bench(lambda: tile_teacher_torch(px1, py1, means2d, conics, opac, colors), iters=30, warmup=10)
    out = dict(M_rep=int(M), n_pixels=int(n_pixels), teacher_ms=float(tt), per_rank={})
    for R in ranks:
        p = init_params(R, qbig.shape[0], rng)                          # ONCE
        ts, _ = bench(lambda: predict_torch(p, qbig), iters=30, warmup=10)
        out["per_rank"][str(R)] = dict(student_ms=float(ts),
                                       intrinsic_gamma=float(tt / max(ts, 1e-9)),
                                       teacher_per_gaussian_us=float(tt / max(M, 1) * 1e3),
                                       student_per_component_us=float(ts / max(R, 1) * 1e3))
    return out

# ---------------- student batching curve (fusion/headroom probe) ----------------
def batching_curve(gtiles, ranks, rng, tile_counts=(1, 2, 4, 8, 16, 32, 64, 128, 256)):
    """Measure per-tile student latency as B tiles are batched into ONE predict call
    (B*256 pixels, shared rank-R params). This probes whether amortizing launch over
    many cells recovers the intrinsic math headroom (the G4 fusion question).
    Also measures teacher per-tile latency at the same B for reference. Params created
    ONCE per (B,R); no H2D in timing loops."""
    out = {}
    for R in ranks:
        curve = []
        for B in tile_counts:
            if B > len(gtiles): break
            batch = gtiles[:B]
            p = init_params(R, batch[0]["px"].shape[0], rng)   # ONCE per (B,R)
            # student: concat pixels, one call
            qcat = torch.cat([t["px"] for t in batch], dim=0).contiguous()
            def student_call():
                predict_torch(p, qcat)
            ts, _ = bench(student_call, iters=30, warmup=10)
            # teacher reference: B separate teacher calls (per-tile baseline)
            def teacher_calls():
                for t in batch:
                    tile_teacher_torch(t["px"][:, 0:1], t["px"][:, 1:2],
                                       t["means2d"], t["conics"], t["opac"], t["colors"])
            tt, _ = bench(teacher_calls, iters=20, warmup=8)
            curve.append(dict(B=B,
                              student_batch_ms=float(ts),
                              student_per_tile_ms=float(ts / B),
                              teacher_pertile_ms=float(tt / B),
                              S_batch=float((tt / B) / max(ts / B, 1e-9))))
        out[str(R)] = curve
    return out

# ---------------- per-tile extraction (reuse G1 gsplat path) ----------------
def extract_tiles(scene, ckpt, data, view_idx, max_tiles, seed):
    import gsplat
    cams = read_cameras_bin(f"{data}/datasets/mipnerf360/{scene}/sparse/0/cameras.bin")
    imgs = read_images_bin(f"{data}/datasets/mipnerf360/{scene}/sparse/0/images.bin")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    means = ck["xyz"].to(DEV).float(); scales = torch.exp(ck["scaling"].to(DEV).float())
    quats = ck["rotation"].to(DEV).float(); opacities = torch.sigmoid(ck["opacity"].to(DEV).float()); shs = ck["shs"].to(DEV).float()
    deg = int(ck["active_sh_degree"]); rng = np.random.default_rng(seed)
    iid, name, cam, qv, tv = imgs[view_idx]; mid, W, H, par = cams[cam]; fx, fy, cx, cy = par
    R_w2c = q2R(qv); t_w2c = np.array(tv); t_c2w = -R_w2c.T @ t_w2c
    viewmat = torch.eye(4, dtype=torch.float32, device=DEV)[None]
    viewmat[0, :3, :3] = torch.as_tensor(R_w2c, dtype=torch.float32, device=DEV)
    viewmat[0, :3, 3] = torch.as_tensor(t_w2c, dtype=torch.float32, device=DEV)
    K = torch.as_tensor([[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]], dtype=torch.float32, device=DEV)
    campos = torch.as_tensor(t_c2w, dtype=torch.float32, device=DEV)[None]
    rc, ra, meta = gsplat.rasterization(means, quats, scales, opacities, shs, viewmat, K, width=W, height=H, sh_degree=deg, render_mode="RGB", packed=True, tile_size=TILE)
    dirs = torch.nn.functional.normalize(means - campos, dim=-1)
    colors = (gsplat.spherical_harmonics(deg, dirs, shs) + 0.5).clamp(0, 1).detach().cpu().numpy()[meta["gaussian_ids"].detach().cpu().numpy()]
    def flat2(x):
        x = meta[x].detach().cpu().numpy(); return x[0] if x.ndim == 3 else x
    means2d = flat2("means2d"); conics = flat2("conics"); opac = flat2("opacities")
    io = meta["isect_offsets"].detach().cpu().numpy().reshape(-1); fi = meta["flatten_ids"].detach().cpu().numpy()
    th, tw = int(meta["tile_height"]), int(meta["tile_width"]); total = len(fi)
    starts = io.astype(np.int64)
    def end_of(ti, tj):
        if tj + 1 < tw: return int(starts[ti * tw + tj + 1])
        if ti + 1 < th: return int(starts[(ti + 1) * tw + 0])
        return total
    tiles = []
    for ti in range(th):
        for tj in range(tw):
            s = int(starts[ti * tw + tj]); e = end_of(ti, tj)
            if e - s > 0: tiles.append((ti, tj, s, e))
    counts = np.array([e - s for (_, _, s, e) in tiles])
    order = np.argsort(counts); ranks_sel = np.empty(len(tiles), dtype=np.int64); ranks_sel[order] = np.arange(len(tiles))
    decile = np.minimum((ranks_sel * 10) // max(len(tiles), 1), 9)
    sel = []
    for dec in range(10):
        pool = np.where(decile == dec)[0]
        if len(pool) == 0: continue
        sel.append(rng.choice(pool, size=max(1, max_tiles // 10), replace=False))
    sel = np.concatenate(sel) if sel else np.array([], dtype=np.int64)
    out = []
    for idx in sel:
        ti, tj, s, e = tiles[idx]; gids = fi[s:e]
        if len(gids) < 1: continue
        ox = tj * TILE; oy = ti * TILE
        xs = ox + np.arange(TILE) + 0.5; ys = oy + np.arange(TILE) + 0.5
        gx, gy = np.meshgrid(xs, ys, indexing="xy")
        px = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
        cx0, cy0 = px.mean(0); px[:, 0] -= cx0; px[:, 1] -= cy0
        out.append(dict(px=px, gids=gids, means2d=means2d, conics=conics, opac=opac, colors=colors, M=int(e - s)))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--views", type=int, default=6); ap.add_argument("--max-tiles-per-view", type=int, default=120)
    ap.add_argument("--ranks", type=str, default="4,8,16"); ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--batched-tiles", type=int, default=64)
    args = ap.parse_args()
    ranks = [int(x) for x in args.ranks.split(",")]
    rng = np.random.default_rng(args.seed)
    view_idx = rng.choice(args.views, size=min(3, args.views), replace=False) if False else list(range(min(3, args.views)))
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    # gather tiles across views
    tiles = []
    for vi in view_idx:
        try:
            tiles += extract_tiles(args.scene, args.ckpt, args.data, vi, args.max_tiles_per_view, args.seed + vi)
        except Exception as ex:
            import traceback; print(f"[view {vi}] extract failed: {ex}", flush=True)
    if not tiles:
        print("NO_TILES", flush=True); return
    print(f"extracted {len(tiles)} tiles; M median={int(np.median([t['M'] for t in tiles]))}", flush=True)

    # move per-tile arrays to GPU once
    gtiles = []
    for t in tiles:
        px = torch.as_tensor(t["px"], device=DEV)
        g = t["gids"]
        gtiles.append(dict(
            px=px,
            means2d=torch.as_tensor(t["means2d"][g], device=DEV),
            conics=torch.as_tensor(t["conics"][g], device=DEV),
            opac=torch.as_tensor(t["opac"][g], device=DEV),
            colors=torch.as_tensor(t["colors"][g], device=DEV),
            M=t["M"],
        ))

    # ---- per-tile timing ----
    per_tile = {R: dict(teacher=[], student=[], student_fb=[]) for R in ranks}
    overhead = []
    for t in gtiles:
        px, py = t["px"][:, 0:1], t["px"][:, 1:2]
        # teacher
        tt, _ = bench(lambda: tile_teacher_torch(px, py, t["means2d"], t["conics"], t["opac"], t["colors"]), iters=30, warmup=10)
        per_tile["_t_teacher"] = tt
        # empty-call overhead floor (tiny 4x4 matmul as dispatch proxy)
        ov, _ = bench(lambda: torch.ones(4, 4, device=DEV) @ torch.ones(4, 4, device=DEV), iters=30, warmup=10)
        overhead.append(ov)
        for R in ranks:
            p = init_params(R, px.shape[0], rng)
            q = t["px"]
            cot = torch.ones(px.shape[0], 4, device=DEV)
            ts, _ = bench(lambda: predict_torch(p, q), iters=30, warmup=10)
            tf, _ = bench(lambda: predict_vjp_torch(p, q, cot), iters=20, warmup=8)
            per_tile[R]["teacher"].append(tt); per_tile[R]["student"].append(ts); per_tile[R]["student_fb"].append(tf)
        per_tile.setdefault("_M", []).append(t["M"])

    # ---- large-P intrinsic (launch-free) math ratio ----
    intrinsic = bench_intrinsic(gtiles, ranks, rng, n_pixels=65536)
    batching = batching_curve(gtiles, ranks, rng)
    batched = intrinsic  # key name kept for downstream; now holds launch-free intrinsic

    # ---- aggregate ----
    Ms = np.array(per_tile["_M"], dtype=np.float64)
    ov_med = float(np.median(overhead))
    result = dict(scene=args.scene, n_tiles=len(gtiles), M_median=float(np.median(Ms)),
                  M_min=int(Ms.min()), M_max=int(Ms.max()), dispatch_overhead_ms=ov_med,
                  per_rank={}, batched=batched, scope_note=(
                  "torch reference math on A100; NOT fused CGT kernel, NOT gsplat fused rasterizer; "
                  "h_overhead is a LOWER BOUND (torch dispatch only, no spatial tree)."))
    for R in ranks:
        tt = np.array(per_tile[R]["teacher"]); ts = np.array(per_tile[R]["student"]); tf = np.array(per_tile[R]["student_fb"])
        gamma_math = float(np.median(tt / np.maximum(ts, 1e-9)))
        gamma_math_p90 = float(np.percentile(tt / np.maximum(ts, 1e-9), 90))
        # work-weighted teacher time and student time
        w = Ms / Ms.sum()
        t_teacher_w = float(np.sum(w * tt)); t_student_w = float(np.sum(w * ts)); tfb_w = float(np.sum(w * tf))
        gamma_w = t_teacher_w / max(t_student_w, 1e-9)
        gamma_w_fb = t_teacher_w / max(tfb_w, 1e-9)
        # h_overhead: dispatch floor / teacher baseline (per-tile)
        h_over = ov_med / max(float(np.median(tt)), 1e-9)
        result["per_rank"][str(R)] = dict(
            teacher_ms_median=float(np.median(tt)), student_ms_median=float(np.median(ts)),
            student_fb_ms_median=float(np.median(tf)), gamma_math_median=gamma_math,
            gamma_math_p90=gamma_math_p90, gamma_work_weighted=gamma_w,
            gamma_work_weighted_fwd_bwd=gamma_w_fb, h_overhead_per_tile=h_over,
        )
    # measured Amdahl S using per-tile measured gamma and h, work-weighted
    for R in ranks:
        g = result["per_rank"][str(R)]["gamma_work_weighted"]
        h = result["per_rank"][str(R)]["h_overhead_per_tile"]
        fwd_ratio = 1.0 / max(g, 1e-9) + h
        result["per_rank"][str(R)]["S_measured_zero_idealh"] = 1.0 / (1.0 / max(g, 1e-9))
        result["per_rank"][str(R)]["S_measured_with_h"] = 1.0 / max(fwd_ratio, 1e-9)

    result["batching_curve"] = batching
    intr = result.get("batched", {})
    ipr = intr.get("per_rank", {})
    result["headline"] = {
        "intrinsic_gamma_R4": ipr.get("4", {}).get("intrinsic_gamma"),
        "intrinsic_gamma_R8": ipr.get("8", {}).get("intrinsic_gamma"),
        "intrinsic_gamma_R16": ipr.get("16", {}).get("intrinsic_gamma"),
        "per_tile_gamma_R4": result["per_rank"]["4"]["gamma_work_weighted"],
        "per_tile_S_with_h_R4": result["per_rank"]["4"]["S_measured_with_h"],
        "note": ("intrinsic_gamma = launch-free per-element math ratio (65536 px single call); "
                 "per_tile_gamma = launch-bound (256 px). A real per-tile renderer is launch-bound, "
                 "so per_tile_S is the operationally relevant ceiling; intrinsic_gamma is the math lower bound."),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    print("WROTE", args.out, flush=True)
    print(json.dumps({k: result[k] for k in ("scene", "n_tiles", "M_median", "dispatch_overhead_ms")}, indent=2), flush=True)
    bc = result.get("batching_curve", {})
    for R in ranks:
        print(f"BATCHING R={R}:", flush=True)
        for pt in bc.get(str(R), []):
            print(f"  B={pt['B']:>4d} student/tile={pt['student_per_tile_ms']:.5f}ms "
                  f"teacher/tile={pt['teacher_pertile_ms']:.5f}ms S_batch={pt['S_batch']:.2f}", flush=True)
    print("INTRINSIC(launch-free):", json.dumps(result.get("batched", {}), indent=2), flush=True)
    for R in ranks:
        pr = result["per_rank"][str(R)]
        print(f"R={R}: per_tile gamma={pr['gamma_work_weighted']:.3f} h_over={pr['h_overhead_per_tile']:.4f} "
              f"S_idealh={pr['S_measured_zero_idealh']:.2f} S_withh={pr['S_measured_with_h']:.3f}", flush=True)
    print("HEADLINE:", json.dumps(result["headline"], indent=2), flush=True)

if __name__ == "__main__":
    main()
