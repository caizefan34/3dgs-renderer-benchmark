#!/usr/bin/env python3
"""Real-scene bounded-event opportunity probe (DSH, MX idle GPU).

Faithful to the frozen BES0 scene_adapter. Two opportunity structures:

 (1) per-OCCURRENCE R/D routing (BES0 literal mechanism, == build_bounds):
     for each (gaussian, tile) occurrence over the tile's ts x ts pixels:
       hi = max alpha (nearest tile pixel to peak); lo = min alpha (farthest
       tile corner); is_R = (lo>0) & (hi/lo <= gamma).
     rho_R(ts) reported for a tile-size sweep. Also the GLOBAL per-gaussian
     bound (min over whole bbox, ~0 for every gaussian). gamma sweep included.
     Per-occurrence expansion is chunked + span-capped to bound memory.

 (2) per-pixel early-termination (where 3DGS speedup actually comes from):
     strided pixel sample, front-to-back cumulative transmittance T, count
     occurrences after T < eps. skipped_fraction -> speedup = 1/(1-sf).

An OPPORTUNITY SCREEN, not a CUDA speedup measurement.
"""
import sys, os, json, time, math, argparse

PACKAGE_SCRIPTS = os.environ["BES0_PACKAGE_ROOT"] + "/scripts"
sys.path.insert(0, PACKAGE_SCRIPTS)
import mx_guard
cfg = mx_guard.assert_mx_host()
import numpy as np
import torch
import scene_adapter as sa

GAMMA = 2.0
EPS = sa.ALPHA_CUTOFF
MAX_SPAN = 8
CHUNK = 40000


def patient_idle_wait(cfg, gpu_uuid, max_wait_min=30):
    deadline = time.time() + max_wait_min * 60
    last = None
    while time.time() < deadline:
        try:
            return mx_guard.wait_idle_window(cfg, gpu_uuid)
        except mx_guard.GuardError as exc:
            last = str(exc); time.sleep(10)
    raise mx_guard.GuardError("GPU_NEVER_IDLE: " + str(last))


def resolve_gpu_uuid(cfg, prefer_index):
    for x in mx_guard.checked_inventory(cfg):
        if x["index"] == prefer_index and x["index"] in cfg["allowed_gpu_indices"]:
            return x["uuid"]
    raise mx_guard.GuardError("no allowed idle-capable GPU at index %d" % prefer_index)


def per_occurrence_rd(mu, conic, depth, radii, op, W, H, gamma=2.0, ts=16):
    """Faithful BES0 per-OCCURRENCE R/D for a given tile size ts (== build_bounds)."""
    vis = (depth > 0.05) & (radii > 0)
    gidx = np.nonzero(vis)[0]
    if gidx.size == 0:
        return dict(n_visible=0, n_occ=0, n_R_occ=0, rho_R_occ=0.0, l_zero_frac_occ=1.0)
    m = mu[gidx]; cc = conic[gidx]; rr = radii[gidx].astype(np.float64); opv = op[gidx]
    a = cc[:, 0].astype(np.float64); b = cc[:, 1].astype(np.float64); c = cc[:, 2].astype(np.float64)
    cx = m[:, 0].astype(np.float64); cy = m[:, 1].astype(np.float64)
    r = rr; cap = sa.ALPHA_CAP; cut = sa.ALPHA_CUTOFF
    tiles_x = (W + ts - 1) // ts; tiles_y = (H + ts - 1) // ts
    r_cap = MAX_SPAN * ts
    r_span = np.minimum(r, r_cap)
    tx0 = np.clip(np.floor((cx - r_span) / ts).astype(np.int64), 0, tiles_x - 1)
    tx1 = np.clip(np.floor((cx + r_span) / ts).astype(np.int64), 0, tiles_x - 1)
    tx1 = np.minimum(tx1, tx0 + MAX_SPAN - 1)
    ty0 = np.clip(np.floor((cy - r_span) / ts).astype(np.int64), 0, tiles_y - 1)
    ty1 = np.clip(np.floor((cy + r_span) / ts).astype(np.int64), 0, tiles_y - 1)
    ty1 = np.minimum(ty1, ty0 + MAX_SPAN - 1)
    span_x = tx1 - tx0 + 1; span_y = ty1 - ty0 + 1
    n_occ = span_x * span_y
    gammas = (2.0, 5.0, 10.0, 30.0, 100.0)
    sweep_num = {g: 0 for g in gammas}
    n_R_occ = 0; total_occ = 0; l_zero_num = 0
    n = gidx.size
    for cs in range(0, n, CHUNK):
        ce = min(n, cs + CHUNK)
        a_c = a[cs:ce]; b_c = b[cs:ce]; c_c = c[cs:ce]
        cx_c = cx[cs:ce]; cy_c = cy[cs:ce]; op_c = opv[cs:ce]
        sx = span_x[cs:ce]; sy = span_y[cs:ce]
        nx0 = tx0[cs:ce]; ny0 = ty0[cs:ce]; noc = n_occ[cs:ce]
        tot = int(noc.sum())
        if tot == 0:
            continue
        gr = np.repeat(np.arange(ce - cs), noc)
        o = np.arange(tot, dtype=np.int64)
        sxr = np.repeat(sx, noc)
        ox = o % sxr; oy = o // sxr
        tx = np.repeat(nx0, noc) + ox
        ty = np.repeat(ny0, noc) + oy
        px0 = tx * ts; px1 = np.minimum(W, px0 + ts)
        py0 = ty * ts; py1 = np.minimum(H, py0 + ts)
        ag = a_c[gr]; bg = b_c[gr]; cg = c_c[gr]
        cgx = cx_c[gr]; cgy = cy_c[gr]; opg = op_c[gr]
        nx = np.clip(np.rint(cgx).astype(np.int64), px0, px1 - 1)
        ny = np.clip(np.rint(cgy).astype(np.int64), py0, py1 - 1)
        dxh = nx - cgx; dyh = ny - cgy
        q_hi = ag*dxh*dxh + 2*bg*dxh*dyh + cg*dyh*dyh
        def qcorner(pxx, pyy):
            dx = pxx - cgx; dy = pyy - cgy
            return ag*dx*dx + 2*bg*dx*dy + cg*dy*dy
        q_lo = np.maximum(np.maximum(qcorner(px0, py0), qcorner(px0, py1 - 1)),
                          np.maximum(qcorner(px1 - 1, py0), qcorner(px1 - 1, py1 - 1)))
        with np.errstate(over='ignore', invalid='ignore'):
            lo = opg * np.exp(-0.5 * q_lo)
            hi = opg * np.exp(-0.5 * q_hi)
        lo = np.where(lo < cut, 0.0, lo)
        lo = np.clip(lo, 0.0, cap); hi = np.clip(hi, 0.0, cap)
        pos = lo > 0.0
        ratio = hi / np.maximum(lo, 1e-300)
        n_R_occ += int((pos & (ratio <= gamma)).sum())
        total_occ += tot
        l_zero_num += int((~pos).sum())
        for g in gammas:
            sweep_num[g] += int((pos & (ratio <= g)).sum())
    rho_R_occ = n_R_occ / total_occ if total_occ else 0.0
    l_zero_occ = l_zero_num / total_occ if total_occ else 1.0
    sweep = {str(g): (sweep_num[g] / total_occ if total_occ else 0.0) for g in gammas}
    return dict(n_visible=int(n), n_occ=total_occ, n_R_occ=n_R_occ,
                rho_R_occ=rho_R_occ, l_zero_frac_occ=l_zero_occ, gamma_sweep=sweep)


def per_pixel_saturation(mu, conic, depth, radii, op, W, H, target_pixels=2500, ts=16):
    vis = (depth > 0.05) & (radii > 0)
    gidx = np.nonzero(vis)[0]
    if gidx.size == 0:
        return dict(n_pixels=0, total_pairs=0, skipped_pairs=0, skipped_frac=0.0,
                    speedup_bound=1.0, mean_gaussians_per_pixel=0.0, mean_sat_index=0.0)
    m = mu[gidx]; cc = conic[gidx]; rr = radii[gidx].astype(np.float64)
    d = depth[gidx]; opv = op[gidx]
    tiles_x = (W + ts - 1) // ts; tiles_y = (H + ts - 1) // ts
    tx = np.clip(np.floor(m[:, 0] / ts).astype(np.int64), 0, tiles_x - 1)
    ty = np.clip(np.floor(m[:, 1] / ts).astype(np.int64), 0, tiles_y - 1)
    key = ty * tiles_x + tx
    order = np.argsort(key, kind="stable")
    keys_sorted = key[order]
    uniq, ustart = np.unique(keys_sorted, return_index=True)
    slist = list(ustart) + [len(order)]
    krange = {int(uniq[i]): (slist[i], slist[i + 1]) for i in range(len(uniq))}
    ring = 2
    stride = max(8, int(math.sqrt(W * H / max(1, target_pixels))))
    xs = np.arange(0, W, stride); ys = np.arange(0, H, stride)
    total_pairs = 0; skipped = 0; npix = 0; sat_idx_sum = 0.0; gpp_sum = 0.0
    for py in ys:
        cty = py // ts
        for px in xs:
            ctx = px // ts
            cand = []
            for dy in range(-ring, ring + 1):
                yy = cty + dy
                if yy < 0 or yy >= tiles_y:
                    continue
                base = yy * tiles_x
                for dx in range(-ring, ring + 1):
                    xx = ctx + dx
                    if xx < 0 or xx >= tiles_x:
                        continue
                    rng = krange.get(base + xx)
                    if rng is not None:
                        cand.append(order[rng[0]:rng[1]])
            if not cand:
                continue
            cand = np.unique(np.concatenate(cand))
            dxp = px - m[cand, 0]; dyp = py - m[cand, 1]
            inside = (np.abs(dxp) <= rr[cand]) & (np.abs(dyp) <= rr[cand])
            sel = cand[inside]
            if sel.size == 0:
                continue
            aa = cc[sel]; dxl = dxp[inside]; dyl = dyp[inside]
            q = aa[:, 0]*dxl*dxl + 2*aa[:, 1]*dxl*dyl + aa[:, 2]*dyl*dyl
            with np.errstate(over='ignore', invalid='ignore'):
                alpha = opv[sel] * np.exp(-0.5 * q)
            alpha = np.clip(alpha, 0.0, sa.ALPHA_CAP)
            alpha = np.where(alpha >= sa.ALPHA_CUTOFF, alpha, 0.0)
            o = np.argsort(d[sel], kind="stable")
            al = alpha[o]
            if al.size:
                Tbefore = np.concatenate([[1.0], np.cumprod(1.0 - al)[:-1]])
                below = np.nonzero(Tbefore < EPS)[0]
                sat = int(below[0]) if below.size else int(al.size)
            else:
                sat = 0
            total_pairs += int(al.size)
            skipped += int(al.size) - sat
            sat_idx_sum += sat; gpp_sum += al.size; npix += 1
    sf = skipped / total_pairs if total_pairs else 0.0
    speedup_bound = 1.0 / (1.0 - sf) if sf < 1.0 else float("inf")
    return dict(n_pixels=npix, total_pairs=int(total_pairs), skipped_pairs=int(skipped),
                skipped_frac=sf, speedup_bound=speedup_bound,
                mean_gaussians_per_pixel=(gpp_sum / npix if npix else 0.0),
                mean_sat_index=(sat_idx_sum / npix if npix else 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--gpu-index", type=int, default=6)
    ap.add_argument("--target-pixels", type=int, default=2500)
    ap.add_argument("--max-cams", type=int, default=0)
    ap.add_argument("--tile-sizes", default="16,8,4,2")
    args = ap.parse_args()
    tile_sizes = [int(x) for x in args.tile_sizes.split(",") if x.strip()]

    gpu_uuid = resolve_gpu_uuid(cfg, args.gpu_index)
    patient_idle_wait(cfg, gpu_uuid)
    print("[guard] idle window OK on %s" % gpu_uuid, flush=True)
    torch.set_num_threads(1); torch.cuda.set_device(0)
    device = "cuda"
    manifest = json.load(open(args.manifest))
    rec = manifest["scenes"][args.scene]
    dev_ids = list(rec["dev_camera_ids"])
    if args.max_cams and args.max_cams > 0:
        dev_ids = dev_ids[:args.max_cams]

    out = {"schema": "bes_opp_probe.v4", "scene": args.scene, "gpu_uuid": gpu_uuid,
           "gamma": GAMMA, "eps": EPS, "tile_sizes": tile_sizes,
           "checkpoint_sha256": rec.get("checkpoint_sha256"),
           "checkpoint_num_points": rec.get("checkpoint_num_points"),
           "cameras": {}, "agg": {}}

    t_load = time.time()
    scene = sa.load_scene(rec, device)
    torch.cuda.synchronize()
    out["load_seconds"] = time.time() - t_load
    out["num_points"] = int(scene.num_points)
    print("[load] %s N=%d in %.1fs" % (args.scene, scene.num_points, out["load_seconds"]), flush=True)
    op_all = sa.opacity_activation(scene.opacity).reshape(-1).detach().cpu().numpy()

    sum_vis = sum_occ = sum_R_occ = 0
    sum_pairs = sum_skipped = 0
    ts_R = {ts: 0 for ts in tile_sizes}
    ts_occ = {ts: 0 for ts in tile_sizes}
    sweep_acc = {g: 0.0 for g in ("2.0", "5.0", "10.0", "30.0", "100.0")}
    for cid in dev_ids:
        cam = scene.cameras[cid]
        t1 = time.time()
        mu2d, conic, depth, radii = sa.project_gaussians_2d(scene, cam)
        mu = mu2d.detach().cpu().numpy(); cc = conic.detach().cpu().numpy()
        dd = depth.detach().cpu().numpy(); rr = radii.detach().cpu().numpy()
        t_proj = time.time() - t1
        rd_by_ts = {}
        for ts in tile_sizes:
            rd_by_ts[ts] = per_occurrence_rd(mu, cc, dd, rr, op_all, cam.width, cam.height, GAMMA, ts)
        pp = per_pixel_saturation(mu, cc, dd, rr, op_all, cam.width, cam.height,
                                   target_pixels=args.target_pixels)
        t_pp = time.time() - (t1 + t_proj)
        out["cameras"][str(cid)] = {"proj_seconds": t_proj, "rd_by_ts": rd_by_ts, "pixel_sat": pp}
        rd16 = rd_by_ts.get(tile_sizes[0], rd_by_ts[tile_sizes[0]])
        sum_vis += rd16["n_visible"]
        for ts in tile_sizes:
            ts_R[ts] += rd_by_ts[ts]["n_R_occ"]; ts_occ[ts] += rd_by_ts[ts]["n_occ"]
        sum_pairs += pp["total_pairs"]; sum_skipped += pp["skipped_pairs"]
        for g in sweep_acc:
            sweep_acc[g] += rd16["gamma_sweep"].get(g, 0.0) * rd16["n_occ"]
        print("[cam %d] vis=%d | rhoR(ts)={%s} | sf=%.4f | proj=%.1fs rd=%.1fs pp=%.1fs" % (
            cid, rd16["n_visible"],
            ",".join("%d:%.5f" % (ts, rd_by_ts[ts]["rho_R_occ"]) for ts in tile_sizes),
            pp["skipped_frac"], t_proj, t_pp, 0.0), flush=True)

    rho_R_occ = sum_R_occ / sum_occ if sum_occ else 0.0
    sf = sum_skipped / sum_pairs if sum_pairs else 0.0
    sweep_mean = {g: (sweep_acc[g] / sum_occ if sum_occ else 0.0) for g in sweep_acc}
    out["agg"] = {
        "n_visible_total": int(sum_vis),
        "rho_R_by_tile_size": {str(ts): (ts_R[ts] / ts_occ[ts] if ts_occ[ts] else 0.0) for ts in tile_sizes},
        "n_R_by_tile_size": {str(ts): int(ts_R[ts]) for ts in tile_sizes},
        "n_occ_by_tile_size": {str(ts): int(ts_occ[ts]) for ts in tile_sizes},
        "gamma_sweep_rho_R_occ": sweep_mean,
        "total_pairs": int(sum_pairs), "skipped_pairs": int(sum_skipped),
        "skipped_frac": sf, "pixel_early_stop_speedup_bound": (1.0 / (1.0 - sf) if sf < 1.0 else None),
    }
    print("[agg] " + json.dumps(out["agg"]), flush=True)
    out_dir = os.environ.get("BES0_OUTPUT_DIR", ".")
    with open(os.path.join(out_dir, "opportunity_%s.json" % args.scene), "w") as f:
        json.dump(out, f, indent=2)
    print("[done] wrote opportunity_%s.json" % args.scene, flush=True)


if __name__ == "__main__":
    main()
