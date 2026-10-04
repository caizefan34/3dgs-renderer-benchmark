#!/usr/bin/env python3
"""Certified quality-speed headroom curve for per-pixel early-termination.

For each dev camera, project + bin (== scene_adapter), gather covering
gaussians per strided pixel, sort front-to-back, and for a sweep of
transmittance floors eps compute the fraction of (pixel,gaussian) occurrences
skippable once cumulative transmittance T < eps. Certified bound: truncating a
pixel at T < eps changes its radiance by at most eps (in [0,1]), i.e. an error
ceiling of eps -> PSNR ceiling ~ -20*log10(eps) dB. This quantifies the ONLY
positive lead from the BES0 falsification (quality-aware early-stop).

Opportunity screen (analytic), not a CUDA timing.
"""
import sys, os, json, time, math, argparse
PACKAGE_SCRIPTS = os.environ["BES0_PACKAGE_ROOT"] + "/scripts"
sys.path.insert(0, PACKAGE_SCRIPTS)
import mx_guard
cfg = mx_guard.assert_mx_host()
import numpy as np
import torch
import scene_adapter as sa

TS = 16
EPS_SWEEP = [1.0/255.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15]


def patient_idle_wait(cfg, gpu_uuid, max_wait_min=30):
    deadline = time.time() + max_wait_min * 60
    while time.time() < deadline:
        try:
            return mx_guard.wait_idle_window(cfg, gpu_uuid)
        except mx_guard.GuardError as exc:
            print("[guard] waiting: %s" % exc, flush=True); time.sleep(10)
    raise mx_guard.GuardError("GPU_NEVER_IDLE")


def resolve_gpu_uuid(cfg, prefer_index):
    for x in mx_guard.checked_inventory(cfg):
        if x["index"] == prefer_index and x["index"] in cfg["allowed_gpu_indices"]:
            return x["uuid"]
    raise mx_guard.GuardError("no allowed idle GPU at index %d" % prefer_index)


def headroom_sweep(mu, conic, depth, radii, op, W, H, target_pixels=2500):
    vis = (depth > 0.05) & (radii > 0)
    gidx = np.nonzero(vis)[0]
    if gidx.size == 0:
        return dict(n_pixels=0, total_pairs=0, sf={e: 0.0 for e in EPS_SWEEP})
    m = mu[gidx]; cc = conic[gidx]; rr = radii[gidx].astype(np.float64)
    d = depth[gidx]; opv = op[gidx]
    tiles_x = (W + TS - 1) // TS; tiles_y = (H + TS - 1) // TS
    tx = np.clip(np.floor(m[:, 0] / TS).astype(np.int64), 0, tiles_x - 1)
    ty = np.clip(np.floor(m[:, 1] / TS).astype(np.int64), 0, tiles_y - 1)
    key = ty * tiles_x + tx
    order = np.argsort(key, kind="stable")
    keys_sorted = key[order]
    uniq, ustart = np.unique(keys_sorted, return_index=True)
    slist = list(ustart) + [len(order)]
    krange = {int(uniq[i]): (slist[i], slist[i + 1]) for i in range(len(uniq))}
    ring = 2
    stride = max(8, int(math.sqrt(W * H / max(1, target_pixels))))
    xs = np.arange(0, W, stride); ys = np.arange(0, H, stride)
    total_pairs = 0
    skipped = {e: 0 for e in EPS_SWEEP}
    npix = 0
    for py in ys:
        cty = py // TS
        for px in xs:
            ctx = px // TS
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
            n = al.size
            if n == 0:
                continue
            Tbefore = np.concatenate([[1.0], np.cumprod(1.0 - al)[:-1]])
            total_pairs += n; npix += 1
            for e in EPS_SWEEP:
                below = np.nonzero(Tbefore < e)[0]
                sat = int(below[0]) if below.size else n
                skipped[e] += n - sat
    sf = {("%.6f" % e): (skipped[e] / total_pairs if total_pairs else 0.0) for e in EPS_SWEEP}
    return dict(n_pixels=npix, total_pairs=int(total_pairs), sf=sf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--gpu-index", type=int, default=7)
    ap.add_argument("--max-cams", type=int, default=0)
    args = ap.parse_args()
    gpu_uuid = resolve_gpu_uuid(cfg, args.gpu_index)
    patient_idle_wait(cfg, gpu_uuid)
    print("[guard] idle OK %s" % gpu_uuid, flush=True)
    torch.set_num_threads(1); torch.cuda.set_device(0)
    man = json.load(open(args.manifest)); rec = man["scenes"][args.scene]
    dev_ids = list(rec["dev_camera_ids"])
    if args.max_cams > 0:
        dev_ids = dev_ids[:args.max_cams]
    scene = sa.load_scene(rec, "cuda")
    op_all = sa.opacity_activation(scene.opacity).reshape(-1).detach().cpu().numpy()
    agg_pairs = 0; agg_sk = {e: 0 for e in EPS_SWEEP}
    cams = {}
    for cid in dev_ids:
        cam = scene.cameras[cid]
        mu2d, conic, depth, radii = sa.project_gaussians_2d(scene, cam)
        mu = mu2d.detach().cpu().numpy(); cc = conic.detach().cpu().numpy()
        dd = depth.detach().cpu().numpy(); rr = radii.detach().cpu().numpy()
        t0 = time.time()
        h = headroom_sweep(mu, cc, dd, rr, op_all, cam.width, cam.height)
        cams[str(cid)] = h
        agg_pairs += h["total_pairs"]
        for e in EPS_SWEEP:
            agg_sk[e] += int(round(h["sf"]["%.6f" % e] * h["total_pairs"]))
        print("[cam %d] pairs=%d sf=%s (%.1fs)" % (cid, h["total_pairs"],
              {("%.4f" % e): round(h["sf"]["%.6f" % e], 4) for e in EPS_SWEEP}, time.time() - t0), flush=True)
    sf_agg = {("%.6f" % e): (agg_sk[e] / agg_pairs if agg_pairs else 0.0) for e in EPS_SWEEP}
    out = {"schema": "bes_headroom_v1", "scene": args.scene, "gpu_uuid": gpu_uuid,
           "eps_sweep": EPS_SWEEP, "cameras": cams,
           "agg": {"total_pairs": int(agg_pairs), "sf_by_eps": sf_agg,
                   "certified_error_ceiling": {("%.6f" % e): e for e in EPS_SWEEP},
                   "psnr_ceiling_dB": {("%.6f" % e): float(-20*math.log10(e)) for e in EPS_SWEEP}}}
    print("[agg] " + json.dumps(out["agg"]), flush=True)
    out_dir = os.environ.get("BES0_OUTPUT_DIR", ".")
    with open(os.path.join(out_dir, "headroom_%s.json" % args.scene), "w") as f:
        json.dump(out, f, indent=2)
    print("[done] wrote headroom_%s.json" % args.scene, flush=True)


if __name__ == "__main__":
    main()
