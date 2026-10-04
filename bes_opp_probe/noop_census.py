#!/usr/bin/env python3
"""No-op gaussian census: where does 3DGS forward render cost actually go?

For each strided pixel, gather covering gaussians (== scene_adapter), compute
alpha at the pixel, and classify each (pixel,gaussian) occurrence:
  significant : alpha >= ALPHA_CUTOFF  (actually contributes)
  no-op       : alpha <  ALPHA_CUTOFF  (bbox-overlaps tile but contributes ~0)
Reports the no-op fraction and the ideal per-pixel alpha-culling speedup
n_total / n_sig (an EXACT saving: skipping alpha<cutoff gaussians does not
change T or the composited color). Opportunity screen (analytic), not CUDA.
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


def census(mu, conic, depth, radii, op, W, H, target_pixels=2500):
    vis = (depth > 0.05) & (radii > 0)
    gidx = np.nonzero(vis)[0]
    if gidx.size == 0:
        return dict(n_pixels=0, total_pairs=0, sig_pairs=0, noop_pairs=0)
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
    total = 0; sig = 0; noop = 0; npp_list = []
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
            is_sig = alpha >= sa.ALPHA_CUTOFF
            ns = int(is_sig.sum()); nn = int((~is_sig).sum())
            total += ns + nn; sig += ns; noop += nn; npp_list.append(ns + nn)
    npp = np.array(npp_list) if npp_list else np.array([0])
    return dict(n_pixels=int(npp.size), total_pairs=int(total), sig_pairs=int(sig),
                noop_pairs=int(noop),
                noop_fraction=(noop / total if total else 0.0),
                ideal_cull_speedup=(total / sig if sig else float('inf')),
                mean_covering_per_pixel=float(npp.mean()) if npp.size else 0.0,
                mean_significant_per_pixel=float(npp.mean() - (noop / npp.size if npp.size else 0)) if npp.size else 0.0,
                p90_significant_per_pixel=float(np.percentile(np.maximum(npp - noop / max(npp.size,1), 0), 90)) if npp.size else 0.0)


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
    agg = dict(total=0, sig=0, noop=0, npix=0); cams = {}
    for cid in dev_ids:
        cam = scene.cameras[cid]
        mu2d, conic, depth, radii = sa.project_gaussians_2d(scene, cam)
        mu = mu2d.detach().cpu().numpy(); cc = conic.detach().cpu().numpy()
        dd = depth.detach().cpu().numpy(); rr = radii.detach().cpu().numpy()
        t0 = time.time()
        c = census(mu, cc, dd, rr, op_all, cam.width, cam.height)
        cams[str(cid)] = c
        agg["total"] += c["total_pairs"]; agg["sig"] += c["sig_pairs"]
        agg["noop"] += c["noop_pairs"]; agg["npix"] += c["n_pixels"]
        print("[cam %d] covering=%d sig=%d noop=%.4f ideal_speedup=%.2fx (%.1fs)" % (
            cid, c["total_pairs"], c["sig_pairs"], c["noop_fraction"], c["ideal_cull_speedup"], time.time() - t0), flush=True)
    out = {"schema": "bes_noop_census_v1", "scene": args.scene, "gpu_uuid": gpu_uuid,
           "cameras": cams,
           "agg": {"total_pairs": int(agg["total"]), "sig_pairs": int(agg["sig"]),
                   "noop_pairs": int(agg["noop"]),
                   "noop_fraction": (agg["noop"] / agg["total"] if agg["total"] else 0.0),
                   "ideal_cull_speedup": (agg["total"] / agg["sig"] if agg["sig"] else None)}}
    print("[agg] " + json.dumps(out["agg"]), flush=True)
    out_dir = os.environ.get("BES0_OUTPUT_DIR", ".")
    with open(os.path.join(out_dir, "noop_%s.json" % args.scene), "w") as f:
        json.dump(out, f, indent=2)
    print("[done] wrote noop_%s.json" % args.scene, flush=True)


if __name__ == "__main__":
    main()
