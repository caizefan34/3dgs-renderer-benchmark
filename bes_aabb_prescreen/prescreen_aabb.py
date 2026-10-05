#!/usr/bin/env python3
"""Decisive pre-screen for the per-tile AABB reject (exact forward speedup lever).

Hypothesis (LAW-ALPHAWALK sanctioned lever): the baseline forward walk stages every
gaussian whose 3*sigma_max SQUARE overlaps a tile, then walks all 256 pixels for
each (tile,gaussian) pair. But the alpha<1/255 region is the per-axis AABB
[cx+-dx_max, cy+-dy_max] with dx_max=sqrt(T*s11), dy_max=sqrt(T*s22), T=2*ln(255*op),
which is FAR tighter than the 3*sigma_max square for anisotropic gaussians. So many
(tile,gaussian) pairs have an AABB that does NOT overlap the tile -> all 256 q-tests
are provably no-op and can be skipped with ONE cheap per-tile test.

This script measures, per dev camera (pure numpy, no kernel):
  walked_pairs     = sum over gaussians of tiles overlapped by 3*sigma_max square
  surviving_pairs  = sum over gaussians of tiles overlapped by the 1/255 AABB
  s_tile           = surviving_pairs / walked_pairs   (per-tile reject q-test fraction)
  s_pixel          = surviving_pixels / walked_pixels (per-pixel reject fraction)
Also reports elongation (sigma_max/sigma_min) and size distributions.

If s_tile is small (<=0.4), the per-tile AABB reject is a real >1.5x exact lever and
a kernel is worth building. If s_tile ~ 1, the lever is dead (consistent with the
circle-reject null) and we close the direction.
"""
import sys, os, json, math, time, argparse
import numpy as np

PACKAGE_SCRIPTS = os.environ["BES0_PACKAGE_ROOT"] + "/scripts"
sys.path.insert(0, PACKAGE_SCRIPTS)
import mx_guard
cfg = mx_guard.assert_mx_host()
import torch
import scene_adapter as sa

TS = 16


def resolve_gpu_uuid(cfg, prefer_index):
    for x in mx_guard.checked_inventory(cfg):
        if x["index"] == prefer_index and x["index"] in cfg["allowed_gpu_indices"]:
            return x["uuid"]
    raise mx_guard.GuardError("no allowed idle GPU at index %d" % prefer_index)


def patient_idle_wait(cfg, gpu_uuid, max_wait_min=30):
    deadline = time.time() + max_wait_min * 60
    while time.time() < deadline:
        try:
            for _ in range(3):
                mx_guard.assert_idle(cfg, gpu_uuid)
                time.sleep(1.0)
            return
        except mx_guard.GuardError as exc:
            print("[guard] waiting: %s" % exc, flush=True); time.sleep(3)
    raise mx_guard.GuardError("GPU_NEVER_IDLE")


def clamp(v, lo, hi):
    return np.clip(v, lo, hi)


def screen_camera(mu2d, conic, depth, radii, op, W, H):
    V = mu2d.shape[0]
    tiles_x = (W + TS - 1) // TS
    tiles_y = (H + TS - 1) // TS
    if V == 0:
        return dict(n_gauss=0, walked_pairs=0, surviving_pairs=0,
                    walked_pixels=0, surviving_pixels=0)
    mu2d = np.asarray(mu2d, dtype=np.float64)
    conic = np.asarray(conic, dtype=np.float64)
    m = mu2d
    # recover covariance diagonal from conic = Sigma^-1 = (1/detS)[[s22,-s12],[-s12,s11]]
    c0 = conic[:, 0]; c1 = conic[:, 1]; c2 = conic[:, 2]
    det_conic = np.maximum(c0 * c2 - c1 * c1, 1e-30)
    s11 = c2 / det_conic          # Sigma[0,0]
    s22 = c0 / det_conic          # Sigma[1,1]
    # eigenvalues of Sigma for elongation
    mid = 0.5 * (s11 + s22)
    detS = 1.0 / det_conic
    disc2 = np.maximum(mid * mid - detS, 0.0)
    lam_max = mid + np.sqrt(disc2)
    lam_min = np.maximum(mid - np.sqrt(disc2), 1e-20)
    sig_max = np.sqrt(lam_max)
    sig_min = np.sqrt(lam_min)
    elong = sig_max / np.maximum(sig_min, 1e-20)

    opv = np.maximum(op, 1e-12)
    T = 2.0 * (np.log(255.0) + np.log(opv))     # q cutoff for alpha<1/255
    Tpos = np.maximum(T, 0.0)
    dx_max = np.sqrt(Tpos * s11)
    dy_max = np.sqrt(Tpos * s22)
    empty = T <= 0.0   # op<=1/255 -> AABB empty -> reject everywhere

    r = radii.astype(np.float64)
    # 3*sigma_max square tile overlap (== build_tile_lists)
    tx0 = clamp(np.floor((m[:, 0] - r) / TS).astype(np.int64), 0, tiles_x - 1)
    tx1 = clamp(np.floor((m[:, 0] + r) / TS).astype(np.int64), 0, tiles_x - 1)
    ty0 = clamp(np.floor((m[:, 1] - r) / TS).astype(np.int64), 0, tiles_y - 1)
    ty1 = clamp(np.floor((m[:, 1] + r) / TS).astype(np.int64), 0, tiles_y - 1)
    tiles_square = (tx1 - tx0 + 1).astype(np.int64) * (ty1 - ty0 + 1).astype(np.int64)

    # 1/255 AABB tile overlap
    ax0 = clamp(np.floor((m[:, 0] - dx_max) / TS).astype(np.int64), 0, tiles_x - 1)
    ax1 = clamp(np.floor((m[:, 0] + dx_max) / TS).astype(np.int64), 0, tiles_x - 1)
    ay0 = clamp(np.floor((m[:, 1] - dy_max) / TS).astype(np.int64), 0, tiles_y - 1)
    ay1 = clamp(np.floor((m[:, 1] + dy_max) / TS).astype(np.int64), 0, tiles_y - 1)
    tiles_aabb = (ax1 - ax0 + 1).astype(np.int64) * (ay1 - ay0 + 1).astype(np.int64)
    tiles_aabb = np.where(empty, 0, tiles_aabb)

    walked_pairs = int(tiles_square.sum())
    surviving_pairs = int(tiles_aabb.sum())
    # pixel-level: surviving pixels ~ AABB area (clamped to image); walked = 256*tiles_square
    aabb_w = np.minimum(2.0 * dx_max, W)
    aabb_h = np.minimum(2.0 * dy_max, H)
    surviving_pixels = float(np.sum(aabb_w * aabb_h))
    walked_pixels = float(walked_pairs) * 256.0

    return dict(
        n_gauss=int(V),
        walked_pairs=walked_pairs, surviving_pairs=surviving_pairs,
        walked_pixels=walked_pixels, surviving_pixels=surviving_pixels,
        s_tile=(surviving_pairs / walked_pairs if walked_pairs else 0.0),
        s_pixel=(surviving_pixels / walked_pixels if walked_pixels else 0.0),
        elong_mean=float(elong.mean()), elong_median=float(np.median(elong)),
        elong_p90=float(np.percentile(elong, 90)), elong_max=float(elong.max()),
        frac_elong_gt10=float((elong > 10).mean()),
        sigmax_median=float(np.median(sig_max)),
        frac_empty=float(empty.mean()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--gpu-index", type=int, default=7)
    ap.add_argument("--max-cams", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = getattr(args, "device", "cuda")
    if str(device).startswith("cuda"):
        gpu_uuid = resolve_gpu_uuid(cfg, args.gpu_index)
        patient_idle_wait(cfg, gpu_uuid)
        print("[guard] idle OK %s" % gpu_uuid, flush=True)
        torch.set_num_threads(1); torch.cuda.set_device(0)
    else:
        gpu_uuid = "cpu-no-gpu"
        torch.set_num_threads(8)
        print("[cpu-mode] no GPU used, no idle wait", flush=True)
    man = json.load(open(args.manifest)); rec = man["scenes"][args.scene]
    dev_ids = list(rec["dev_camera_ids"])
    if args.max_cams > 0:
        dev_ids = dev_ids[:args.max_cams]
    scene = sa.load_scene(rec, device)
    op_all = sa.opacity_activation(scene.opacity).reshape(-1).detach().cpu().numpy()
    agg = dict(walked_pairs=0, surviving_pairs=0, walked_pixels=0.0, surviving_pixels=0.0,
               n_gauss=0, elong_wt=0.0)
    cams = {}
    for cid in dev_ids:
        cam = scene.cameras[cid]
        mu2d, conic, depth, radii = sa.project_gaussians_2d(scene, cam)
        mu2d = mu2d.detach().cpu().numpy()
        conic = conic.detach().cpu().numpy()
        depth = depth.detach().cpu().numpy()
        radii = radii.detach().cpu().numpy()
        vis = (depth > 0.05) & (radii > 0)
        c = screen_camera(mu2d[vis], conic[vis], depth[vis], radii[vis], op_all[vis],
                          cam.width, cam.height)
        cams[str(cid)] = c
        agg["walked_pairs"] += c["walked_pairs"]; agg["surviving_pairs"] += c["surviving_pairs"]
        agg["walked_pixels"] += c["walked_pixels"]; agg["surviving_pixels"] += c["surviving_pixels"]
        agg["n_gauss"] += c["n_gauss"]
        agg["elong_wt"] += c["elong_mean"] * c["n_gauss"]
        print("[cam %d] nG=%d walked_pairs=%d surv_pairs=%d s_tile=%.4f s_pixel=%.4f elong_med=%.2f elong_p90=%.2f sigmax_med=%.2f empty=%.3f" % (
            cid, c["n_gauss"], c["walked_pairs"], c["surviving_pairs"], c["s_tile"], c["s_pixel"],
            c["elong_median"], c["elong_p90"], c["sigmax_median"], c["frac_empty"]), flush=True)
    s_tile = agg["surviving_pairs"] / agg["walked_pairs"] if agg["walked_pairs"] else 0.0
    s_pixel = agg["surviving_pixels"] / agg["walked_pixels"] if agg["walked_pixels"] else 0.0
    out = dict(schema="bes_aabb_prescreen_v1", scene=args.scene, gpu_uuid=gpu_uuid,
               cams=cams,
               agg=dict(n_gauss=agg["n_gauss"],
                        walked_pairs=agg["walked_pairs"],
                        surviving_pairs=agg["surviving_pairs"],
                        walked_pixels=agg["walked_pixels"],
                        surviving_pixels=agg["surviving_pixels"],
                        s_tile=s_tile, s_pixel=s_pixel,
                        elong_mean_weighted=agg["elong_wt"] / max(agg["n_gauss"], 1)))
    print("[agg] " + json.dumps({k: out["agg"][k] for k in
        ["n_gauss","walked_pairs","surviving_pairs","s_tile","s_pixel","elong_mean_weighted"]}), flush=True)
    out_dir = os.environ.get("BES0_OUTPUT_DIR", ".")
    with open(os.path.join(out_dir, "aabb_%s.json" % args.scene), "w") as f:
        json.dump(out, f, indent=2)
    print("[done] wrote aabb_%s.json" % args.scene, flush=True)


if __name__ == "__main__":
    main()
