#!/usr/bin/env python3
"""M1 cost profile: strong-baseline step components, coarse/proxy branch,
anchor generation/transfer/storage, and raster work fractions.

All timings are CUDA-event medians over repeats on the shared A100
(timing_grade=FUNCTIONAL_ONLY; see env/occupancy record). Raw per-iteration
samples are written to a CSV. Components that cannot be measured inside the
frozen binary (internal projection/sort split of the rasterizer, sparse-kernel
overheads that do not exist yet) are marked HYPOTHESIS in the JSON, never as
measured values.
"""
import argparse
import csv
import json
import os
import statistics
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gx_common as G  # noqa: E402


def ev_time(fn, repeats, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
           for _ in range(repeats)]
    samples = []
    for a, b in evs:
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        samples.append(a.elapsed_time(b))
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--camera-index", type=int, required=True)
    ap.add_argument("--camera-seq", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--baseline-iters", type=int, default=30)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--max-blocks", type=int, default=32)
    ap.add_argument("--halo", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()

    t0_wall = time.time()
    t, ident = G.boot_gmain(args.gpu)
    fused_ident = G.check_fused_lock()
    G.add_grc3d_path()
    from grc3d.core import Anchor, blocks, block_losses, loss_map, upsample
    from grc3d.probe import load_bundle
    from grc3d.renderers import render_gsplat

    OUT = {"scene": args.scene, "bundle": args.bundle,
           "bundle_sha256": G.sha256_file(args.bundle),
           "snapshot": args.snapshot, "snapshot_sha256": G.sha256_file(args.snapshot),
           "identity": G.run_identity(t, ident, args.gpu),
           "fused_loss_identity": fused_ident,
           "timing_grade": "FUNCTIONAL_ONLY / shared GPU (mechanism diagnostics only)",
           "timing_note": ("per-iteration phase split of the real training step is taken "
                           "from the replay runs (timing.json of replay_*_14000); this "
                           "script measures coarse branch, proxy losses, anchor costs and "
                           "raster work fractions on the same states"),
           "components_ms": {}, "counts": {}, "fractions": {}, "hypotheses": {}}
    OUT["identity"]["gpu_inventory_at_start"] = G.gpu_inventory()

    dataset = G.build_dataset(args.scene)
    seq = np.load(args.camera_seq)
    assert args.camera_index in set(int(v) for v in seq)
    cam, gt = dataset.get_item(args.camera_index)
    H_, W_ = int(cam.image_height), int(cam.image_width)
    camera = G.camera_dict(cam)
    target = gt.detach().float()
    ch, cw = H_ // 4, W_ // 4
    degree = int(torch.load(args.bundle, map_location="cpu",
                            weights_only=True)["sh_degree"])

    # ---------------- raster work fractions + depth complexity --------------
    import gsplat
    snap = torch.load(args.snapshot, map_location="cpu", weights_only=False)
    params = {k: snap[k].cuda().float() for k in
              ("xyz", "rotation", "scaling", "opacity", "shs")}
    with torch.no_grad():
        r, _, meta = gsplat.rasterization(
            means=params["xyz"], quats=F.normalize(params["rotation"], dim=-1),
            scales=params["scaling"].exp(), opacities=params["opacity"].sigmoid(),
            colors=params["shs"], viewmats=camera["viewmat"][None],
            Ks=camera["K"][None], width=W_, height=H_, tile_size=16, packed=False,
            sh_degree=degree, radius_clip=0.0, eps2d=0.3, render_mode="RGB")
    OUT["counts"]["meta_keys"] = sorted(meta.keys())
    rw = OUT["counts"]["raster_work"] = {}
    rw["n_gaussians"] = int(params["xyz"].shape[0])
    rw["n_visible_radii_gt0"] = int((meta["radii"] > 0).any(dim=-1).sum()) \
        if "radii" in meta else None
    if "isect_ids" in meta and meta["isect_ids"] is not None:
        isect = meta["isect_ids"]
        total_isect = int(isect.numel())
        rw["total_tile_gaussian_intersections"] = total_isect
        rw["tiles_touched"] = int(torch.unique(isect // (1 << 32)).numel()) \
            if isect.numel() else 0
        rw["n_tiles_total"] = ((H_ + 15) // 16) * ((W_ + 15) // 16)
        # selected 32 probe blocks: tile span from block geometry (64x64 centers,
        # halo 5 -> 74x74 patch); count intersections of those tiles exactly
        part = blocks(H_, W_, args.block_size)
        idx = list(range(len(part)))
        idx = torch.randperm(len(part), generator=torch.Generator().manual_seed(args.seed)
                             )[:args.max_blocks].tolist()
        tiles = set()
        for i in idx:
            b = part[i]
            y0, x0 = max(0, b.y0 - args.halo), max(0, b.x0 - args.halo)
            y1, x1 = min(H_, b.y1 + args.halo), min(W_, b.x1 + args.halo)
            for ty in range(y0 // 16, (y1 - 1) // 16 + 1):
                for tx in range(x0 // 16, (x1 - 1) // 16 + 1):
                    tiles.add(ty * (W_ // 16) + tx)
        rw["selected_unique_pixels"] = int(sum(
            (min(H_, part[i].y1 + args.halo) - max(0, part[i].y0 - args.halo)) *
            (min(W_, part[i].x1 + args.halo) - max(0, part[i].x0 - args.halo))
            for i in idx))
        rw["n_pixels_total"] = int(H_ * W_)
        rw["selected_unique_pixels_fraction"] = \
            rw["selected_unique_pixels"] / rw["n_pixels_total"]
        rw["selected_tiles"] = len(tiles)
        rw["selected_tiles_fraction_of_total"] = len(tiles) / rw["n_tiles_total"]
        tile_ids = isect // (1 << 32)
        sel_mask = torch.isin(tile_ids, torch.tensor(sorted(tiles), device=isect.device))
        rw["selected_tile_intersections"] = int(sel_mask.sum())
        rw["selected_tile_intersections_fraction"] = \
            (rw["selected_tile_intersections"] / total_isect) if total_isect else None
    del r, meta, params
    torch.cuda.empty_cache()

    # ---------------- component timings (coarse branch, losses, anchors) ----
    p, old, camera_b, target_b, degree_b, _ = load_bundle(args.bundle, "cuda")
    assert degree_b == degree
    g = {"means": p["means"].detach(), "quats": p["quats"].detach(),
         "log_scales": p["log_scales"].detach(), "opacity_logits": p["opacity_logits"].detach(),
         "sh0": p["sh0"].detach(), "shN": p["shN"].detach()}

    import fused_ssim

    def fused_loss(pred_hwc, gt_hwc):
        b = pred_hwc.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
        gt_ = gt_hwc.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
        l1 = F.l1_loss(b, gt_)
        ssim = fused_ssim.fused_ssim(b, gt_, padding="same", train=True)
        return 0.8 * l1 + 0.2 * (1.0 - ssim)

    vm, K = camera["viewmat"], camera["K"]

    def baseline_raster(leaves, height, width):
        r, _, meta = gsplat.rasterization(
            means=leaves["means"], quats=F.normalize(leaves["quats"], dim=-1),
            scales=leaves["log_scales"].exp(), opacities=leaves["opacity_logits"].sigmoid(),
            colors=torch.cat([leaves["sh0"], leaves["shN"]], 1),
            viewmats=vm[None], Ks=K[None], width=width, height=height, tile_size=16,
            packed=False, sh_degree=degree, radius_clip=0.0, eps2d=0.3,
            render_mode="RGB", absgrad=True)
        return r[0].clamp(0, 1)

    cm = OUT["components_ms"]
    cm["baseline_fine_forward"] = ev_time(
        lambda: baseline_raster(g, H_, W_), args.repeats)
    cm["coarse_quarter_forward"] = ev_time(
        lambda: baseline_raster(g, ch, cw), args.repeats)

    for name, hh, ww in (("fused_loss_at_full_res", H_, W_),
                         ("fused_loss_at_coarse_res", ch, cw)):
        img = baseline_raster(g, hh, ww).detach()
        tg = target if (hh, ww) == (H_, W_) else F.interpolate(
            target.permute(2, 0, 1)[None], size=(hh, ww), mode="bilinear",
            align_corners=False)[0].permute(1, 2, 0).clamp(0, 1)
        cm[name] = ev_time(lambda: fused_loss(img, tg), args.repeats)

    def coarse_fb():
        leaves = {k: v.clone().requires_grad_(True) for k, v in g.items()}
        img = baseline_raster(leaves, ch, cw)
        loss = fused_loss(img, F.interpolate(
            target.permute(2, 0, 1)[None], size=(ch, cw), mode="bilinear",
            align_corners=False)[0].permute(1, 2, 0).clamp(0, 1))
        grads = torch.autograd.grad(loss, tuple(leaves.values()))
        del leaves, img, loss, grads
    cm["coarse_branch_forward_loss_backward"] = ev_time(coarse_fb, args.repeats, warmup=2)

    def fine_fb():
        leaves = {k: v.clone().requires_grad_(True) for k, v in g.items()}
        img = baseline_raster(leaves, H_, W_)
        loss = fused_loss(img, target)
        grads = torch.autograd.grad(loss, tuple(leaves.values()))
        del leaves, img, loss, grads
    cm["baseline_fine_forward_loss_backward"] = ev_time(fine_fb, args.repeats, warmup=2)

    with torch.no_grad():
        a = Anchor.capture(render_gsplat(old, camera, H_, W_, degree, eps2d=0.3),
                           render_gsplat(old, camera, ch, cw, degree, eps2d=0.3))
    cm["anchor_A_generation_full_res"] = ev_time(
        lambda: render_gsplat(old, camera, H_, W_, degree, eps2d=0.3), args.repeats)
    cm["anchor_B_generation_coarse"] = ev_time(
        lambda: render_gsplat(old, camera, ch, cw, degree, eps2d=0.3), args.repeats)
    A = a.fine.detach()
    B = a.coarse.detach()
    OUT["counts"]["anchor_bytes"] = {
        "A_full_res": int(A.numel() * A.element_size()),
        "B_coarse": int(B.numel() * B.element_size()),
        "A_shape": list(A.shape), "B_shape": list(B.shape)}
    cm["anchor_A_d2d_copy"] = ev_time(lambda: A.clone(), args.repeats)
    cm["anchor_B_d2d_copy"] = ev_time(lambda: B.clone(), args.repeats)

    def proxy_full_loss():
        coarse_now = render_gsplat(p, camera, ch, cw, degree, eps2d=0.3)
        proxy = upsample(a.proxy(coarse_now), H_, W_)
        loss_map(proxy, target)
    cm["proxy_branch_full_res_separable_SSIM_no_bwd"] = ev_time(proxy_full_loss, 10, warmup=2)

    def proxy_upsample_only():
        upsample(a.proxy(render_gsplat(p, camera, ch, cw, degree, eps2d=0.3)), H_, W_)
    cm["proxy_upsample_no_loss"] = ev_time(proxy_upsample_only, args.repeats)

    del a, A, B, p, old
    torch.cuda.empty_cache()

    # ---------------- summaries --------------------------------------------
    def med(name):
        v = cm.get(name)
        return statistics.median(v) if v else None

    base_fwd = med("baseline_fine_forward")
    coarse_f = med("coarse_quarter_forward")
    OUT["fractions"] = {
        "coarse_fwd_over_fine_fwd": (coarse_f / base_fwd) if base_fwd else None,
    }
    if "coarse_branch_forward_loss_backward" in cm:
        OUT["fractions"]["coarse_branch_fb_over_fine_fwd"] = (
            med("coarse_branch_forward_loss_backward") / base_fwd)
    if "baseline_fine_forward_loss_backward" in cm:
        OUT["fractions"]["coarse_branch_fb_over_fine_fb"] = (
            med("coarse_branch_forward_loss_backward") /
            med("baseline_fine_forward_loss_backward"))
    OUT["hypotheses"] = {
        "sparse_fine_branch_r": {
            "statement": ("cost of rendering the selected 64x64 centers + halo 5 at "
                          "full resolution as a fraction of the baseline fine raster "
                          "forward; no sparse kernel exists in the frozen binary"),
            "measured_bounds": {
                "unique_pixel_fraction": rw.get("selected_unique_pixels_fraction"),
                "tile_fraction": rw.get("selected_tiles_fraction_of_total"),
                "intersection_fraction": rw.get("selected_tile_intersections_fraction")},
            "proposed_range": [rw.get("selected_tile_intersections_fraction"),
                               rw.get("selected_unique_pixels_fraction")],
            "note": ("lower bound = measured depth-complexity fraction (tile-selective "
                     "raster reading the same sorted lists); upper bound = unique-pixel "
                     "fraction assuming per-pixel cost equal to dense raster")},
        "internal_projection_sort_split": {
            "statement": ("fraction of coarse forward spent in projection vs sort vs "
                          "raster; the frozen gsplat binary exposes no counters"),
            "status": "HYPOTHESIS, not measured"},
        "eta_quality_steps": {"statement": "equal-quality step multiplier; unknown in M1",
                              "status": "HYPOTHESIS"},
        "delta_dense_calibration": {"statement": "dense-calibration proportion and cost",
                                    "status": "HYPOTHESIS"},
        "zeta_amortization": {"statement": "extra init/amortized costs", "status": "HYPOTHESIS"},
    }
    OUT["wall_time_s"] = time.time() - t0_wall
    OUT["gpu_inventory_at_end"] = G.gpu_inventory()

    with open(args.out, "w") as f:
        json.dump(OUT, f, indent=2, allow_nan=False)
    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["component", "sample_index", "ms"])
        for name, samples in cm.items():
            if samples:
                for i, s in enumerate(samples):
                    w.writerow([name, i, s])
    for name in sorted(cm):
        if cm[name]:
            v = cm[name]
            print(f"[cost] {name}: median={statistics.median(v):.4f} ms "
                  f"min={min(v):.4f} max={max(v):.4f} n={len(v)}", flush=True)
    print(f"[cost] raster_work: {json.dumps(rw)}", flush=True)
    print(f"[cost] wrote {args.out} + {args.csv}", flush=True)


if __name__ == "__main__":
    main()
