#!/usr/bin/env python3
"""P11 Gate L4: dynamic kernel profile (torch.profiler CUPTI; nsys/ncu are
unavailable/blocked on this host -- P10 precedent) for:
  A. publication SepSSIM path
  B. same-semantics fused-SSIM path (if L1 passes)

Profiles per scene (late 30K state, representative camera):
  - loss-only loop (loss fwd+bwd on detached render): kernel inventory;
  - full training step (render+loss+bwd+densify-stat/optimizer as applicable):
    full kernel table + phase composition.

Categorizes kernels: loss fwd, loss bwd, image-adjoint writes, raster bwd
reads/kernel, projection/SH bwd, optimizer, launches/host gaps.

Computes conservative / moderate / optimistic physically removable bounds for a
hypothetical exact loss-adjoint->raster fusion ON THE FUSED-SSIM BASELINE,
as % T_iter (T_iter from the L3 CUDA-event measurement; profile shares used
for attribution).
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/epic05_p11")
import p11_common as C

A100_PEAK_BW_GB_S = 1555.0  # A100-SXM4-80GB HBM2e peak
LAUNCH_OVERHEAD_US = 2.5    # conservative per-launch overhead used in bounds


def categorize_kernel(name):
    n = name.lower()
    if "fusedssim" in n or "fused_ssim" in n:
        return "fused_ssim"
    if any(k in n for k in ("rasterize_to_pixels", "rasterize_3dgs", "rasterization_3dgs")):
        if "bwd" in n or "backward" in n:
            return "raster_bwd"
        return "raster_fwd"
    if "projection" in n:
        return ("projection_bwd" if ("bwd" in n or "backward" in n) else "projection_fwd")
    if "spherical" in n or "sh_" in n or "_sh" in n:
        return ("sh_bwd" if ("bwd" in n or "backward" in n) else "sh_fwd")
    if "isect" in n or "radix" in n or "cub" in n or "device_reduce" in n or "scan" in n:
        return "isect_sort"
    if "adam" in n or "multi_tensor" in n:
        return "optimizer"
    if "conv" in n or "cudnn" in n or "gemm" in n or "implicit" in n:
        return "sep_loss_conv"
    if "elementwise" in n or "vectorized" in n or "reduce_kernel" in n or \
       "unrolled" in n or "tensoriterator" in n:
        return "elementwise_loss_or_other"
    if "memset" in n or "fill" in n or "zero" in n:
        return "memset_fill"
    if "memcpy" in n or "copy" in n:
        return "memcpy"
    return "other"


def profile_loop(fn, n_iters, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(n_iters):
            fn()
        torch.cuda.synchronize()
    rows = []
    total_gpu_us = 0.0
    for ev in prof.key_averages():
        if ev.self_device_time_total and ev.self_device_time_total > 0:
            rows.append({
                "name": ev.key,
                "count": ev.count,
                "gpu_us_total": ev.self_device_time_total,
                "gpu_us_per_iter": ev.self_device_time_total / n_iters,
                "cpu_us_total": ev.self_cpu_time_total,
                "category": categorize_kernel(ev.key),
            })
            total_gpu_us += ev.self_device_time_total
    rows.sort(key=lambda r: -r["gpu_us_total"])
    return {"n_iters": n_iters, "kernels": rows,
            "total_gpu_us": total_gpu_us,
            "total_gpu_ms_per_iter": total_gpu_us / n_iters / 1000.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default="/root/epic05_p11/dumps/P11_LOSS_KERNEL_PROFILE.json")
    ap.add_argument("--n-loss-iters", type=int, default=60)
    ap.add_argument("--n-step-iters", type=int, default=40)
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    torch.manual_seed(0)

    C.bootstrap_gmain()
    C.import_recipe()
    gpu_rec = C.assert_clean_gpu(args.gpu)
    ssim_fn = C.SepSSIM(device="cuda")

    per_scene = {}
    for scene in C.SCENES:
        model, dataset, extent, meta = C.load_model_and_dataset(
            scene, ckpt_path=os.path.join(C.FULL30K, f"gmain03_s42_{scene}", "ckpt_30000.pt"))
        seq = C.camera_sequence_for(scene)
        cam_idx = int(seq[0])
        cam, gt = dataset.get_item(cam_idx)
        gt = gt.contiguous()
        with torch.no_grad():
            r0, image, radii, means2d = C.render_with_meta(
                model, cam, model.active_sh_degree)
        img = image.detach().contiguous()
        img_b = img.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
        gt_b = gt.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
        H, W = img.shape[0], img.shape[1]

        # ---------------- A. SepSSIM loss-only ----------------
        img_leaf = img.detach().requires_grad_(True)
        def sep_loss_step():
            loss = C.sep_total_loss(img_leaf, gt, ssim_fn)
            torch.autograd.grad(loss, img_leaf)
        profA = profile_loop(sep_loss_step, args.n_loss_iters)

        # ---------------- B. fused-SSIM loss-only ----------------
        img_leaf_b = img_b.detach().requires_grad_(True)
        def fused_loss_step():
            x = img_leaf_b
            L1 = F.l1_loss(x, gt_b)
            s = C.fused_ssim_fn(x, gt_b, padding="same")
            loss = 0.8 * L1 + 0.2 * (1.0 - s)
            torch.autograd.grad(loss, img_leaf_b)
        profB = profile_loop(fused_loss_step, args.n_loss_iters)

        # ---------------- full training step, SepSSIM path ----------------
        def full_step_sep():
            r0, image, radii_full, means2d = C.render_with_meta(
                model, cam, model.active_sh_degree)
            L1 = F.l1_loss(image, gt)
            dssim = ssim_fn(image, gt)
            loss = 0.8 * L1 + 0.2 * dssim
            loss.backward()
            radii = radii_full[0]
            vis = (radii > 0).any(dim=-1)
            n_visible = int(vis.sum())  # trainer does this every iteration (D2H)
            model.optimizer.step()
            model.optimizer.zero_grad(set_to_none=True)
        profFS = profile_loop(full_step_sep, args.n_step_iters)

        # ---------------- full training step, fused path ----------------
        def full_step_fused():
            r0, image, radii_full, means2d = C.render_with_meta(
                model, cam, model.active_sh_degree)
            x = image.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
            g = gt_b
            L1 = F.l1_loss(x, g)
            s = C.fused_ssim_fn(x, g, padding="same")
            loss = 0.8 * L1 + 0.2 * (1.0 - s)
            loss.backward()
            radii = radii_full[0]
            vis = (radii > 0).any(dim=-1)
            n_visible = int(vis.sum())  # trainer does this every iteration (D2H)
            model.optimizer.step()
            model.optimizer.zero_grad(set_to_none=True)
        profFF = profile_loop(full_step_fused, args.n_step_iters)

        # ---------------- removable-bound model (fused baseline) ----------------
        bounds = compute_bounds(profB, profFF, H, W, scene)
        per_scene[scene] = {
            "state": {"source": meta["state_source"], "sha256": meta["state_sha256"],
                      "N": meta["state_N"]},
            "camera_index": cam_idx, "image_hw": [H, W],
            "loss_only_sepssim": profA,
            "loss_only_fused": profB,
            "full_step_sepssim": profFS,
            "full_step_fused": profFF,
            "fusion_bounds_fused_baseline": bounds,
        }
        print(f"[{scene}] loss-only GPU ms/iter: sep={profA['total_gpu_ms_per_iter']:.3f} "
              f"fused={profB['total_gpu_ms_per_iter']:.3f}")
        print(f"[{scene}] bounds (fused baseline): conservative={bounds['conservative_pct_titer']:.2f}% "
              f"moderate={bounds['moderate_pct_titer']:.2f}% "
              f"optimistic={bounds['optimistic_pct_titer']:.2f}% of T_iter="
              f"{bounds['T_iter_ms']:.2f}ms")

    record = {
        "phase": "P11 Gate L4 - dynamic kernel profile (CUPTI via torch.profiler)",
        "gpu": gpu_rec,
        "profiler_note": "nsys not installed on host; ncu hardware counters "
                         "driver-blocked (P10 precedent). CUPTI via torch.profiler "
                         "gives per-kernel GPU times; absolute times slightly "
                         "inflated by CUPTI overhead -- T_iter normalization uses "
                         "the L3 CUDA-event measurement.",
        "a100_peak_bw_gb_s": A100_PEAK_BW_GB_S,
        "launch_overhead_us": LAUNCH_OVERHEAD_US,
        "per_scene": per_scene,
    }
    C.dump_json(record, args.out)


def compute_bounds(profB, profFF, H, W, scene):
    """Physically removable bound for exact loss-adjoint->raster fusion, on the
    fused-SSIM strong baseline.

    conservative: only traffic/launches that certainly disappear:
      - full-image dL/dimage write (loss bwd final output) + read (raster bwd
        v_render_colors): 2 * H*W*3*4 bytes at peak BW;
      - loss-phase kernel launches that disappear (each subsumed kernel's launch).
    moderate: + avoidable intermediate tensor traffic in the loss phase:
      fused fwd saves dm_dmu1/dm_dsigma1_sq/dm_dsigma12 (3 full-image writes) and
      bwd reads them + img1/img2 + dL_dmap; L1 reads img+gt and writes grad.
      A tile-fused design recomputes these in-tile from (image, GT) halo reads,
      so the global-tensor materializations disappear (arithmetic remains).
    optimistic: + the launch-gap/host overhead measured inside the fused loss
      phase (steps between kernels), i.e. moderate counted at achieved loss-kernel
      bandwidth plus measured inter-kernel gaps.
    NOT counted anywhere: SSIM arithmetic that the fused design still executes;
    renderer work that remains; impossible peak-BW savings twice.
    """
    img_bytes = H * W * 3 * 4
    bw = A100_PEAK_BW_GB_S * 1e9

    # loss-phase kernels of the fused path (from loss-only profile)
    loss_rows = [r for r in profB["kernels"]]
    n_loss_launches_per_iter = sum(r["count"] for r in loss_rows) / profB["n_iters"]
    loss_gpu_ms = profB["total_gpu_ms_per_iter"]

    # T_iter from the full-step profile (CUPTI-inflated); cross-checked in
    # assembly against L3 CUDA-event T_iter (the authoritative one).
    titer_prof = profFF["total_gpu_ms_per_iter"]
    # use the L3 dump if present for the authoritative T_iter
    titer_ms = titer_prof
    l3path = "/root/epic05_p11/dumps/P11_L3_RENDERER_BWD.json"
    if os.path.exists(l3path):
        try:
            l3 = json.load(open(l3path))
            t = l3["per_scene"][scene]["late_30k"]["titer"]["T_iter_ms"]
            titer_ms = float(t)
        except Exception:
            pass

    # conservative
    dl_write_read_ms = 2 * img_bytes / bw * 1000.0
    launch_ms = n_loss_launches_per_iter * LAUNCH_OVERHEAD_US / 1000.0
    conservative_ms = dl_write_read_ms + launch_ms

    # moderate: intermediate traffic in fused loss fwd+bwd
    # fwd writes: ssim_map(no, mean-reduced), dm_dmu1, dm_dsigma1_sq, dm_dsigma12
    #             -> 3 full-image writes; reads img1, img2 (2)
    # bwd reads: img1, img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12, dL_dmap (6)
    #            writes grad (1); L1: reads img,gt (2), writes grad-part (1)
    # the dL/dimage write+read is already counted in conservative (do not double)
    moderate_extra_tensors = (3 + 2 + 6 + 1 + 2 + 1) * img_bytes
    moderate_ms = conservative_ms + moderate_extra_tensors / bw * 1000.0

    # optimistic: moderate + measured inter-kernel gaps in the fused loss phase
    # (loss wall time from CUDA events if available minus its GPU time), i.e.
    # the host/launch tail of the standalone loss phase.
    # From the L2 dump: fused combined fwd+bwd one-pass median.
    opt_extra = 0.0
    l2path = "/root/epic05_p11/dumps/P11_LOSS_TIMING.json"
    if os.path.exists(l2path):
        try:
            l2 = json.load(open(l2path))
            s = l2["per_scene"][scene]["scene_summary"]
            onepass = None
            for cam in l2["per_scene"][scene]["per_camera"]:
                if cam["camera_index"] is not None:
                    onepass = cam["summary"]["fused_fwd_plus_bwd_onepass_ms"]
                    break
            if onepass is not None:
                opt_extra = max(0.0, onepass - loss_gpu_ms - moderate_ms * 0.0)
        except Exception:
            pass
    optimistic_ms = moderate_ms + opt_extra

    return {
        "T_iter_ms": titer_ms,
        "T_iter_ms_source": "L3 CUDA-event T_iter (late_30k)" if titer_ms != titer_prof
                            else "L4 profile GPU total (fallback)",
        "image_bytes_fp32_3ch": img_bytes,
        "dl_image_write_read_ms_at_peak": dl_write_read_ms,
        "loss_launches_per_iter": n_loss_launches_per_iter,
        "loss_launch_overhead_ms": launch_ms,
        "conservative_ms": conservative_ms,
        "moderate_ms": moderate_ms,
        "optimistic_ms": optimistic_ms,
        "conservative_pct_titer": conservative_ms / titer_ms * 100.0,
        "moderate_pct_titer": moderate_ms / titer_ms * 100.0,
        "optimistic_pct_titer": optimistic_ms / titer_ms * 100.0,
        "model": {
            "conservative": "dL/dimage full-image write+read at peak BW + loss-phase "
                            "launch overhead",
            "moderate": "conservative + avoidable intermediate tensor traffic "
                        "(dm_* saves/loads, image/GT re-reads, dL_dmap, L1 grads) "
                        "recomputed in-tile from halo reads",
            "optimistic": "moderate + measured host/launch tail of the standalone "
                          "fused loss phase (CUDA-event one-pass minus CUPTI GPU time)",
            "not_counted": ["SSIM arithmetic a fused design still executes",
                            "renderer work that remains",
                            "double-counted peak-bandwidth savings"],
        },
        "fused_loss_gpu_ms_per_iter": loss_gpu_ms,
    }


if __name__ == "__main__":
    main()
