#!/usr/bin/env python3
"""P11 Gate L2: loss-only timing decomposition (renderer NOT timed).

Detached rendered images with requires_grad_(True) from the frozen 30K DSH-A
states. Benchmarks, per scene and camera:
  L1 fwd, L1 bwd-to-image, SepSSIM fwd, SepSSIM bwd-to-image,
  combined SepSSIM+L1 fwd, combined bwd-to-image,
  fused-SSIM fwd, fused-SSIM bwd-to-image, combined fused-SSIM+L1 fwd/bwd.

Protocol: warmup>=20, measurements>=100, reps>=5, CUDA events, clean GPU.
Backward timings use a retained graph (pure backward, no fwd rebuild);
combined fwd/bwd timings rebuild the graph every call (training-faithful).

Outputs T_loss_sep_total, T_loss_fused_total, S_loss per scene.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/epic05_p11")
import p11_common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default="/root/epic05_p11/dumps/P11_LOSS_TIMING.json")
    ap.add_argument("--n-cameras", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--reps", type=int, default=5)
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
        cams = [int(x) for x in seq[:args.n_cameras]]

        per_cam = []
        for cam_idx in cams:
            cam, gt = dataset.get_item(cam_idx)
            with torch.no_grad():
                r0, image, radii, means2d = C.render_with_meta(
                    model, cam, model.active_sh_degree)
            img = image.detach().contiguous()      # [H,W,3]
            gt = gt.contiguous()
            res = {"camera_index": cam_idx, "image_shape": list(img.shape)}

            # ---------- L1 ----------
            img_leaf = img.detach().requires_grad_(True)
            L1 = F.l1_loss(img_leaf, gt)
            res["l1_forward"] = C.bench_events(
                lambda: F.l1_loss(img_leaf, gt), args.warmup, args.iters, args.reps)
            res["l1_backward"] = C.bench_events(
                lambda: torch.autograd.grad(L1, img_leaf, retain_graph=True),
                args.warmup, args.iters, args.reps)
            res["l1_forward_backward"] = C.bench_events(
                lambda: torch.autograd.grad(F.l1_loss(img_leaf, gt), img_leaf),
                args.warmup, args.iters, args.reps)

            # ---------- SepSSIM ----------
            img_leaf = img.detach().requires_grad_(True)
            dssim = ssim_fn(img_leaf, gt)
            res["sepssim_forward"] = C.bench_events(
                lambda: ssim_fn(img_leaf, gt), args.warmup, args.iters, args.reps)
            res["sepssim_backward"] = C.bench_events(
                lambda: torch.autograd.grad(dssim, img_leaf, retain_graph=True),
                args.warmup, args.iters, args.reps)

            # ---------- combined SepSSIM+L1 ----------
            img_leaf = img.detach().requires_grad_(True)
            loss_sep = C.sep_total_loss(img_leaf, gt, ssim_fn)
            res["combined_sep_forward"] = C.bench_events(
                lambda: C.sep_total_loss(img_leaf, gt, ssim_fn),
                args.warmup, args.iters, args.reps)
            res["combined_sep_backward"] = C.bench_events(
                lambda: torch.autograd.grad(loss_sep, img_leaf, retain_graph=True),
                args.warmup, args.iters, args.reps)
            res["combined_sep_forward_backward"] = C.bench_events(
                lambda: torch.autograd.grad(C.sep_total_loss(img_leaf, gt, ssim_fn),
                                            img_leaf),
                args.warmup, args.iters, args.reps)
            del loss_sep

            # ---------- fused-SSIM ----------
            img_b = img.detach().unsqueeze(0).permute(0, 3, 1, 2).contiguous()
            gt_b = gt.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
            img_leaf_b = img_b.detach().requires_grad_(True)
            ssim_f = C.fused_ssim_fn(img_leaf_b, gt_b, padding="same")
            res["fused_forward"] = C.bench_events(
                lambda: C.fused_ssim_fn(img_leaf_b, gt_b, padding="same"),
                args.warmup, args.iters, args.reps)
            res["fused_backward"] = C.bench_events(
                lambda: torch.autograd.grad(ssim_f, img_leaf_b, retain_graph=True),
                args.warmup, args.iters, args.reps)

            # ---------- combined fused-SSIM+L1 ----------
            def fused_combined():
                x = img_leaf_b
                L1f = F.l1_loss(x, gt_b)
                s = C.fused_ssim_fn(x, gt_b, padding="same")
                return 0.8 * L1f + 0.2 * (1.0 - s)
            img_leaf_b2 = img_b.detach().requires_grad_(True)
            img_leaf_b = img_leaf_b2
            loss_f = fused_combined()
            res["combined_fused_forward"] = C.bench_events(
                fused_combined, args.warmup, args.iters, args.reps)
            res["combined_fused_backward"] = C.bench_events(
                lambda: torch.autograd.grad(loss_f, img_leaf_b, retain_graph=True),
                args.warmup, args.iters, args.reps)
            res["combined_fused_forward_backward"] = C.bench_events(
                lambda: torch.autograd.grad(fused_combined(), img_leaf_b),
                args.warmup, args.iters, args.reps)
            del loss_f

            # NOTE on the fused path's L1: the trainer's L1 operates on [H,W,3]
            # HWC; the fused-SSIM path works on [B,C,H,W]. For the strong-baseline
            # loss module the L1 is evaluated on the same permuted layout so the
            # module is one coherent unit; L1 on a permuted view has identical
            # semantics/values (l1_loss is layout-independent up to fp reduction
            # order). The separate l1_* rows above use the trainer's exact HWC
            # call for reference.

            def mom(key):
                return res[key]["median_of_medians_ms"]

            t_sep_total = mom("combined_sep_forward") + mom("combined_sep_backward")
            t_fused_total = mom("combined_fused_forward") + mom("combined_fused_backward")
            res["summary"] = {
                "T_loss_sep_total_ms": t_sep_total,
                "T_loss_fused_total_ms": t_fused_total,
                "S_loss": t_sep_total / t_fused_total if t_fused_total > 0 else None,
                "sep_fwd_plus_bwd_onepass_ms": mom("combined_sep_forward_backward"),
                "fused_fwd_plus_bwd_onepass_ms": mom("combined_fused_forward_backward"),
            }
            per_cam.append(res)
            print(f"[{scene} cam{cam_idx}] sep_total={t_sep_total:.3f}ms "
                  f"fused_total={t_fused_total:.3f}ms S={res['summary']['S_loss']:.1f}x")

        scene_summary = {
            "T_loss_sep_total_ms": float(np.mean([r["summary"]["T_loss_sep_total_ms"]
                                                  for r in per_cam])),
            "T_loss_fused_total_ms": float(np.mean([r["summary"]["T_loss_fused_total_ms"]
                                                    for r in per_cam])),
            "S_loss_mean": float(np.mean([r["summary"]["S_loss"] for r in per_cam])),
        }
        scene_summary["S_loss"] = (scene_summary["T_loss_sep_total_ms"] /
                                   scene_summary["T_loss_fused_total_ms"])
        per_scene[scene] = {
            "state": {"source": meta["state_source"], "sha256": meta["state_sha256"],
                      "iteration": meta["state_iteration"], "N": meta["state_N"]},
            "per_camera": per_cam,
            "scene_summary": scene_summary,
        }
        print(f"[{scene}] SUMMARY S_loss={scene_summary['S_loss']:.1f}x "
              f"sep={scene_summary['T_loss_sep_total_ms']:.3f}ms "
              f"fused={scene_summary['T_loss_fused_total_ms']:.3f}ms")

    record = {
        "phase": "P11 Gate L2 - loss-only timing decomposition",
        "gpu": gpu_rec,
        "protocol": {"warmup": args.warmup, "measurements": args.iters,
                     "reps": args.reps, "timer": "CUDA events",
                     "backward_timing": "autograd.grad on retained graph "
                                        "(pure backward); combined fwd/bwd rows "
                                        "rebuild the graph per call",
                     "renderer_excluded": "detached rendered images with "
                                          "requires_grad_(True)"},
        "per_scene": per_scene,
        "note": "baseline modernization evidence only, NOT a contribution",
    }
    C.dump_json(record, args.out)


if __name__ == "__main__":
    main()
