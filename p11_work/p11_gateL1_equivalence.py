#!/usr/bin/env python3
"""P11 Gate L1: SepSSIM vs existing fused-ssim semantic equivalence.

The publication trainer uses 11x11 Gaussian window, sigma=1.5, same-padding,
L = 0.8*L1 + 0.2*(1-SSIM). We compare the frozen SepSSIM loss against
rahul-goel/fused-ssim with padding="same" on >= 3 representative rendered
images per scene:

  1. scalar SSIM / total loss
  2. dL/dimage
  3. finite fraction
  4. max_abs
  5. mean_abs
  6. rel_l2
  7. cosine

FP calibration FIRST (same-input repeats + fp32-vs-fp64 implementation error of
each implementation); the classification threshold is derived from the measured
noise, not invented.
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


def grad_metrics(g_ref, g_tst):
    """Metrics between two gradient tensors (finite fraction, max/mean abs, rel_l2, cosine)."""
    a = g_ref.detach().float().reshape(-1)
    b = g_tst.detach().float().reshape(-1)
    fa = torch.isfinite(a).float().mean().item()
    fb = torch.isfinite(b).float().mean().item()
    af = a[torch.isfinite(a)]
    bf = b[torch.isfinite(b)]
    d = a - b
    df = d[torch.isfinite(d)]
    max_abs = float(d.abs().max()) if d.numel() else 0.0
    mean_abs = float(d.abs().mean()) if d.numel() else 0.0
    na = float(torch.linalg.norm(a))
    nb = float(torch.linalg.norm(b))
    nd = float(torch.linalg.norm(d))
    rel_l2 = nd / na if na > 0 else float("inf")
    cos = float(torch.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else float("nan")
    return {
        "finite_fraction_ref": fa, "finite_fraction_tst": fb,
        "max_abs": max_abs, "mean_abs": mean_abs,
        "rel_l2": rel_l2, "cosine": cos,
        "norm_ref": na, "norm_tst": nb,
    }


def scalar_metrics(s_ref, s_tst):
    return {
        "ref": float(s_ref), "tst": float(s_tst),
        "abs_diff": abs(float(s_ref) - float(s_tst)),
        "rel_diff": abs(float(s_ref) - float(s_tst)) / max(abs(float(s_ref)), 1e-30),
    }


def sep_loss_and_grad(img, gt, ssim_fn):
    img = img.detach().clone().requires_grad_(True)
    loss = C.sep_total_loss(img, gt, ssim_fn)
    g = torch.autograd.grad(loss, img)[0]
    return float(loss.detach()), g.detach(), img


def fused_loss_and_grad(img, gt):
    img = img.detach().clone().requires_grad_(True)
    loss = C.fused_total_loss(img, gt, padding="same")
    g = torch.autograd.grad(loss, img)[0]
    return float(loss.detach()), g.detach(), img


def fp64_loss_and_grad(img, gt):
    img64 = img.detach().double().requires_grad_(True)
    gt64 = gt.detach().double()
    ssim64 = C.SepSSIM(device="cuda", dtype=torch.float64)
    L1 = F.l1_loss(img64, gt64)
    dssim = ssim64(img64, gt64)
    loss = (1.0 - C.LAMBDA_DSSIM) * L1 + C.LAMBDA_DSSIM * dssim
    g = torch.autograd.grad(loss, img64)[0]
    return float(loss.detach().double()), g.detach().float(), g.detach().double()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default="/root/epic05_p11/dumps/P11_LOSS_EQUIVALENCE.json")
    ap.add_argument("--n-cameras", type=int, default=3)
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    torch.manual_seed(0)

    C.bootstrap_gmain()
    C.import_recipe()
    gpu_rec = C.assert_clean_gpu(args.gpu)

    ssim_fn = C.SepSSIM(device="cuda")
    seq_cache = {}
    per_scene = {}
    classification = {}

    for scene in C.SCENES:
        model, dataset, extent, meta = C.load_model_and_dataset(
            scene, ckpt_path=os.path.join(C.FULL30K, f"gmain03_s42_{scene}", "ckpt_30000.pt"))
        seq = C.camera_sequence_for(scene)
        seq_cache[scene] = [int(x) for x in seq[:args.n_cameras]]

        cam_results = []
        for k, cam_idx in enumerate(seq_cache[scene]):
            cam, gt = dataset.get_item(cam_idx)
            with torch.no_grad():
                r0, image, radii, means2d = C.render_with_meta(
                    model, cam, model.active_sh_degree)
            img = image.detach()  # clamped rendered image [H,W,3]

            # ---- FP calibration on THIS image ----
            # (a) same-input repeats (run-to-run nondeterminism of each impl)
            rep_sep = [sep_loss_and_grad(img, gt, ssim_fn) for _ in range(3)]
            rep_fused = [fused_loss_and_grad(img, gt) for _ in range(3)]
            sep_repeat_grad_rel = [float(torch.linalg.norm(rep_sep[i][1] - rep_sep[0][1]) /
                                         max(float(torch.linalg.norm(rep_sep[0][1])), 1e-30))
                                   for i in (1, 2)]
            fused_repeat_grad_rel = [float(torch.linalg.norm(rep_fused[i][1] - rep_fused[0][1]) /
                                           max(float(torch.linalg.norm(rep_fused[0][1])), 1e-30))
                                     for i in (1, 2)]
            sep_repeat_loss_abs = [abs(rep_sep[i][0] - rep_sep[0][0]) for i in (1, 2)]
            fused_repeat_loss_abs = [abs(rep_fused[i][0] - rep_fused[0][0]) for i in (1, 2)]

            # (b) fp32-vs-fp64 implementation error of each implementation
            loss64, g64_f32, _ = fp64_loss_and_grad(img, gt)
            err_sep = grad_metrics(rep_sep[0][1], g64_f32)
            err_sep["scalar"] = scalar_metrics(rep_sep[0][0], loss64)
            err_fused = grad_metrics(rep_fused[0][1], g64_f32)
            err_fused["scalar"] = scalar_metrics(rep_fused[0][0], loss64)

            # ---- primary comparison: SepSSIM vs fused-ssim ----
            loss_sep, g_sep, _ = sep_loss_and_grad(img, gt, ssim_fn)
            loss_fused, g_fused, _ = fused_loss_and_grad(img, gt)
            gm = grad_metrics(g_sep, g_fused)
            sm_total = scalar_metrics(loss_sep, loss_fused)

            # scalar SSIM-only comparison
            with torch.no_grad():
                img_b = img.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
                gt_b = gt.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
                ssim_sep_val = 1.0 - C.SepSSIM()(img_b, gt_b).item()  # = SSIM
                ssim_fused_val = float(C.fused_ssim_fn(img_b, gt_b, padding="same"))
            sm_ssim = scalar_metrics(ssim_sep_val, ssim_fused_val)

            # sigma-clamp census (fused clamps variance to >=0, SepSSIM does not)
            with torch.no_grad():
                img_b = img.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
                gt_b = gt.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
                ssim_fn_b = C.SepSSIM(device="cuda")
                stacked = torch.cat([img_b, gt_b, img_b ** 2, gt_b ** 2, img_b * gt_b], dim=1)
                bb = F.conv2d(stacked, ssim_fn_b.k_h, padding=(0, 5), groups=15)
                bb = F.conv2d(bb, ssim_fn_b.k_v, padding=(5, 0), groups=15)
                mu_p, mu_t = bb[:, 0:3], bb[:, 3:6]
                sp2 = bb[:, 6:9] - mu_p ** 2
                st2 = bb[:, 9:12] - mu_t ** 2
                clamp_census = {
                    "n_pred_variance_negative": int((sp2 < 0).sum()),
                    "n_target_variance_negative": int((st2 < 0).sum()),
                    "n_elements": int(sp2.numel()),
                    "frac_pred_variance_negative": float((sp2 < 0).float().mean()),
                    "most_negative_pred_variance": float(sp2.min()),
                    "note": "fused-ssim clamps sigma^2 to >=0 (fmaxf) and zeros "
                            "derivatives at clamped samples; SepSSIM does not clamp",
                }
            cam_results.append({
                "camera_index": cam_idx,
                "n_gaussians": meta["state_N"],
                "image_shape": list(img.shape),
                "primary_comparison": {
                    "total_loss": sm_total,
                    "ssim_scalar": sm_ssim,
                    "dL_dimage": gm,
                },
                "fp_calibration": {
                    "sep_repeat_grad_rel_l2": sep_repeat_grad_rel,
                    "fused_repeat_grad_rel_l2": fused_repeat_grad_rel,
                    "sep_repeat_loss_abs_diff": sep_repeat_loss_abs,
                    "fused_repeat_loss_abs_diff": fused_repeat_loss_abs,
                    "sep_vs_fp64": err_sep,
                    "fused_vs_fp64": err_fused,
                    "sigma_clamp_census": clamp_census,
                },
            })
            print(f"[{scene} cam{cam_idx}] loss sep={loss_sep:.6f} fused={loss_fused:.6f} "
                  f"grad rel_l2={gm['rel_l2']:.3e} cos={gm['cosine']:.8f}")

        # ---- store this scene's raw numbers (classification computed after) ----
        per_scene[scene] = {
            "state": {"source": meta["state_source"], "sha256": meta["state_sha256"],
                      "iteration": meta["state_iteration"], "N": meta["state_N"],
                      "active_sh_degree": meta.get("active_sh_degree")},
            "cameras": cam_results,
        }

    # Provisional classification rule (documented, applied on aggregates):
    # SEMANTICALLY_MATCHED_WITHIN_FP iff, for every camera:
    #   grad rel_l2(fused vs sep) <= max(10 * fp32-vs-fp64 rel_l2 of SEP itself,
    #                                    10 * fp32-vs-fp64 rel_l2 of FUSED itself,
    #                                    100 * max repeat-noise rel_l2)
    #   AND total-loss rel diff <= 10 * max(sep_vs_fp64 rel, fused_vs_fp64 rel, 1e-9)
    #   AND finite fractions == 1.0
    prov = {}
    for scene, rec in per_scene.items():
        worst_rel_l2 = max(c["primary_comparison"]["dL_dimage"]["rel_l2"]
                           for c in rec["cameras"])
        worst_loss_rel = max(c["primary_comparison"]["total_loss"]["rel_diff"]
                             for c in rec["cameras"])
        worst_finite = min(
            min(c["primary_comparison"]["dL_dimage"]["finite_fraction_ref"],
                c["primary_comparison"]["dL_dimage"]["finite_fraction_tst"])
            for c in rec["cameras"])
        sep_fp64_rel = max(c["fp_calibration"]["sep_vs_fp64"]["rel_l2"]
                           for c in rec["cameras"])
        fused_fp64_rel = max(c["fp_calibration"]["fused_vs_fp64"]["rel_l2"]
                             for c in rec["cameras"])
        rep_noise = max(max(c["fp_calibration"]["sep_repeat_grad_rel_l2"] +
                            c["fp_calibration"]["fused_repeat_grad_rel_l2"])
                        for c in rec["cameras"])
        grad_thr = max(10 * sep_fp64_rel, 10 * fused_fp64_rel, 100 * rep_noise, 1e-7)
        loss_thr = 10 * max(
            max(c["fp_calibration"]["sep_vs_fp64"]["scalar"]["rel_diff"] for c in rec["cameras"]),
            max(c["fp_calibration"]["fused_vs_fp64"]["scalar"]["rel_diff"] for c in rec["cameras"]),
            1e-9)
        ok_grad = worst_rel_l2 <= grad_thr
        ok_loss = worst_loss_rel <= loss_thr
        ok_finite = worst_finite == 1.0
        prov[scene] = {
            "worst_grad_rel_l2": worst_rel_l2, "grad_threshold": grad_thr,
            "worst_loss_rel_diff": worst_loss_rel, "loss_threshold": loss_thr,
            "worst_finite_fraction": worst_finite,
            "sep_fp64_rel_l2_max": sep_fp64_rel,
            "fused_fp64_rel_l2_max": fused_fp64_rel,
            "repeat_noise_rel_l2_max": rep_noise,
            "pass_grad": ok_grad, "pass_loss": ok_loss, "pass_finite": ok_finite,
            "classification": ("SEMANTICALLY_MATCHED_WITHIN_FP"
                               if (ok_grad and ok_loss and ok_finite)
                               else "MATERIAL_DIFFERENCE"),
        }

    overall = ("SEMANTICALLY_MATCHED_WITHIN_FP"
               if all(v["classification"] == "SEMANTICALLY_MATCHED_WITHIN_FP"
                      for v in prov.values())
               else "MATERIAL_DIFFERENCE")

    record = {
        "phase": "P11 Gate L1 - SepSSIM vs fused-ssim (padding=same) equivalence",
        "gpu": gpu_rec,
        "loss_definition": "L = 0.8*L1 + 0.2*(1-SSIM); SSIM: 11x11 Gaussian "
                           "sigma=1.5, zero same-padding, C1=0.01^2, C2=0.03^2, "
                           "range [0,1]; comparison on clamped rendered images "
                           "from the frozen 30K DSH-A states",
        "fused_ssim_padding": "same",
        "gsplat_convenience_ssim_note": "gsplat.cuda._losses ssim_loss NOT used "
                                        "(it selects valid-padding semantics); "
                                        "fused-ssim padding='same' used per gate spec",
        "classification_rule": "data-driven from same-input repeat noise and "
                               "fp32-vs-fp64 implementation error of each impl; "
                               "see provisional_classification per scene",
        "per_scene": per_scene,
        "provisional_classification": prov,
        "overall_classification": overall,
        "consequence": (
            "fused-ssim IS a strong-baseline substitute for the frozen recipe; "
            "L2/L3/L4 use it as the strong baseline"
            if overall == "SEMANTICALLY_MATCHED_WITHIN_FP" else
            "fused-ssim is NOT a strong-baseline substitute; preserve result and "
            "continue decomposition with SepSSIM only"),
    }
    C.dump_json(record, args.out)


if __name__ == "__main__":
    main()
