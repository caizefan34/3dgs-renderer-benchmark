#!/usr/bin/env python3
"""GRC3D M0 correctness audit on EPIC-05 (DSH_EXECUTION.md section 2).

Modes:
  --ckpt PATH : restore a real GMAIN_FUSED checkpoint through the existing
                trainer/model.restore path and audit the raw-parameter export
                mapping, then run renderer / SSIM / halo parity on that state.
  --bundle P  : run the same renderer / SSIM / halo parity on a canonical M1
                bundle (bundle + target + camera are the inputs), no export
                audit.

Pre-registered CUDA fp32 thresholds (DSH_EXECUTION.md line 48):
  loss abs <= 1e-6; image maxabs <= 1e-5; per-group gradient relative L2 <= 1e-4
  (per-group maxabs <= 1e-7 when the reference gradient norm < 1e-8).
No threshold is relaxed after measurement.
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gx_common as G  # noqa: E402

passing = {}

OUT = {
    "tag": None, "mode": None, "identity": {}, "mapping_audit": {},
    "renderer_parity": {}, "ssim_parity": {}, "p1_corrector": {}, "halo": {},
    "thresholds": {"loss_abs": 1e-6, "image_maxabs": 1e-5,
                   "group_grad_rel_l2": 1e-4, "group_grad_maxabs_when_small_norm": 1e-7},
    "checks": {},
}


def check(name, ok, detail=None):
    OUT["checks"][name] = {"pass": bool(ok), "detail": detail}
    print(f"  [check] {name}: {'PASS' if ok else 'FAIL'} {detail if detail else ''}",
          flush=True)
    return bool(ok)


def rel_l2(a, b):
    num = (a - b).double().norm().item()
    den = a.double().norm().item()
    return num, den, (num / den if den >= 1e-8 else None), (a - b).abs().max().item()


def group_report(ref, test):
    return {"reference_norm": ref.double().norm().item(),
            "diff_norm": (ref - test).double().norm().item(),
            "relative_l2": (rel_l2(ref, test)[2]),
            "maxabs": (ref - test).abs().max().item()}


def grad_ok(ref, test):
    num, den, rl2, mx = rel_l2(ref, test)
    if den < 1e-8:
        return mx <= 1e-7, {"relative_l2": None, "maxabs": mx, "note": "small reference norm"}
    return rl2 <= 1e-4, {"relative_l2": rl2, "maxabs": mx}


def make_leaves(params):
    out = {}
    for k, v in params.items():
        out[k] = v.detach().to("cuda").float().clone().requires_grad_(True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="bicycle")
    ap.add_argument("--ckpt")
    ap.add_argument("--bundle")
    ap.add_argument("--camera-seq", default=None,
                    help="frozen camera_sequence.npy used to pre-select the camera")
    ap.add_argument("--camera-index", type=int, default=-1)
    ap.add_argument("--gpu", type=int, default=6)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert (args.ckpt is None) != (args.bundle is None), "exactly one input"
    OUT["tag"] = args.tag or os.path.basename(args.ckpt or args.bundle)
    OUT["mode"] = "ckpt" if args.ckpt else "bundle"

    t, ident = G.boot_gmain(args.gpu)
    fused_ident = G.check_fused_lock()
    G.add_grc3d_path()
    from grc3d import core, renderers

    print(f"[m0] boot ok arm=gmain so={ident['so_sha256'][:16]}... "
          f"fused_so={fused_ident['so_sha256'][:16]}...", flush=True)

    dataset = G.build_dataset(args.scene)
    seq = None
    if args.camera_seq and os.path.exists(args.camera_seq):
        import numpy as np
        seq = np.load(args.camera_seq)
    cam_idx = args.camera_index
    cam_rule = "explicit"
    if cam_idx < 0:
        assert seq is not None, "need --camera-seq or --camera-index"
        cam_idx = int(seq[0])
        cam_rule = "first index of frozen camera_sequence (pre-registered)"
    cam, gt = dataset.get_item(cam_idx)
    H_, W_ = int(cam.image_height), int(cam.image_width)

    OUT["identity"].update(G.run_identity(t, ident, args.gpu))
    OUT["identity"].update({
        "scene": args.scene, "resolution_hw": [H_, W_], "eps2d": 0.3,
        "background": "black", "tile_size": 16, "packed": False,
        "camera_index": cam_idx, "camera_rule": cam_rule,
        "camera_hash": G.camera_identity(cam, W_, H_),
        "camera_json_sha256": G.sha256_file(
            os.path.join(G.REPO, "data", "official", "mipnerf360", args.scene, "cameras.json")),
        "gt_file": str(dataset._valid_image_paths[cam_idx]),
        "gt_file_sha256": G.sha256_file(dataset._valid_image_paths[cam_idx]),
        "fused_loss_identity": fused_ident,
    })

    camera = G.camera_dict(cam)
    ch, cw = H_ // 4, W_ // 4
    target = gt.detach().float()

    if args.ckpt:
        ckpt_sha = G.sha256_file(args.ckpt)
        model, ck = G.restore_model(args.ckpt)
        params = {k: v.detach().cpu().float() for k, v in
                  G.bundle_params_from_model(model).items()}
        degree = int(model.active_sh_degree)
        OUT["identity"].update({"checkpoint": args.ckpt, "checkpoint_sha256": ckpt_sha,
                                "checkpoint_iteration": int(ck["iteration"]),
                                "checkpoint_arm": ck["arm"], "n_gaussians": int(model._xyz.shape[0]),
                                "active_sh_degree": degree, "seed": 42})
        a = OUT["mapping_audit"]
        a["model_source"] = "baseline/reference_v1/gaussian_model.py (EPIC repo, read-only)"
        a["model_attribute_map"] = {
            "means<-_xyz": "gaussian_model.py L85-86 get_xyz returns _xyz (raw)",
            "quats<-_rotation": "L81-82 get_rotation = F.normalize(_rotation); raw _rotation exported",
            "log_scales<-_scaling": "L76-78 get_scaling = exp(_scaling); raw log exported",
            "opacity_logits<-_opacity": "L88-90 get_opacity = sigmoid(_opacity).squeeze(-1); raw logit exported [N]",
            "sh0<-_shs[:, :1]": "L92-96 get_features slices _shs to (active_sh_degree+1)^2; raw full tensor exported",
            "shN<-_shs[:, 1:]": "same tensor; cat(sh0, shN) == _shs",
        }
        a["export_py_deviation"] = {
            "finding": ("grc3d/export.py snapshot() targets graphdeco attribute names "
                        "_features_dc/_features_rest; the EPIC reference_v1 GaussianModel "
                        "has no such attributes (single _shs [N,16,3] tensor). The bundle "
                        "mapping was adapted explicitly for this model after reading its "
                        "source; the package file was not modified."),
            "adapted": {"sh0": "_shs[:, :1]", "shN": "_shs[:, 1:]"},
        }
        # elementwise mapping equality against the native checkpoint (device-agnostic)
        eq = {}
        _pc = {k: v.detach().cpu() for k, v in params.items()}
        eq["means==ckpt.xyz"] = bool(torch.equal(_pc["means"], ck["xyz"].cpu()))
        eq["quats==ckpt.rotation"] = bool(torch.equal(_pc["quats"], ck["rotation"].cpu()))
        eq["log_scales==ckpt.scaling"] = bool(torch.equal(_pc["log_scales"], ck["scaling"].cpu()))
        eq["opacity_logits==ckpt.opacity"] = bool(torch.equal(
            _pc["opacity_logits"], ck["opacity"].cpu().reshape(-1)))
        eq["sh0==ckpt.shs[:, :1]"] = bool(torch.equal(_pc["sh0"], ck["shs"].cpu()[:, :1]))
        eq["shN==ckpt.shs[:, 1:]"] = bool(torch.equal(_pc["shN"], ck["shs"].cpu()[:, 1:]))
        a["checkpoint_field_equality"] = eq
        # activation semantics on the live model
        act = {}
        act["exp(log_scales)==get_scaling"] = bool(torch.equal(
            params["log_scales"].cuda().exp(), model.get_scaling))
        act["sigmoid(opacity_logits)==get_opacity"] = bool(torch.equal(
            params["opacity_logits"].cuda().sigmoid(), model.get_opacity))
        act["normalize(quats)==get_rotation"] = bool(torch.equal(
            F.normalize(params["quats"].cuda(), dim=-1), model.get_rotation))
        act["sh0|shN==get_features(full)"] = bool(torch.equal(
            torch.cat([params["sh0"].cuda(), params["shN"].cuda()], 1), model._shs))
        with torch.no_grad():
            act["log_scales_stats"] = {"min": float(params["log_scales"].min()),
                                       "max": float(params["log_scales"].max())}
            act["opacity_logits_stats"] = {"min": float(params["opacity_logits"].min()),
                                           "max": float(params["opacity_logits"].max())}
            act["n_gaussians"] = int(params["means"].shape[0])
            act["sh_tensor_shape"] = list(params["sh0"].shape[1:]), list(params["shN"].shape[1:])
        a["activation_semantics"] = act
        check("mapping.checkpoint_fields_equal", all(eq.values()), eq)
        check("mapping.activation_semantics", all(v for k, v in act.items() if isinstance(v, bool)), act)
        del model
        torch.cuda.empty_cache()
    else:
        data = torch.load(args.bundle, map_location="cpu", weights_only=True)
        params = {k: v.detach().float() for k, v in data["params"].items()}
        degree = int(data["sh_degree"])
        OUT["identity"].update({"bundle": args.bundle, "bundle_sha256": G.sha256_file(args.bundle),
                                "n_gaussians": int(params["means"].shape[0]),
                                "active_sh_degree": degree,
                                "bundle_metadata": {k: v for k, v in data.get("metadata", {}).items()
                                                    if isinstance(v, (str, int, float, bool))}})

    import gsplat
    print(f"[m0] state N={params['means'].shape[0]:,} degree={degree} HxW={H_}x{W_}", flush=True)

    # ---------------- B. fine renderer parity (baseline vs package) ---------
    V = torch.randn((H_, W_, 3), generator=torch.Generator(device="cuda").manual_seed(4200),
                    device="cuda")
    vm, K = camera["viewmat"], camera["K"]

    def baseline_render(leaves, absgrad=True):
        r, alpha, meta = gsplat.rasterization(
            means=leaves["means"], quats=F.normalize(leaves["quats"], dim=-1),
            scales=leaves["log_scales"].exp(), opacities=leaves["opacity_logits"].sigmoid(),
            colors=torch.cat([leaves["sh0"], leaves["shN"]], 1),
            viewmats=vm[None], Ks=K[None], width=W_, height=H_, tile_size=16, packed=False,
            sh_degree=degree, radius_clip=0.0, eps2d=0.3, render_mode="RGB", absgrad=absgrad)
        return r[0].clamp(0, 1), meta

    def package_render(leaves):
        return renderers.render_gsplat(leaves, camera, H_, W_, degree, eps2d=0.3)

    lb = make_leaves(params)
    pb, meta_b = baseline_render(lb, absgrad=True)
    gb = torch.autograd.grad((pb * V).sum(), tuple(lb.values()))
    lb_ = make_leaves(params)
    pp = package_render(lb_)
    gp = torch.autograd.grad((pp * V).sum(), tuple(lb_.values()))

    rp = OUT["renderer_parity"]
    rp["forward"] = {"maxabs": (pb - pp).abs().max().item(),
                     "meanabs": (pb - pp).abs().mean().item(),
                     "sum_abs": (pb - pp).abs().sum().item()}
    rp["forward"]["pass"] = rp["forward"]["maxabs"] <= 1e-5
    rp["groups"] = {}
    all_ok = True
    for k, g_ref, g_tst in zip(lb.keys(), gb, gp):
        ok, det = grad_ok(g_ref, g_tst)
        rp["groups"][k] = dict(det, reference_norm=g_ref.double().norm().item(),
                               pass_=bool(ok))
        all_ok &= ok
    rp["checks"] = {"forward_maxabs<=1e-5": rp["forward"]["pass"],
                    "group_grad_rel_l2<=1e-4": bool(all_ok)}
    rp["note"] = ("package renderer runs absgrad=False (diagnostic switch); baseline runs "
                  "absgrad=True. Cotangent V is seeded random, same for both.")
    check("renderer.forward_image", rp["forward"]["pass"], rp["forward"])
    check("renderer.group_gradients", all_ok, {k: v["relative_l2"] for k, v in rp["groups"].items()})

    # absgrad invariance: baseline with absgrad=False must give identical grads
    lb2 = make_leaves(params)
    pb2, _ = baseline_render(lb2, absgrad=False)
    gb2 = torch.autograd.grad((pb2 * V).sum(), tuple(lb2.values()))
    rp["absgrad_invariance"] = {"forward_maxabs": (pb - pb2).abs().max().item(),
                                "groups": {}}
    inv_ok = True
    for k, g_ref, g_tst in zip(lb.keys(), gb, gb2):
        num, den, rl2, mx = rel_l2(g_ref, g_tst)
        rp["absgrad_invariance"]["groups"][k] = {"relative_l2": rl2, "maxabs": mx}
        inv_ok &= (rl2 is not None and rl2 <= 1e-6) or mx <= 1e-9
    check("renderer.absgrad_invariance", inv_ok,
          {k: v["relative_l2"] for k, v in rp["absgrad_invariance"]["groups"].items()})

    # normalization-order probe: package fed pre-normalized quats
    lb3 = make_leaves(params)
    lb3["quats"] = F.normalize(lb3["quats"], dim=-1).requires_grad_(True)
    pp3 = package_render(lb3)
    rp["quats_pre_normalized"] = {"forward_maxabs": (pp - pp3).abs().max().item()}
    del lb, lb_, lb2, lb3, pb2, pp3, meta_b
    torch.cuda.empty_cache()

    # ---------------- C. SSIM parity (package separable vs fused) -----------
    pred = pb.detach().clone().requires_grad_(True)
    lf = G.fused_loss(pred, target)
    gf = torch.autograd.grad(lf, pred, retain_graph=True)[0]
    lp = core.block_losses(pred, target, core.blocks(H_, W_, 64)).sum()
    gp_img = torch.autograd.grad(lp, pred)[0]
    sp = OUT["ssim_parity"]
    sp["scalar"] = {"fused": lf.item(), "package_loss_map": lp.item(),
                    "abs_diff": abs(lf.item() - lp.item()),
                    "pass": abs(lf.item() - lp.item()) <= 1e-6}
    num, den, rl2, mx = rel_l2(gf, gp_img)
    sp["image_gradient"] = {"relative_l2": rl2, "maxabs": mx,
                            "reference_norm": gf.double().norm().item(),
                            "maxabs<=1e-5": mx <= 1e-5}
    check("ssim.scalar_loss", sp["scalar"]["pass"], sp["scalar"])
    check("ssim.image_gradient", mx <= 1e-5, sp["image_gradient"])

    # model gradients under each loss (baseline renderer path both times)
    lf1 = make_leaves(params)
    pf1, _ = baseline_render(lf1)
    lossf = G.fused_loss(pf1, target)
    gf1 = torch.autograd.grad(lossf, tuple(lf1.values()))
    lf2 = make_leaves(params)
    pf2, _ = baseline_render(lf2)
    lossp = core.block_losses(pf2, target, core.blocks(H_, W_, 64)).sum()
    gp2 = torch.autograd.grad(lossp, tuple(lf2.values()))
    sp["model_gradients"] = {}
    mg_ok = True
    for k, g_ref, g_tst in zip(lf1.keys(), gf1, gp2):
        ok, det = grad_ok(g_ref, g_tst)
        sp["model_gradients"][k] = dict(det, reference_norm=g_ref.double().norm().item(),
                                       pass_=bool(ok))
        mg_ok &= ok
    check("ssim.model_gradients", mg_ok,
          {k: v["relative_l2"] for k, v in sp["model_gradients"].items()})
    del lf1, lf2, pf1, pf2, gf1, gp2
    torch.cuda.empty_cache()

    # ---------------- D. p=1 corrector vs dense -----------------------------
    lz = make_leaves(params)
    with torch.no_grad():
        frozen = {k: v.detach().to("cuda").float() for k, v in params.items()}
        anchor = core.Anchor.capture(
            renderers.render_gsplat(frozen, camera, H_, W_, degree, eps2d=0.3),
            renderers.render_gsplat(frozen, camera, ch, cw, degree, eps2d=0.3))
    partition = core.blocks(H_, W_, 64)
    obj, info = core.reference_step(
        lambda: renderers.render_gsplat(lz, camera, H_, W_, degree, eps2d=0.3),
        renderers.render_gsplat(lz, camera, ch, cw, degree, eps2d=0.3),
        target, anchor, partition, 1.0)
    dense = core.block_losses(
        renderers.render_gsplat(lz, camera, H_, W_, degree, eps2d=0.3), target, partition).sum()
    g_obj = torch.autograd.grad(obj, tuple(lz.values()))
    g_dense = torch.autograd.grad(dense, tuple(lz.values()))
    pc = OUT["p1_corrector"] = {"loss_abs_diff": abs(obj.item() - dense.item()),
                                "n_selected": int(info["selected"].sum().item()),
                                "groups": {}}
    p1_ok = pc["loss_abs_diff"] <= 1e-6
    for k, g_ref, g_tst in zip(lz.keys(), g_dense, g_obj):
        ok, det = grad_ok(g_ref, g_tst)
        pc["groups"][k] = dict(det, pass_=bool(ok))
        p1_ok &= ok
    check("p1_corrector.vs_dense", p1_ok, {"loss_abs_diff": pc["loss_abs_diff"]})
    del lz
    torch.cuda.empty_cache()

    # ---------------- E. halo checks on the real image ----------------------
    parent = pb.detach().clone().requires_grad_(True)
    parts = core.block_losses(parent, target, partition)
    n_rows, n_cols = (H_ + 63) // 64, (W_ + 63) // 64
    idx = lambda r, c: r * n_cols + c
    subset = [idx(0, 0), idx(0, n_cols - 1), idx(n_rows - 1, 0), idx(n_rows - 1, n_cols - 1),
              idx(0, n_cols // 2), idx(n_rows // 2, 0),
              idx(n_rows // 2, n_cols - 1), idx(n_rows - 1, n_cols // 2),
              idx(n_rows - 1, 7), idx(n_rows - 1, n_cols - 8),
              idx(n_rows // 3, n_cols // 3), idx(2 * n_rows // 3, 2 * n_cols // 3)]
    g = torch.Generator().manual_seed(0)
    subset += torch.randperm(len(partition), generator=g)[:8].tolist()
    subset = sorted(set(subset))
    halo = OUT["halo"] = {"n_blocks": len(partition), "rows": n_rows, "cols": n_cols,
                          "partial_last_row_px": H_ - (n_rows - 1) * 64,
                          "all_block_scalar_maxabs": 0.0, "grad_subset": {}}
    worst = 0.0
    for i, b in enumerate(partition):
        s = b.halo(H_, W_)
        local = core.patch_loss(parent[s.y0:s.y1, s.x0:s.x1], target[s.y0:s.y1, s.x0:s.x1],
                                b, s, H_, W_)
        worst = max(worst, abs(local.item() - parts[i].item()))
    halo["all_block_scalar_maxabs"] = worst
    gok = True
    for i in subset:
        b = partition[i]
        s = b.halo(H_, W_)
        local = core.patch_loss(parent[s.y0:s.y1, s.x0:s.x1], target[s.y0:s.y1, s.x0:s.x1],
                                b, s, H_, W_)
        g1 = torch.autograd.grad(local, parent, retain_graph=False)[0]
        g2 = torch.autograd.grad(parts[i], parent, retain_graph=True)[0]
        num, den, rl2, mx = rel_l2(g2, g1)
        halo["grad_subset"][f"block_{i}"] = {"relative_l2": rl2, "maxabs": mx,
                                             "is_edge": i in (0, n_cols - 1, (n_rows - 1) * n_cols,
                                                              n_rows * n_cols - 1)}
        gok &= (mx <= 1e-5)
        del g1, g2
    check("halo.all_block_scalars<=1e-6", worst <= 1e-6, {"maxabs": worst})
    check("halo.gradients<=1e-5", gok, {"n_subset": len(subset)})

    OUT["cuda_memory_peak_gb"] = torch.cuda.max_memory_allocated() / (2 ** 30)
    OUT["verdict_M0"] = "PASS" if all(c["pass"] for c in OUT["checks"].values()) else "FAIL"
    print(f"[m0] verdict: {OUT['verdict_M0']}", flush=True)
    G.write_json(args.out, OUT)
    print(f"[m0] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
