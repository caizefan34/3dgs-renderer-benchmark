#!/usr/bin/env python3
"""M0 bundle-mode noise floor: attribute the 14K-bundle parity FAIL.

The ckpt-mode addendum (m0_bicycle_30k_addendum.json) established that the
package renderer is bitwise identical to the strong baseline given the same
raw-quat inputs, and that all gradient excesses sit inside the measured gsplat
backward run-to-run nondeterminism floor. This script runs the same two
sections (noise floor + package-vs-baseline-raw) on the exact M1 bundle format
that feeds the probe, so the bundle states used for kappa carry the same
attribution. No threshold is relaxed; raw numbers only.
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gx_common as G  # noqa: E402

OUT = {"tag": None, "identity": {}, "noise_floor": {}, "package_vs_baseline_raw": {},
       "checks": {}}


def check(name, ok, detail=None):
    OUT["checks"][name] = {"pass": bool(ok), "detail": detail}
    print(f"  [check] {name}: {'PASS' if ok else 'FAIL'} {detail if detail else ''}",
          flush=True)


def rel_l2(a, b):
    num = (a - b).double().norm().item()
    den = a.double().norm().item()
    return {"diff_norm": num, "reference_norm": den,
            "relative_l2": (num / den if den >= 1e-8 else None),
            "maxabs": (a - b).abs().max().item()}


def make_leaves(params, device="cuda"):
    return {k: v.detach().to(device).float().clone().requires_grad_(True)
            for k, v in params.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--camera-index", type=int, required=True)
    ap.add_argument("--camera-seq", required=True)
    ap.add_argument("--scene", default="bicycle")
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t, ident = G.boot_gmain(args.gpu)
    fused_ident = G.check_fused_lock()
    G.add_grc3d_path()
    from grc3d import renderers

    dataset = G.build_dataset(args.scene)
    import numpy as np
    seq = np.load(args.camera_seq)
    # camera contract: bundle camera index must be a member of the frozen sequence
    assert args.camera_index in set(int(v) for v in seq), "camera not in frozen sequence"
    cam, gt = dataset.get_item(args.camera_index)
    H_, W_ = int(cam.image_height), int(cam.image_width)
    camera = G.camera_dict(cam)
    target = gt.detach().float()

    data = torch.load(args.bundle, map_location="cpu", weights_only=True)
    params = {k: v.detach().float() for k, v in data["params"].items()}
    degree = int(data["sh_degree"])
    bc = data["camera"]
    OUT["identity"] = {
        "scene": args.scene, "bundle": args.bundle,
        "bundle_sha256": G.sha256_file(args.bundle),
        "camera_index": args.camera_index,
        "camera_hash": G.camera_identity(cam, W_, H_),
        "bundle_camera_hash": G.camera_identity(
            type("C", (), {"viewmatrix": bc["viewmat"], "K": bc["K"]})(), W_, H_),
        "n_gaussians": int(params["means"].shape[0]), "sh_degree": degree,
        "eps2d": 0.3, "so_sha256": ident["so_sha256"],
        "fused_so_sha256": fused_ident["so_sha256"]}
    cam_match = {
        "viewmat_equal": bool(torch.equal(bc["viewmat"].float(), camera["viewmat"].cpu())),
        "K_equal": bool(torch.equal(bc["K"].float(), camera["K"].cpu())),
        "hw_equal": (int(bc["height"]), int(bc["width"])) == (H_, W_),
        "target_equal_dataset": bool(torch.equal(
            data["target"].float(), target.cpu().float())),
    }
    OUT["identity"]["bundle_camera_matches_dataset"] = cam_match
    check("bundle.camera_and_target_match_dataset",
          all(cam_match.values()), cam_match)
    print(f"[nf] N={OUT['identity']['n_gaussians']:,} degree={degree}", flush=True)

    V = torch.randn((H_, W_, 3), generator=torch.Generator(device="cuda").manual_seed(4200),
                    device="cuda")
    vm, K = camera["viewmat"], camera["K"]

    def render_call(leaves):
        import gsplat
        r, _, meta = gsplat.rasterization(
            means=leaves["means"], quats=leaves["quats"],
            scales=leaves["log_scales"].exp(), opacities=leaves["opacity_logits"].sigmoid(),
            colors=torch.cat([leaves["sh0"], leaves["shN"]], 1),
            viewmats=vm[None], Ks=K[None], width=W_, height=H_, tile_size=16, packed=False,
            sh_degree=degree, radius_clip=0.0, eps2d=0.3, render_mode="RGB",
            absgrad=True, rasterize_mode="classic")
        return r[0].clamp(0, 1), meta

    def pkg_call(leaves):
        return renderers.render_gsplat(leaves, camera, H_, W_, degree, eps2d=0.3)

    # ---------------- B. backward noise floor (identical calls) -------------
    nf = OUT["noise_floor"] = {"n_repeats": args.repeats,
                               "call": "raw_quats absgrad=True",
                               "note": "forward is bitwise identical across repeats; any "
                                       "gradient difference is backward nondeterminism"}
    imgs, reps = [], []
    for rep in range(args.repeats):
        lr_ = make_leaves(params)
        img, _ = render_call(lr_)
        g = torch.autograd.grad((img * V).sum(), tuple(lr_.values()))
        imgs.append(img.detach())
        reps.append({k: gg.detach().clone() for k, gg in zip(lr_.keys(), g)})
        nf.setdefault("repeat_group_norms", []).append(
            {k: float(gg.double().norm().item()) for k, gg in zip(lr_.keys(), g)})
        del lr_, g
        torch.cuda.empty_cache()
    nf["forward_maxabs_across_repeats"] = max((im - imgs[0]).abs().max().item()
                                              for im in imgs[1:])
    nf["pairwise"] = {}
    for i in range(args.repeats):
        for j in range(i + 1, args.repeats):
            nf["pairwise"][f"rep{i}_vs_rep{j}"] = {k: rel_l2(reps[i][k], reps[j][k])
                                                   for k in reps[i]}
    floor = {}
    for k in nf["pairwise"]["rep0_vs_rep1"]:
        vals = [p[k]["relative_l2"] for p in nf["pairwise"].values()]
        mx = [p[k]["maxabs"] for p in nf["pairwise"].values()]
        floor[k] = {"rel_l2_min": min(vals), "rel_l2_max": max(vals),
                    "maxabs_max": max(mx),
                    "rel_l2_note": "None = reference norm < 1e-8" if None in vals else None}
    nf["floor_per_group"] = floor
    del reps, imgs
    torch.cuda.empty_cache()
    print("[nf] backward repeat forward maxabs:", nf["forward_maxabs_across_repeats"],
          flush=True)
    for k, v in floor.items():
        print(f"[nf] floor {k}: rel_l2_max {v['rel_l2_max']}, "
              f"maxabs_max {v['maxabs_max']:.3e}", flush=True)

    # ---------------- C. package vs baseline (identical raw-quat call) ------
    pv = OUT["package_vs_baseline_raw"] = {}
    lb = make_leaves(params)
    img_b, _ = render_call(lb)
    gb = torch.autograd.grad((img_b * V).sum(), tuple(lb.values()))
    lp_ = make_leaves(params)
    img_p = pkg_call(lp_)
    gp = torch.autograd.grad((img_p * V).sum(), tuple(lp_.values()))
    pv["forward_maxabs"] = (img_b - img_p).abs().max().item()
    pv["groups"] = {}
    for k, gb_, gp_ in zip(lb.keys(), gb, gp):
        d = rel_l2(gb_, gp_)
        within = (d["relative_l2"] is None or floor[k]["rel_l2_max"] is None or
                  d["relative_l2"] <= floor[k]["rel_l2_max"] * 1.05)
        pv["groups"][k] = dict(d, floor_rel_l2_max=floor[k]["rel_l2_max"],
                               within_floor=bool(within))
    for k, v in pv["groups"].items():
        print(f"[nf] pkg-vs-baseline(raw) {k}: rel_l2={v['relative_l2']} "
              f"floor={v['floor_rel_l2_max']} within_floor={v['within_floor']}", flush=True)
    check("pkg_vs_baseline.forward_bitwise",
          pv["forward_maxabs"] == 0.0, {"forward_maxabs": pv["forward_maxabs"]})
    check("pkg_vs_baseline.gradients_within_noise_floor",
          all(v["within_floor"] for v in pv["groups"].values()),
          {k: {"rel_l2": v["relative_l2"], "floor": v["floor_rel_l2_max"]}
           for k, v in pv["groups"].items()})

    G.write_json(args.out, OUT)
    print(f"[nf] wrote {args.out}", flush=True)
    print(f"[nf] verdict: {'PASS' if all(c['pass'] for c in OUT['checks'].values()) else 'FAIL'}",
          flush=True)


if __name__ == "__main__":
    main()
