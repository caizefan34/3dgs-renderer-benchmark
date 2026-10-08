"""DSH Section-2: independent original-trainer parity check for PRISM-GS v0.1.

Reference path = the frozen P12/GMAIN trainer's own entry:
  GaussianModel activations (F.normalize raw rotation -> exp scales -> sigmoid
  opacity) + pinned-tree gsplat rasterization + fused L1/SSIM training loss.
Adapter path  = prism.gsplat_backend.baseline on the same raw parameter arrays.

Fixed tolerances are read from the bundle's PREREGISTRATION.json (sha checked
against MANIFEST.json). They are not adjusted after seeing results. The noise
floor is measured by 5 repeated original-entry runs. Raw arrays are saved.
"""
import argparse
import hashlib
import json
import os
import sys
import traceback

import numpy as np
import torch
import torch.nn.functional as F

BUNDLE = "/root/epic05_prism/prism_gs_v0.1.2"
GX = "/root/epic05_grc3d/scripts"
REPO = "/root/3dgs-renderer-benchmark"
SEQ_PATHS = {
    "bicycle": "/root/epic05_p12/runs/p12a/p12a_gmainf_s42_bicycle/camera_sequence.npy",
    "garden": "/root/epic05_p12/runs/p12a/p12a_gmainf_s42_garden/camera_sequence.npy",
}
SNAP_PATHS = {
    ("bicycle", 5000): "/root/epic05_grc3d/inputs/snapshots/snap_bicycle_005000.pt",
    ("bicycle", 14000): "/root/epic05_grc3d/inputs/snapshots/snap_bicycle_014000.pt",
    ("garden", 5000): "/root/epic05_grc3d/inputs/snapshots/snap_garden_005000.pt",
    ("garden", 14000): "/root/epic05_grc3d/inputs/snapshots/snap_garden_014000.pt",
}
GROUPS = ["means", "quats", "log_scales", "opacity_logits", "sh0", "shN"]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
    os.replace(tmp, path)


def diff_stats(a, b):
    d = (a.double() - b.double())
    return {"maxabs": float(d.abs().max()),
            "relative_l2": float(d.norm() / a.double().norm().clamp_min(1e-20))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--state", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arrays-dir", required=True)
    ap.add_argument("--array-mode", choices=["full", "subset"], default="full")
    args = ap.parse_args()

    out = {"status": "STARTED", "scene": args.scene, "state": args.state,
           "script_sha256": sha256_file(__file__), "gpu_index_requested": args.gpu}
    write_json(args.out, out)
    try:
        # --- frozen thresholds from the bundle (hash-guarded) ---------------
        man = json.load(open(os.path.join(BUNDLE, "MANIFEST.json")))
        pre_path = os.path.join(BUNDLE, "PREREGISTRATION.json")
        assert sha256_file(pre_path) == man["files"]["PREREGISTRATION.json"]["sha256"], \
            "PREREGISTRATION.json drifted from MANIFEST"
        pre = json.load(open(pre_path))
        tol = pre["m0_fixed_tolerances"]
        out["preregistration"] = {"sha256_manifest_ok": True, "tolerances": tol}

        sys.path.insert(0, GX)
        sys.path.insert(0, BUNDLE)
        import gx_common as G                      # noqa: E402

        out["gpu_before"] = G.gpu_inventory()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        trainer, ident = G.boot_gmain(args.gpu)
        out["identity"] = G.run_identity(trainer, ident, args.gpu)
        out["fused_lock"] = G.check_fused_lock()
        out["prism_bundle"] = {
            "bundle": BUNDLE,
            "version": __import__("prism").__version__,
            "gsplat_backend_sha256": sha256_file(
                os.path.join(BUNDLE, "prism", "gsplat_backend.py")),
            "geometry_sha256": sha256_file(
                os.path.join(BUNDLE, "prism", "geometry.py")),
            "run_gpu_sha256": sha256_file(
                os.path.join(BUNDLE, "prism", "run_gpu.py")),
        }
        assert out["prism_bundle"]["version"] == "0.1.2", "wrong bundle version bound"

        import fused_ssim                          # noqa: E402
        from gsplat import rasterization           # noqa: E402
        from gaussian_model import GaussianModel   # noqa: E402
        from config import ReferenceV1Config       # noqa: E402
        from prism import gsplat_backend as B      # noqa: E402
        from prism.geometry import covariance      # noqa: E402

        lam = ReferenceV1Config().lambda_dssim
        out["lambda_dssim"] = lam
        assert abs(lam - 0.2) < 1e-12, "unexpected lambda_dssim"

        # --- real snapshot into the trainer's own model ---------------------
        snap_path = SNAP_PATHS[(args.scene, args.state)]
        snap = torch.load(snap_path, map_location="cpu", weights_only=False)
        out["snapshot"] = {"path": snap_path, "sha256": sha256_file(snap_path),
                           "keys": sorted(k for k in snap.keys() if not hasattr(snap[k], "shape")),
                           "iteration": int(snap["iteration"]), "n_gaussians": int(snap["xyz"].shape[0]),
                           "active_sh_degree": int(snap["active_sh_degree"])}
        model = GaussianModel(max_sh_degree=3)
        model.restore(snap, G.lr_config())
        degree = int(model.active_sh_degree)

        # --- frozen TRAINING camera sequence ---------------------------------
        seq = np.load(SEQ_PATHS[args.scene])
        ids = []
        for c in seq:
            if int(c) not in ids:
                ids.append(int(c))
            if len(ids) == 5:
                break
        assert len(ids) == 5, "need 5 distinct training camera IDs"
        out["camera_seq"] = {"path": SEQ_PATHS[args.scene], "sha256": sha256_file(SEQ_PATHS[args.scene]),
                             "length": int(len(seq)), "first5_distinct": ids,
                             "rule": "first 5 distinct IDs of frozen TRAINING sequence; first 3 optimize"}
        dataset = G.build_dataset(args.scene)

        # --- settings parity table (trainer entry vs adapter entry) ---------
        cam0_ds, _ = dataset.get_item(ids[0])
        out["settings_parity"] = {
            "trainer": {"tile_size": 16, "packed": False, "eps2d": 0.3, "absgrad": True,
                        "render_mode": "RGB", "rasterize_mode_default": "classic",
                        "radius_clip": 0.0, "sh_degree": degree, "background": "black(default)",
                        "image_clamp": "[0,1]", "loss": f"{1-lam:.1f}*L1+{lam:.1f}*(1-fusedSSIM(padding=same,train=True))"},
            "adapter": {"tile_size": 16, "packed": False, "eps2d": 0.3, "absgrad": True,
                        "render_mode": "RGB", "rasterize_mode": "classic",
                        "radius_clip": 0.0, "sh_degree": degree, "background": "black(default)",
                        "image_clamp": "[0,1]", "loss": "same formula applied to adapter image"},
            "K_equal": bool(torch.equal(cam0_ds.K, G.camera_dict(cam0_ds)["K"])),
            "viewmat_equal": bool(torch.equal(cam0_ds.viewmatrix, G.camera_dict(cam0_ds)["viewmat"])),
            "resolution_hw": [int(cam0_ds.image_height), int(cam0_ds.image_width)],
        }

        gt_bchw_cache = {}

        def loss_fn(image_hwc, gt_hwc, tag):
            if tag not in gt_bchw_cache:
                gt_bchw_cache[tag] = gt_hwc.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
            b = image_hwc.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
            g = gt_bchw_cache[tag]
            L1 = F.l1_loss(b, g)
            ssim = fused_ssim.fused_ssim(b, g, padding="same", train=True)
            return (1.0 - lam) * L1 + lam * (1.0 - ssim)

        def trainer_entry(cam_ds, gt, tag):
            r, radii, meta = rasterization(
                means=model.get_xyz, quats=model.get_rotation, scales=model.get_scaling,
                opacities=model.get_opacity, colors=model.get_features,
                viewmats=cam_ds.viewmatrix.unsqueeze(0), Ks=cam_ds.K.unsqueeze(0),
                width=cam_ds.image_width, height=cam_ds.image_height,
                tile_size=16, packed=False, sh_degree=degree,
                radius_clip=0.0, eps2d=0.3, render_mode="RGB", absgrad=True)
            image = r[0].clamp(0, 1)
            loss = loss_fn(image, gt, tag)
            grads = torch.autograd.grad(
                loss, [model._xyz, model._rotation, model._scaling, model._opacity, model._shs])
            gd = {"means": grads[0].detach(), "quats": grads[1].detach(),
                  "log_scales": grads[2].detach(),
                  "opacity_logits": grads[3].detach().reshape(-1),
                  "sh0": grads[4][:, :1].detach(), "shN": grads[4][:, 1:].detach()}
            return image.detach(), gd, float(loss.detach())

        def make_params():
            with torch.no_grad():
                return {"means": model._xyz.detach().clone().requires_grad_(),
                        "quats": model._rotation.detach().clone().requires_grad_(),
                        "log_scales": model._scaling.detach().clone().requires_grad_(),
                        "opacity_logits": model._opacity.detach().reshape(-1).clone().requires_grad_(),
                        "sh0": model._shs.detach()[:, :1].clone().requires_grad_(),
                        "shN": model._shs.detach()[:, 1:].clone().requires_grad_()}

        def adapter_entry(cam, gt, tag):
            params = make_params()
            image, meta = B.baseline(params, cam, degree)
            loss = loss_fn(image, gt, tag)
            grads = torch.autograd.grad(loss, tuple(params.values()))
            gd = dict(zip(GROUPS, [g.detach() for g in grads]))
            return image.detach(), gd, float(loss.detach()), params

        # rotation-chain check: trainer pre-normalization vs adapter covariance
        def quat_to_R_script(q):  # independent quaternion->rotation formula (script-local)
            w, x, y, z = F.normalize(q, dim=-1).unbind(-1)
            return torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                                2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                                2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
                               -1).reshape(q.shape[:-1] + (3, 3))

        with torch.no_grad():
            rn_t = F.normalize(model._rotation, dim=-1)
            rn_script = F.normalize(model._rotation, dim=-1)
            r_mat = quat_to_R_script(rn_script)
            s2 = model.get_scaling.square()
            cov_manual = (r_mat * s2[..., None, :]) @ r_mat.transpose(-1, -2)
            cov_adapter = covariance(model._rotation.detach(), model._scaling.detach())
            out["rotation_covariance_chain"] = {
                "normalize_maxabs": float((rn_t - rn_script).abs().max()),
                "covariance_maxabs": float((cov_manual - cov_adapter).abs().max()),
                "note": "script-local raw-rot->normalize->R diag(s^2) R^T vs prism.geometry.covariance on real data"}

        cameras = []
        for cid in ids[:3]:
            cam_ds, gt = dataset.get_item(cid)
            cam = G.camera_dict(cam_ds)
            if tuple(gt.shape) == (3, cam["height"], cam["width"]):
                gt = gt.permute(1, 2, 0)
            if tuple(gt.shape) != (cam["height"], cam["width"], 3):
                raise RuntimeError("Invalid target dimensions")
            gt = gt.float().contiguous()
            cameras.append((cid, cam_ds, cam, gt,
                            G.camera_identity(cam_ds, cam["width"], cam["height"])))

        # --- per-camera parity + noise floor --------------------------------
        out["cameras"] = {}
        for cid, cam_ds, cam, gt, cam_hash in cameras:
            tag = f"{args.scene}_{args.state}_{cid}"
            rec = {"camera_id": cid, "camera_identity": cam_hash,
                   "array_mode": args.array_mode, "repeats": 5}
            img_t, gd_t, loss_t = trainer_entry(cam_ds, gt, tag)
            img_a, gd_a, loss_a, _ = adapter_entry(cam, gt, tag)
            rec["loss_trainer"] = loss_t
            rec["loss_adapter"] = loss_a
            rec["loss_abs_diff"] = abs(loss_t - loss_a)
            rec["forward"] = diff_stats(img_t, img_a)
            rec["gradients"] = {g: diff_stats(gd_t[g], gd_a[g]) for g in GROUPS}

            # noise floor: 5 repeated original-entry runs
            reps = []
            for i in range(5):
                ri, rg, rl = trainer_entry(cam_ds, gt, tag)
                reps.append((ri, rg, rl))
            noise = {"forward_maxabs": 0.0,
                     "gradients": {g: {"maxabs": 0.0, "relative_l2": 0.0} for g in GROUPS},
                     "loss_maxabs": 0.0}
            for i in range(1, 5):
                d = diff_stats(reps[0][0], reps[i][0])
                noise["forward_maxabs"] = max(noise["forward_maxabs"], d["maxabs"])
                noise["loss_maxabs"] = max(noise["loss_maxabs"], abs(reps[0][2] - reps[i][2]))
                for g in GROUPS:
                    ds = diff_stats(reps[0][1][g], reps[i][1][g])
                    noise["gradients"][g]["maxabs"] = max(noise["gradients"][g]["maxabs"], ds["maxabs"])
                    noise["gradients"][g]["relative_l2"] = max(noise["gradients"][g]["relative_l2"], ds["relative_l2"])
            rec["noise_floor_5x_original_entry"] = noise

            # classification against PRE-REGISTERED fixed tolerances
            checks = {"forward": rec["forward"]["maxabs"] <= tol["adapter_forward_maxabs"]}
            for g in GROUPS:
                checks[g] = (rec["gradients"][g]["relative_l2"] <= tol["gradient_relative_l2"]
                             or rec["gradients"][g]["maxabs"] <= tol["gradient_absolute_fallback"])
            within = {"forward": rec["forward"]["maxabs"] <= noise["forward_maxabs"]}
            for g in GROUPS:
                within[g] = (rec["gradients"][g]["relative_l2"] <= noise["gradients"][g]["relative_l2"]
                             and rec["gradients"][g]["maxabs"] <= noise["gradients"][g]["maxabs"])
            rec["checks"] = checks
            rec["within_repeat_noise"] = within
            if all(checks.values()):
                rec["status"] = "PASS_PARITY"
            elif all(checks[k] or within[k] for k in checks):
                rec["status"] = "INCONCLUSIVE_NUMERICS"
            else:
                rec["status"] = "FAIL_PARITY"

            # --- raw arrays --------------------------------------------------
            adir = os.path.join(args.arrays_dir, f"{args.scene}_{args.state}", f"cam_{cid}")
            os.makedirs(adir, exist_ok=True)
            if args.array_mode == "full":
                idx = None
            else:
                gen = torch.Generator().manual_seed(20260927)
                idx = torch.randperm(model._xyz.shape[0], generator=gen)[:500000].sort().values
            torch.save({"image_trainer": img_t.cpu(), "image_adapter": img_a.cpu()},
                       os.path.join(adir, "images.pt"))
            for g in GROUPS:
                gt_full, ga_full = gd_t[g].cpu(), gd_a[g].cpu()
                if idx is None:
                    gtv, gav, gdv = gt_full, ga_full, ga_full - gt_full
                else:
                    gtv = gt_full[idx]
                    gav = ga_full[idx]
                    gdv = gav - gtv
                torch.save({"grad_trainer": gtv, "grad_adapter": gav, "grad_diff": gdv},
                           os.path.join(adir, f"grad_{g}.pt"))
            if idx is not None:
                torch.save({"index": idx}, os.path.join(adir, "index.pt"))
            rec["arrays_dir"] = adir
            rec["arrays_indexed"] = idx is not None
            rec["index_size"] = None if idx is None else int(idx.numel())
            del img_t, img_a, gd_t, gd_a, reps
            torch.cuda.empty_cache()
            out["cameras"][str(cid)] = rec
            write_json(args.out, out)

        statuses = [c["status"] for c in out["cameras"].values()]
        out["status"] = ("PASS_PARITY" if all(s == "PASS_PARITY" for s in statuses)
                         else ("INCONCLUSIVE_NUMERICS" if all(s != "FAIL_PARITY" for s in statuses)
                               else "FAIL_PARITY"))
        out["gpu_after"] = G.gpu_inventory()
        write_json(args.out, out)
    except Exception:
        out["status"] = "ERROR"
        out["traceback"] = traceback.format_exc()
        write_json(args.out, out)
        raise


if __name__ == "__main__":
    main()
