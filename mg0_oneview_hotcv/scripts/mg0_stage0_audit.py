#!/usr/bin/env python3
"""MG0 Stage 0 — identity / restore audit (per scene).

Verifies, against frozen expectations:
- P12 trainer + N1 save-only snapshot trainer (the producer of the checkpoints)
- C0 renderer/core/binaries (c0_ext, core, fused-ssim, HIGS renderer module)
- room/bicycle 15K/20K/25K checkpoints: exact Gaussian tensors, Adam first/
  second moments and group step, SH degree, iteration, num_points
- bit-exact reload / perturbation-restore / real-action-restore state machine
- camera ordering (GTDataset space = cameras.json order = training stream)
- original full-resolution loss (fused .so lock, 1080p, L=0.8*L1+0.2*(1-SSIM))

If any expectation fails: write MG0_BLOCKER.json and exit non-zero (STOP).
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mg0_core as C

# ---------------------------------------------------------------- expectations
EXPECTED_FILES = {
    "p12_trainer": (C.P12_TRAINER,
                    "675b0267f3a1a6a0fa63fadb37c8536481c1c7a6de37a4b7f6ffa98a449050c5"),
    "n1_snapshot_trainer": (C.N1_SNAP_TRAINER,
                            "49a79538d5418cdf795b8a8f9919b2e3f9bc5f7550c7f966b030e3ae14d8dbf4"),
    "c0_ext_so": (C.C0_SO,
                  "d26cccf46c7932bfb5fd44ee05e553af27e17c93eaffb7dacf3197258a513f11"),
    "core_so": (C.CORE_SO,
                "935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e"),
    "fused_ssim_so": ("/root/miniforge3/envs/gsplat/lib/python3.10/site-packages/"
                      "fused_ssim_cuda.cpython-310-x86_64-linux-gnu.so",
                      C.P11_FUSED_SO_SHA256),
    "higs_renderer_py": (os.path.join(C.C0_WT, "gsplat/experimental/render/functional/"
                                      "gaussian_inference.py"),
                         "d599aae3c1864993934d65646bfcb33e07e6046e389f028230752200df0c3a8a"),
    "refv1_gaussian_model_py": (os.path.join(C.REFERENCE_V1_DIR, "gaussian_model.py"),
                                "68731e375013a6f24aa4d81e5986402b1ac50f51cddf20688e358be705b1a60b"),
    "refv1_config_py": (os.path.join(C.REFERENCE_V1_DIR, "config.py"),
                        "ea8e870476a3ba71c0ae9e1123a6fa5b1cf696bad983bdb6535d6fd04b7acc56"),
    "refv1_colmap_reader_py": (os.path.join(C.REFERENCE_V1_DIR, "colmap_reader.py"),
                               "b30a2a345f7f37cdf3087298750d714d7117e1e93ad742e766075be304729b1b"),
    "phase7_dataset_py": (os.path.join(C.PHASE7_DIR, "dataset.py"),
                          "e6d8bfb1034bee102abc176060a8c7783951d253426ea7b5ea27daa16c258402"),
}
PARAM_ORDER = ("xyz", "shs", "opacity", "scaling", "rotation")


def blocker(reason, detail):
    C.json_dump({"verdict": "MG0_BLOCKER", "reason": reason,
                 "detail": detail, "ts": time.strftime("%Y-%m-%d %H:%M:%S")},
                os.path.join(C.ARTIFACTS, "MG0_BLOCKER.json"))
    sys.exit(1)


def audit_checkpoint(scene, it, dataset, fused_loss, smoke):
    ck = C.load_checkpoint(scene, it)
    path = os.path.join(C.SNAP_RUNS[scene], f"ckpt_{it}.pt")
    rec = {"path": path, "bytes": os.path.getsize(path),
           "sha256": C.sha256_file(path), "keys": sorted(ck.keys()),
           "iteration": int(ck["iteration"]), "scene": ck.get("scene"),
           "arm": ck.get("arm"), "num_points": int(ck.get("num_points", -1)),
           "active_sh_degree": int(ck["active_sh_degree"]),
           "N": int(ck["xyz"].shape[0]),
           "spatial_lr_scale": float(ck["spatial_lr_scale"])}

    # ---- exact Gaussian tensors
    tstats = {}
    for k in ("xyz", "shs", "scaling", "rotation", "opacity",
              "max_radii2D", "xyz_gradient_accum", "denom"):
        t = ck[k]
        rec.setdefault("tensor_shapes", {})[k] = list(t.shape)
        rec.setdefault("tensor_dtypes", {})[k] = str(t.dtype)
        rec.setdefault("tensor_finite", {})[k] = bool(torch.isfinite(t).all())
        tstats[k] = {"min": float(t.min()), "max": float(t.max()),
                     "mean": float(t.float().mean()),
                     "std": float(t.float().std()) if t.numel() > 1 else 0.0}
    rec["tensor_stats"] = tstats

    # ---- reload determinism (bit-exact)
    ck2 = C.load_checkpoint(scene, it)
    rec["reload_bit_exact"] = all(torch.equal(ck[k], ck2[k]) for k in
                                  ("xyz", "shs", "scaling", "rotation", "opacity",
                                   "max_radii2D", "xyz_gradient_accum", "denom"))
    del ck2

    # ---- optimizer (Adam first/second moments, group step)
    osd = ck["optimizer_state_dict"]
    pg = osd["param_groups"]
    state = osd["state"]
    rec["optimizer_num_groups"] = len(pg)
    rec["optimizer_groups"] = []
    for gi, g in enumerate(pg):
        pidx = g["params"][0] if isinstance(g["params"], (list, tuple)) else g["params"]
        name = PARAM_ORDER[gi]
        st = state[pidx]
        mom_shape = list(st["exp_avg"].shape)
        rec["optimizer_groups"].append({
            "index": gi, "name": g.get("name"), "param_tensor": name,
            "lr": float(g["lr"]), "betas": [float(b) for b in g["betas"]],
            "eps": float(g["eps"]),
            "weight_decay": float(g.get("weight_decay", 0.0)),
            "step": float(st["step"]),
            "exp_avg_shape": mom_shape,
            "exp_avg_sq_shape": list(st["exp_avg_sq"].shape),
            "shape_matches_param": mom_shape == list(ck[name].shape),
            "exp_avg_finite": bool(torch.isfinite(st["exp_avg"]).all()),
            "exp_avg_sq_finite": bool(torch.isfinite(st["exp_avg_sq"]).all()),
            "exp_avg_sq_min": float(st["exp_avg_sq"].min()),
        })
    steps = [r["step"] for r in rec["optimizer_groups"]]
    rec["adam_step_values"] = steps
    rec["adam_step_all_equal"] = len(set(steps)) == 1

    # ---- restore state machine (perturbation + real action), bit-exact
    model = C.build_model(ck)
    C.scene_reset()
    sm = C.StateMaster(model)
    snap_p = [p.detach().clone() for p in sm.params]
    snap_s = [(st["step"].detach().clone(), st["exp_avg"].detach().clone(),
               st["exp_avg_sq"].detach().clone())
              for st in (model.optimizer.state[p] for p in sm.params)]

    def state_matches():
        for p, sp in zip(sm.params, snap_p):
            if not torch.equal(p.detach(), sp):
                return False
        for p, (stp, ea, eas) in zip(sm.params, snap_s):
            st = model.optimizer.state[p]
            if not (torch.equal(st["step"], stp) and torch.equal(st["exp_avg"], ea)
                    and torch.equal(st["exp_avg_sq"], eas)):
                return False
        return True

    with torch.no_grad():
        for p in sm.params:
            p.add_(torch.randn_like(p) * 1e-2)
        for p in sm.params:
            st = model.optimizer.state[p]
            st["exp_avg"].add_(torch.randn_like(st["exp_avg"]) * 1e-2)
            st["exp_avg_sq"].add_(torch.randn_like(st["exp_avg_sq"]).abs() * 1e-2)
            st["step"].add_(7)
    sm.restore()
    rec["restore_after_random_perturbation_bit_exact"] = state_matches()

    # real action: one fresh single-view gradient + ordinary Adam step
    cam, gt = dataset.get_item(smoke)
    loss_smoke, g = C.view_grad_and_loss(model, cam, gt, fused_loss)
    g_summary = {"global_sumsq": float((g.double() ** 2).sum()),
                 "finite": bool(torch.isfinite(g).all()),
                 "nonzero_rows": int((g.abs().sum(dim=1) > 0).sum())}
    model.optimizer.step()
    rec["update_norms_after_real_action"] = sm.param_update_norm()
    rec["adam_step_after_action"] = float(
        model.optimizer.state[sm.params[0]]["step"])
    sm.restore()
    rec["restore_after_real_action_bit_exact"] = state_matches()
    rec["smoke"] = {"view_index": int(smoke), "fused_loss": loss_smoke,
                    "grad": g_summary}
    del model, sm, snap_p, snap_s, g, ck
    torch.cuda.empty_cache()
    return rec, tstats


def main():
    scene = sys.argv[1]
    t0 = time.time()
    out = {"campaign": "MG0_OneView_GS_HotCV", "scene": scene,
           "stage": "0_source_checkpoint_audit",
           "frozen_date": "2026-09-29",
           "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
           "gpu_uuid": os.environ.get("MG0_GPU_UUID", "unknown"),
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset")}

    # ---- 1. frozen file identity
    checks = []
    for name, (path, want) in EXPECTED_FILES.items():
        got = C.sha256_file(path) if os.path.exists(path) else "MISSING"
        checks.append({"name": name, "path": path, "expected": want,
                       "actual": got, "ok": got == want})
    out["identity_checks"] = checks
    bad = [c for c in checks if not c["ok"]]
    if bad:
        blocker("identity_mismatch", bad)

    boot = C.bootstrap_c0()
    fused_loss = C.make_fused_loss()
    out["bootstrap"] = boot
    out["fused_loss_so"] = {"path": fused_loss.so_path,
                            "sha256": fused_loss.so_sha256,
                            "lock_ok": fused_loss.so_sha256 == C.P11_FUSED_SO_SHA256}
    out["torch"] = torch.__version__
    out["cuda"] = torch.version.cuda
    out["gpu_name"] = torch.cuda.get_device_name(0)

    # ---- 2. dataset / camera ordering / resolution
    dataset = C.load_scene_dataset(scene)
    names = []
    for cam in dataset._valid_cameras:
        nm = getattr(cam, "image_name", None) or getattr(cam, "img_name", None)
        names.append(str(nm))
    cam0, gt0 = dataset.get_item(0)
    out["dataset"] = {
        "n_cameras": len(dataset), "n_valid_indices": len(dataset._valid_indices),
        "camera_names_first3": names[:3], "camera_names_last3": names[-3:],
        "camera_names_sha256": __import__("hashlib").sha256(
            json.dumps(names).encode()).hexdigest(),
        "ordering": "GTDataset _valid_cameras order == cameras.json order "
                    "(canonical training-camera ordering; index space 0..N-1)",
        "resolution": "1080p", "target_wh": [dataset.target_width, dataset.target_height],
        "gt_shape_sample": list(gt0.shape),
        "cam0_image_wh": [int(cam0.image_width), int(cam0.image_height)],
        "gt_dtype": str(gt0.dtype),
    }
    seqp = os.path.join(C.SNAP_RUNS[scene], "camera_sequence.npy")
    seq = np.load(seqp)
    out["camera_sequence"] = {
        "path": seqp, "sha256": C.sha256_file(seqp), "shape": list(seq.shape),
        "dtype": str(seq.dtype), "min": int(seq.min()), "max": int(seq.max()),
        "n_unique": int(len(np.unique(seq))),
        "matches_dataset_index_space": bool(
            int(seq.min()) == 0 and int(seq.max()) == len(dataset) - 1
            and len(np.unique(seq)) == len(dataset))}
    out["loss_resolution"] = {
        "loss": "0.8*L1 + 0.2*(1-SSIM_fused[padding=same,train=True])",
        "lambda_dssim": C.LAMBDA_DSSIM, "resolution": "1080p (full effective)",
        "eps2d": C.EPS2D, "renderer": "HIGS dynamic (c0)", "arm": "c0",
    }

    # ---- 3. checkpoints
    ckpt_records = {}
    raw_stats = {}
    for it in C.STAGE_CKPTS:
        print(f"[stage0] auditing {scene}@{it} ...", flush=True)
        rec, tstats = audit_checkpoint(scene, it, dataset, fused_loss, smoke=0)
        ckpt_records[str(it)] = rec
        raw_stats[str(it)] = tstats
    out["checkpoints"] = ckpt_records

    ok_all = all(c["ok"] for c in checks)
    for it, rec in ckpt_records.items():
        ok_all = ok_all and rec["reload_bit_exact"] \
            and rec["restore_after_random_perturbation_bit_exact"] \
            and rec["restore_after_real_action_bit_exact"] \
            and rec["adam_step_all_equal"] \
            and all(r["shape_matches_param"] and r["exp_avg_finite"]
                    and r["exp_avg_sq_finite"] for r in rec["optimizer_groups"])
    out["stage0_ok"] = bool(ok_all)

    # ---- provenance copies (snapshot run outputs)
    prov = {}
    for fn in ("results.json", "timing.json", "quality.json", "final_status.json"):
        p = os.path.join(C.SNAP_RUNS[scene], fn)
        if os.path.exists(p):
            prov[fn] = {"sha256": C.sha256_file(p), "bytes": os.path.getsize(p)}
    out["snapshot_run_provenance"] = {"run_dir": C.SNAP_RUNS[scene], "files": prov}
    out["script_sha256"] = {
        "mg0_core.py": C.sha256_file(os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "mg0_core.py")),
        "mg0_stage0_audit.py": C.sha256_file(os.path.abspath(__file__)),
    }
    out["elapsed_s"] = time.time() - t0

    C.json_dump(out, os.path.join(C.ARTIFACTS, f"MG0_SOURCE_IDENTITY_{scene}.json"))
    C.json_dump({"scene": scene, "checkpoints": ckpt_records},
                os.path.join(C.ARTIFACTS, f"MG0_CHECKPOINT_IDENTITY_{scene}.json"))
    C.json_dump({"scene": scene, "tensor_stats": raw_stats},
                os.path.join(C.ARTIFACTS, f"MG0_AUDIT_RAW_{scene}.json"))
    if not ok_all:
        blocker("stage0_check_failed", {"scene": scene,
                                        "checkpoint_summary": {k: {
                                            "reload": v["reload_bit_exact"],
                                            "perturb": v["restore_after_random_perturbation_bit_exact"],
                                            "action": v["restore_after_real_action_bit_exact"]}
                                            for k, v in ckpt_records.items()}})
    print(f"[stage0] {scene} OK in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
