#!/usr/bin/env python3
"""P13-Y oracle: adapter parity (Sec.5) + Y1 (Sec.6) + Y2 (Sec.7) + Y3 (Sec.8).

Structural-copy guarantee: this script imports the capture trainer source as a
module and AST-extracts the trainer's OWN `render_with_meta` and
`train_ssim_fn` definitions, exec'ing them inside that module's namespace.
The adapter therefore IS the trainer's render/loss code path (no reimplementation),
and the parity gate (Sec.5) verifies it bitwise from captured raw state:
  same state + same camera -> trainer frame vs adapter frame bitwise; else
  max|I_A-I_B| <= 1e-7 AND fused-loss |L_A-L_B|/|L_B| <= 1e-7.
Any failure -> STOP_ADAPTER_PARITY_FAILED (no oracle results are produced).

Y1: exact renderer-meta snapshots (P13-X p13y_probe_core instrumentation) for
    both endpoint states under the shared revisit camera.
Y2: central FD of R along d = theta1 - theta0 (h = 0.01 primary, 0.005/0.02
    crosscheck, no post-hoc h selection); E_delta primary.
Y3: lam1 = dL/dI1 at theta1 (fixed); g_true = J(theta1)^T lam1 versus
    g_stale = J(theta0)^T lam1, per raw parameter group.
"""
import argparse
import ast
import hashlib
import json
import math
import os
import sys
import textwrap
import time
import types

import numpy as np
import torch

ART = "/root/epic05_p13y/artifacts"
CAPTURE_TRAINER = "/root/epic05_p13y/publication_trainer_p13y_capture.py"
FUSED_SO_SHA = "dd6a4c799b98ff9fa82b4203acce3b36b1529c97155c0ef4cd50fd0f583d8ebb"
H_VALUES = (0.01, 0.005, 0.02)     # 0.01 = protocol primary; others crosscheck
H_PRIMARY = 0.01

# --------------------------------------------------------------- adapter ---


def _extract_func(src, tree, name):
    hits = [n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(hits) != 1:
        raise SystemExit(f"AST extraction of {name}: {len(hits)} definitions")
    return textwrap.dedent(ast.get_source_segment(src, hits[0]))


def load_adapter():
    """Import the capture trainer as a module and AST-copy its render + loss."""
    src = open(CAPTURE_TRAINER, encoding="utf-8").read()
    tree = ast.parse(src)
    mod = types.ModuleType("p13y_capture_trainer")
    mod.__file__ = CAPTURE_TRAINER
    mod.__dict__["__name__"] = "p13y_capture_trainer"
    exec(compile(src, CAPTURE_TRAINER, "exec"), mod.__dict__)
    mod.args = types.SimpleNamespace(arm="c0")
    mod.eps2d_eff = 0.3
    mod.rasterization = None
    mod.GaussianRasterizationSettings = None
    for name in ("render_with_meta", "train_ssim_fn"):
        exec(_extract_func(src, tree, name), mod.__dict__)
    if not callable(getattr(mod, "train_ssim_fn", None)) or \
       not callable(getattr(mod, "render_with_meta", None)):
        raise SystemExit("adapter extraction failed")
    return mod


def verify_fused_lock(mod):
    """Re-run the P12 fused-loss lock + padding probe (RNG-transparent)."""
    import fused_ssim as fsm
    import fused_ssim_cuda as fsc
    so_sha = mod.sha256_file(fsc.__file__)
    if so_sha != FUSED_SO_SHA:
        raise SystemExit(f"fused_ssim_cuda.so hash {so_sha} != locked {FUSED_SO_SHA}")
    mod._fused_ssim_mod = fsm
    rc, rcu = torch.get_rng_state(), torch.cuda.get_rng_state()
    torch.manual_seed(42)
    pa = torch.rand(1, 3, 192, 256, device="cuda")
    pb = torch.rand(1, 3, 192, 256, device="cuda")
    m = fsm.FusedSSIMMap.apply(0.01 ** 2, 0.03 ** 2, pa, pb, "same", True, 2)
    val = float(fsm.fused_ssim(pa, pb, padding="same", train=True))
    torch.set_rng_state(rc)
    torch.cuda.set_rng_state(rcu)
    return {"so_path": fsc.__file__, "so_sha256": so_sha,
            "map_shape_ok": tuple(m.shape) == tuple(pa.shape),
            "probe_ssim_scalar": val}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def bytes_fingerprint(t):
    u = np.ascontiguousarray(t.detach().cpu().numpy())
    h = hashlib.sha256()
    h.update(str(u.dtype).encode())
    h.update(np.asarray(u.shape, dtype=np.int64).tobytes())
    h.update(u.tobytes())
    return h.hexdigest()


STATE_ORDER = ("xyz", "scaling", "rotation", "opacity", "shs")


def state_fingerprint(raw):
    h = hashlib.sha256()
    for name in STATE_ORDER:
        v = np.ascontiguousarray(raw[name].detach().cpu().numpy())
        h.update(name.encode())
        h.update(str(v.dtype).encode())
        h.update(np.asarray(v.shape, dtype=np.int64).tobytes())
        h.update(v.tobytes())
    return h.hexdigest()


def camera_fingerprint(cam):
    m = (cam.viewmatrix.detach().cpu().numpy().tobytes()
         + cam.K.detach().cpu().numpy().tobytes()
         + f"{int(cam.image_width)}x{int(cam.image_height)}".encode())
    return hashlib.sha256(m).hexdigest()


def load_state(state_dir):
    """Load raw state dict + meta; re-verify every recorded hash."""
    with open(os.path.join(state_dir, "meta.json")) as f:
        meta = json.load(f)
    raw = torch.load(os.path.join(state_dir, "state.pt"), map_location="cpu",
                     weights_only=False)
    checks = {"state_fingerprint": state_fingerprint(raw),
              "per_tensor": {k: bytes_fingerprint(raw[k]) for k in STATE_ORDER}}
    checks["state_sha256_match"] = checks["state_fingerprint"] == meta["state_sha256"]
    checks["per_tensor_match"] = all(
        checks["per_tensor"][k] == meta["per_tensor_sha256"][k] for k in STATE_ORDER)
    if not (checks["state_sha256_match"] and checks["per_tensor_match"]):
        raise SystemExit(f"state hash mismatch in {state_dir}")
    return raw, meta, checks


def build_model(adapter, raw, active_sh_degree, device="cuda"):
    """Reconstruct the live model from raw state, exactly as the trainer holds it.

    Parameters require grad (as in the trainer); non-gradient renders wrap the
    call in torch.no_grad(), which is the trainer's own eval-path combination.
    """
    m = adapter.GaussianModel(max_sh_degree=3)
    for k in STATE_ORDER:
        setattr(m, f"_{k}", torch.nn.Parameter(
            raw[k].to(device).contiguous(), requires_grad=True))
    m.active_sh_degree = int(active_sh_degree)
    return m


def mark_scene_dirty(adapter):
    """Reset the C0 HIGS dynamic scene tracker before each independent render.

    The trainer owns one long-lived scene handle: it renders its live tensors and
    calls `mark_dirty()` after densify/prune/opacity-reset. The oracle instead
    renders INDEPENDENT captured states, each of which is a different master
    tensor set with no live training graph behind it. `mark_dirty()` cannot be
    used for that (a no-grad forward leaves a pending backward and the frozen
    renderer then refuses to mutate topology), so each render starts a fresh
    scene: `reset()` releases the handle and the next forward rebinds to the
    current tensors -- the same condition as the trainer's own first forward,
    which the parity gate verifies bitwise against the trainer's captured frame.
    """
    from gsplat.experimental.render.functional.gaussian_inference import (
        _HIGS_DYNAMIC_SCENE,
    )
    _HIGS_DYNAMIC_SCENE.reset()


def render_frame(adapter, model, cam, sh_degree, grad=False):
    mark_scene_dirty(adapter)
    if grad:
        with torch.enable_grad():
            frame, _r, _m = adapter.render_with_meta(model, cam, sh_degree)
    else:
        with torch.no_grad():
            frame, _r, _m = adapter.render_with_meta(model, cam, sh_degree)
    return frame


def fused_loss_value(adapter, img, gt):
    with torch.no_grad():
        return float(adapter.train_ssim_fn(img.clamp(0.0, 1.0), gt))


def psnr(a, b):
    mse = float(((a - b) ** 2).mean())
    return 100.0 if mse <= 1e-12 else 20.0 * math.log10(1.0 / math.sqrt(mse))


# ------------------------------------------------------------------- Y3 ----

GROUPS = {
    "means": lambda raw: raw["xyz"],
    "opacity": lambda raw: raw["opacity"],
    "scales": lambda raw: raw["scaling"],
    "quats": lambda raw: raw["rotation"],
    "sh0": lambda raw: raw["shs"][:, 0, :],
    "shN": lambda raw: raw["shs"][:, 1:, :],
}


def group_metrics(g_true, g_stale):
    a = g_true.detach().reshape(-1).float()
    b = g_stale.detach().reshape(-1).float()
    na, nb = float(a.norm()), float(b.norm())
    out = {"n": int(a.numel()), "norm_true": na, "norm_stale": nb,
           "norm_ratio": (nb / na) if na > 0 else None,
           "cos": (float((a @ b) / (na * nb)) if na > 0 and nb > 0 else None),
           "rel_l2": (float((b - a).norm() / na) if na > 0 else None)}
    k = max(1, int(math.ceil(0.1 * a.numel())))
    idx = torch.topk(a.abs(), k).indices
    aa, bb = a[idx], b[idx]
    same = ((aa > 0) & (bb > 0)) | ((aa < 0) & (bb < 0))
    out["top10_sign_agreement"] = float(same.float().mean())
    out["top10_n"] = int(k)
    out["top10_zero_stale"] = int((bb == 0).sum())
    return out


# ------------------------------------------------------------------ Y1 -----

def y1_snapshots(probe_core, adapter, raw0, raw1, shd, cam, scene, pair_tag, out_dir,
                 t0, t1):
    # probe_core.snapshot_from_meta is the verbatim P13-X instrument: it names
    # files by (scene, numeric anchor_id, numeric iteration). Pair identity and
    # t0/t1 identity therefore live in the per-pair subdirectory, not the stem.
    pair_dir = os.path.join(out_dir, f"pair_{pair_tag}")
    os.makedirs(pair_dir, exist_ok=True)
    call_rec = {"fn": "p13y_probe_core.classic_rasterize (P13-X-identical call)",
                "kwargs": dict(probe_core.P13Y_CLASSIC_CALL_KWARGS),
                "gsplat_file": probe_core.gsplat_file()}
    metas = {}
    for tag, raw, iteration, anchor_id in (("t0", raw0, t0, 0), ("t1", raw1, t1, 1)):
        m = build_model(adapter, raw, shd)
        with torch.no_grad():
            _r, _a, meta = probe_core.classic_rasterize(
                means=m.get_xyz, quats=m.get_rotation, scales=m.get_scaling,
                opacities=m.get_opacity, colors=m.get_features,
                viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
                width=cam.image_width, height=cam.image_height,
                sh_degree=shd)
        probe_meta = {
            "radii": meta["radii"].detach().reshape(-1, 2).min(dim=-1).values,
            "depths": meta["depths"],
            "isect_offsets": meta["isect_offsets"],
            "flatten_ids": meta["flatten_ids"],
            "isect_ids": meta["isect_ids"],
        }
        metas[tag] = probe_core.snapshot_from_meta(
            probe_meta, iteration=int(iteration), scene=scene,
            anchor_id=anchor_id, out_dir=pair_dir)
        del m
        torch.cuda.empty_cache()
    cmp = probe_core.compare_snapshots(metas["t0"]["raw_npz"], metas["t1"]["raw_npz"])
    return {"t0_summary": metas["t0"], "t1_summary": metas["t1"], "compare": cmp,
            "classic_call": call_rec}


# --------------------------------------------------------------- modes -----

def mode_parity(args, adapter, dataset, probe_core):
    run_dir = args.run_dir
    probe_dirs = sorted(d for d in os.listdir(run_dir) if d.startswith("probe_i"))
    if not probe_dirs:
        raise SystemExit(f"no probe dirs in {run_dir}")
    results, all_pass, all_bitwise = [], True, True
    for pd in probe_dirs:
        pdir = os.path.join(run_dir, pd)
        with open(os.path.join(pdir, "probe_meta.json")) as f:
            pm = json.load(f)
        raw, smeta, sck = load_state(os.path.dirname(pm["state_dir"]))
        cam, gt = dataset.get_item(int(pm["cam_idx"]))
        cam_sha = camera_fingerprint(cam)
        m = build_model(adapter, raw, smeta["active_sh_degree"])
        frame = render_frame(adapter, m, cam, int(pm["sh_degree"]))
        del m
        tr = torch.load(os.path.join(pdir, "trainer_frame.pt"), map_location="cpu",
                        weights_only=False)["frame"].to(device=frame.device,
                                                        dtype=frame.dtype)
        bitwise = bool(frame.shape == tr.shape) and bool(torch.equal(frame, tr))
        mad = float((frame.float() - tr.float()).abs().max().item())
        loss_a = fused_loss_value(adapter, frame, gt)
        loss_t = float(pm["trainer_loss"])
        lrel = abs(loss_a - loss_t) / abs(loss_t) if loss_t != 0 else abs(loss_a)
        ok = bitwise or (mad <= 1e-7 and lrel <= 1e-7)
        all_pass &= ok
        all_bitwise &= bitwise
        rec = {
            "probe": pd, "iteration": pm["iteration"], "cam_idx": pm["cam_idx"],
            "sh_degree": pm["sh_degree"],
            "camera_sha256_trainer": pm["camera_sha256"],
            "camera_sha256_adapter": cam_sha,
            "camera_identity_ok": cam_sha == pm["camera_sha256"],
            "state_sha256_recorded": pm["state_sha256"],
            "state_sha256_adapter_recheck": sck["state_fingerprint"],
            "state_hash_ok": sck["state_sha256_match"] and sck["per_tensor_match"],
            "frame_bitwise_equal": bitwise, "frame_max_abs_diff": mad,
            "n_gaussians": smeta["n_gaussians"],
            "loss_trainer": loss_t, "loss_adapter": loss_a, "loss_rel_err": lrel,
            "pass": ok,
        }
        results.append(rec)
        print(f"  [parity {pd}] bitwise={bitwise} mad={mad:.3e} loss_rel={lrel:.3e} "
              f"pass={ok}", flush=True)
        del frame
        torch.cuda.empty_cache()
    rep = {"phase": "P13-Y Sec.5 adapter parity", "scene": args.scene,
           "run_dir": run_dir, "date": "2026-09-27",
           "gate": {"rule": ("bitwise identical; else max|I_A-I_B| <= 1e-7 AND "
                             "|L_A-L_B|/|L_B| <= 1e-7"),
                    "all_bitwise": bool(all_bitwise), "all_pass": bool(all_pass),
                    "status": "PASS" if all_pass else "STOP_ADAPTER_PARITY_FAILED"},
           "adapter": {"capture_trainer": CAPTURE_TRAINER,
                       "capture_trainer_sha256": sha256_file(CAPTURE_TRAINER),
                       "extracted": ["render_with_meta", "train_ssim_fn"],
                       "extraction": "AST source copy from the capture trainer"},
           "probes": results}
    out = args.out or f"{ART}/P13Y_ADAPTER_PARITY_{args.scene}.json"
    with open(out, "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps({k: rep[k] for k in ("gate",)}, indent=2))
    print("WROTE", out)
    return all_pass


def mode_oracle(args, adapter, dataset, probe_core):
    plan = json.load(open(args.plan))
    scene = plan["scene"]
    cap_dir = args.capture_dir
    entries = plan["quantile_selection"]
    out_dir = args.outdir or f"/root/epic05_p13y/pair_results/{scene}"
    os.makedirs(out_dir, exist_ok=True)
    y1_dir = os.path.join(out_dir, "y1_snapshots")
    gt_cache = {}

    def gt_of(cam_idx):
        if cam_idx not in gt_cache:
            cam, gt = dataset.get_item(cam_idx)
            gt_cache[cam_idx] = (cam, gt)
        return gt_cache[cam_idx]

    summary = []
    for idx, entry in enumerate(entries):
        pair = entry.get("pair") or entry.get("chosen")
        tag = f"q{int(round(entry['quantile'] * 100)):02d}"
        if args.pairs and idx not in args.pairs:
            continue
        t0, t1, cam_id, lag = pair["t0"], pair["t1"], pair["camera_id"], pair["lag"]
        rec = {"phase": "P13-Y pair oracle", "scene": scene, "pair_index": idx,
               "quantile_tag": tag,
               "pair_spec": {"camera_id": cam_id, "t0": t0, "t1": t1, "lag": lag,
                             "lag_target": entry["lag_target"],
                             "quantile": entry["quantile"]},
               "date": "2026-09-27", "h_primary": H_PRIMARY,
               "h_values": list(H_VALUES)}
        t_start = time.perf_counter()
        try:
            raw0, meta0, _ck0 = load_state(os.path.join(cap_dir, f"i{t0:05d}"))
            raw1, meta1, _ck1 = load_state(os.path.join(cap_dir, f"i{t1:05d}"))
            assert meta0["cam_idx_of_this_iteration"] == cam_id, "t0 cam mismatch"
            assert meta1["cam_idx_of_this_iteration"] == cam_id, "t1 cam mismatch"
            assert meta0["n_gaussians"] == meta1["n_gaussians"], "topology differs"
            shd0, shd1 = meta0["active_sh_degree"], meta1["active_sh_degree"]
            cam, gt = gt_of(cam_id)
            cam_sha = camera_fingerprint(cam)
            rec["states"] = {
                "t0": {"dir": f"i{t0:05d}", "sha256": meta0["state_sha256"],
                       "n": meta0["n_gaussians"], "sh_degree": shd0,
                       "npy_lr": meta0["optimizer_xyz_lr"]},
                "t1": {"dir": f"i{t1:05d}", "sha256": meta1["state_sha256"],
                       "n": meta1["n_gaussians"], "sh_degree": shd1},
            }
            rec["camera"] = {"cam_idx": cam_id, "sha256": cam_sha,
                             "sh_degree_used": shd1}
            rec["topology_frozen"] = True
            torch.cuda.empty_cache()

            # ---- Y1: renderer-meta graph stability (classic meta path) ----
            if args.skip_y1:
                rec["Y1"] = {"skipped": True}
            else:
                t_y1 = time.perf_counter()
                rec["Y1"] = y1_snapshots(probe_core, adapter, raw0, raw1, shd1, cam,
                                         scene, tag, y1_dir, t0, t1)
                rec["Y1"]["wall_s"] = time.perf_counter() - t_y1

            # ---- Y2: forward Taylor linearity ----
            d = {k: (raw1[k] - raw0[k]) for k in STATE_ORDER}
            m0 = build_model(adapter, raw0, shd1)
            m1 = build_model(adapter, raw1, shd1)
            I0 = render_frame(adapter, m0, cam, shd1).detach().clone()
            I1 = render_frame(adapter, m1, cam, shd1).detach().clone()
            del m0, m1
            torch.cuda.empty_cache()
            norm_I1 = float(I1.norm())
            norm_dI = float((I1 - I0).norm())
            delta_rel = norm_dI / norm_I1 if norm_I1 > 0 else float("nan")
            L1 = fused_loss_value(adapter, I1, gt)
            per_h = {}
            Jd_store = {}
            for h in H_VALUES:
                ra = {k: (raw0[k] + h * d[k]) for k in STATE_ORDER}
                rb = {k: (raw0[k] - h * d[k]) for k in STATE_ORDER}
                ma = build_model(adapter, ra, shd1)
                mb = build_model(adapter, rb, shd1)
                Ia = render_frame(adapter, ma, cam, shd1).detach()
                Ib = render_frame(adapter, mb, cam, shd1).detach()
                Jd = (Ia - Ib) / (2.0 * h)
                del ma, mb, Ia, Ib, ra, rb
                torch.cuda.empty_cache()
                I_lin = I0 + Jd
                Jd_store[h] = Jd.clone()
                e_delta = float((I_lin - I1).norm()) / max(norm_dI, 1e-300)
                L_lin = fused_loss_value(adapter, I_lin, gt)
                per_h[str(h)] = {
                    "Jd_norm": float(Jd.norm()),
                    "E_delta_unclamped": e_delta,
                    "E_delta_clamped": float((I_lin.clamp(0, 1) - I1).norm())
                    / max(norm_dI, 1e-300),
                    "E_I": float((I_lin - I1).norm()) / max(norm_I1, 1e-300),
                    "psnr_lin_vs_I1": psnr(I_lin.clamp(0, 1), I1),
                    "L_lin_clamped_input": L_lin,
                    "L_rel_err_clamped_input": abs(L_lin - L1) / abs(L1),
                    "L_lin_raw_input": fused_loss_value(adapter, I_lin, gt),
                }
            cross = {}
            hs = list(H_VALUES)
            for i in range(len(hs)):
                for j in range(i + 1, len(hs)):
                    a, b = hs[i], hs[j]
                    da, db = Jd_store[b], Jd_store[a]
                    cross[f"{b}_vs_{a}"] = float((da - db).norm()) / max(
                        float(db.norm()), 1e-300)
            h_consistency_max = max(cross.values()) if cross else None
            del Jd_store
            torch.cuda.empty_cache()
            prim = per_h[str(H_PRIMARY)]
            y2_valid = bool(delta_rel >= 1e-6 and (h_consistency_max or 0) <= 0.05)
            y2_pass = bool(y2_valid and prim["E_delta_unclamped"] <= 0.10
                           and prim["L_rel_err_clamped_input"] <= 0.005)
            y2_cat = bool(prim["E_delta_unclamped"] > 0.50
                          or prim["L_rel_err_clamped_input"] > 0.05)
            rec["Y2"] = {
                "norm_I1": norm_I1, "norm_I1_minus_I0": norm_dI,
                "delta_rel": delta_rel, "valid": y2_valid,
                "h_consistency_max_rel": h_consistency_max,
                "h_consistency_pairwise": cross,
                "per_h": per_h, "primary_h": H_PRIMARY,
                "primary": {"E_delta": prim["E_delta_unclamped"],
                            "E_I": prim["E_I"],
                            "psnr_lin_vs_I1": prim["psnr_lin_vs_I1"],
                            "L_rel_err": prim["L_rel_err_clamped_input"]},
                "pass": y2_pass, "catastrophic": y2_cat,
                "gate": "E_delta <= 0.10 AND |dL|/L <= 0.005",
                "L1_trainer_side": L1,
            }

            # ---- Y3: stale Jacobian backward with fixed lam1 ----
            # theta1 graph: lam1 = dL/dI1 (clamped image), g_true = J(t1)^T lam1
            m1g = build_model(adapter, raw1, shd1)
            p1 = [getattr(m1g, f"_{k}") for k in STATE_ORDER]
            I1g = render_frame(adapter, m1g, cam, shd1, grad=True)
            I1c = I1g.clamp(0, 1)
            L1g = adapter.train_ssim_fn(I1c, gt)
            lam1 = torch.autograd.grad(L1g, I1c, retain_graph=True)[0]
            lam_stats = {"norm": float(lam1.norm()),
                         "mean_abs": float(lam1.abs().mean()),
                         "frac_zero": float((lam1 == 0).float().mean()),
                         "n_out_of_range_pixels": int(((I1g < 0) | (I1g > 1)).sum()),
                         "n_pixels": int(I1g.numel())}
            g_true = torch.autograd.grad(I1c, p1, grad_outputs=lam1,
                                         retain_graph=False)
            del m1g, I1g, I1c, L1g
            torch.cuda.empty_cache()
            # same lam1 adjoint, theta0 graph: g_stale = J(t0)^T lam1
            m0g = build_model(adapter, raw0, shd1)
            p0 = [getattr(m0g, f"_{k}") for k in STATE_ORDER]
            I0g = render_frame(adapter, m0g, cam, shd1, grad=True)
            g_stale = torch.autograd.grad(I0g.clamp(0, 1), p0, grad_outputs=lam1,
                                          retain_graph=False)
            del m0g, I0g
            torch.cuda.empty_cache()
            # two-step vs direct check (fresh graph; validates the split path)
            m1d = build_model(adapter, raw1, shd1)
            p1d = [getattr(m1d, f"_{k}") for k in STATE_ORDER]
            I1d = render_frame(adapter, m1d, cam, shd1, grad=True)
            L1d = adapter.train_ssim_fn(I1d.clamp(0, 1), gt)
            g_direct = torch.autograd.grad(L1d, p1d, retain_graph=False)
            del m1d, I1d, L1d
            torch.cuda.empty_cache()
            two_step = {k: {"rel_diff": float((g_true[i] - g_direct[i]).norm()
                                              / max(float(g_direct[i].norm()), 1e-300)),
                            "norm_direct": float(g_direct[i].norm())}
                        for i, k in enumerate(STATE_ORDER)}
            g_true_d = {k: g_true[i] for i, k in enumerate(STATE_ORDER)}
            g_stale_d = {k: g_stale[i] for i, k in enumerate(STATE_ORDER)}
            gm = {
                "means": group_metrics(g_true_d["xyz"], g_stale_d["xyz"]),
                "opacity": group_metrics(g_true_d["opacity"], g_stale_d["opacity"]),
                "scales": group_metrics(g_true_d["scaling"], g_stale_d["scaling"]),
                "quats": group_metrics(g_true_d["rotation"], g_stale_d["rotation"]),
                "sh0": group_metrics(g_true_d["shs"][:, 0, :], g_stale_d["shs"][:, 0, :]),
                "shN": group_metrics(g_true_d["shs"][:, 1:, :], g_stale_d["shs"][:, 1:, :]),
                "shs_all": group_metrics(g_true_d["shs"], g_stale_d["shs"]),
            }
            prim_cos = gm["means"]["cos"]
            y3_pass = bool(
                gm["means"]["cos"] is not None and gm["means"]["cos"] >= 0.995
                and gm["means"]["rel_l2"] <= 0.05
                and gm["opacity"]["cos"] is not None
                and gm["opacity"]["cos"] >= 0.995 and gm["opacity"]["rel_l2"] <= 0.05
                and all(gm[g]["cos"] is not None and gm[g]["cos"] >= 0.990
                        and gm[g]["rel_l2"] <= 0.10
                        for g in ("scales", "quats", "sh0", "shN")))
            y3_cat = bool(prim_cos is not None and (prim_cos < 0.98
                          or gm["means"]["rel_l2"] > 0.25))
            rec["Y3"] = {
                "lam1_stats": lam_stats,
                "groups": gm,
                "primary_means_opacity_rule": "cos >= 0.995 AND rel_l2 <= 0.05",
                "secondary_rule": "cos >= 0.990 AND rel_l2 <= 0.10",
                "two_step_vs_direct": two_step,
                "pass": y3_pass, "catastrophic": y3_cat,
                "cos_means": prim_cos,
                "E_rel_means": gm["means"]["rel_l2"],
            }
            rec["verdict"] = {
                "Y2_pass": y2_pass, "Y3_pass": y3_pass,
                "joint_pass": bool(y2_pass and y3_pass),
                "catastrophic": bool(y2_cat or y3_cat),
            }
            del I0, I1, g_true, g_stale, g_direct, g_true_d, g_stale_d, lam1
            torch.cuda.empty_cache()
        except Exception as e:  # preserve failures verbatim
            rec["error"] = repr(e)
            rec["verdict"] = {"Y2_pass": False, "Y3_pass": False,
                              "joint_pass": False, "catastrophic": None,
                              "invalid": True}
            import traceback
            rec["traceback"] = traceback.format_exc()
        rec["wall_s"] = time.perf_counter() - t_start
        fname = os.path.join(out_dir, f"pair_{idx:02d}_{tag}_cam{cam_id}_t{t0}_{t1}.json")
        with open(fname, "w") as f:
            json.dump(rec, f, indent=2, default=str)
        print(f"  [pair {idx:02d} {tag} cam{cam_id} {t0}->{t1}] "
              f"Y2={'PASS' if rec['verdict'].get('Y2_pass') else 'fail'} "
              f"Y3={'PASS' if rec['verdict'].get('Y3_pass') else 'fail'} "
              f"E_delta={rec.get('Y2', {}).get('primary', {}).get('E_delta')} "
              f"cos_means={rec.get('Y3', {}).get('cos_means')} -> {fname}", flush=True)
        summary.append({"pair_index": idx, "tag": tag, "camera_id": cam_id,
                        "t0": t0, "t1": t1, "lag": lag,
                        "file": fname, "verdict": rec["verdict"],
                        "E_delta": rec.get("Y2", {}).get("primary", {}).get("E_delta"),
                        "cos_means": rec.get("Y3", {}).get("cos_means"),
                        "E_rel_means": rec.get("Y3", {}).get("E_rel_means"),
                        "Y1": {k: rec.get("Y1", {}).get("compare", {}).get(k)
                               for k in ("visible_jaccard", "membership_jaccard",
                                         "unchanged_membership_tile_fraction",
                                         "order_inversion_rate_median_tile",
                                         "unchanged_full_order_tile_fraction")}})
    out = os.path.join(out_dir, "_SUMMARY.json")
    with open(out, "w") as f:
        json.dump({"scene": scene, "plan": args.plan, "pairs": summary}, f, indent=2)
    print("WROTE", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["parity", "oracle"])
    ap.add_argument("--scene", required=True)
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--run-dir", default=None, help="parity: capture run dir with probe_i*")
    ap.add_argument("--capture-dir", default=None, help="oracle: states dir ({cfg}/iXXXXX)")
    ap.add_argument("--plan", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--pairs", type=int, nargs="*", default=None)
    ap.add_argument("--skip-y1", action="store_true")
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))

    adapter = load_adapter()
    ident = adapter.bootstrap_c0()
    import gsplat  # noqa: F401
    from gsplat.experimental import rasterize_gaussian_higs_dynamic as _rgd
    adapter.rasterize_gaussian_higs_dynamic = _rgd
    from gaussian_model import GaussianModel  # noqa: F401
    from config import ReferenceV1Config  # noqa: F401
    from dataset import GTDataset
    adapter.GaussianModel = GaussianModel
    adapter.ReferenceV1Config = ReferenceV1Config
    adapter.config = ReferenceV1Config()
    fused = verify_fused_lock(adapter)
    print(f"  [oracle] bootstrap={ident.get('so_sha256', '')[:16]} fused={fused['so_sha256'][:16]}",
          flush=True)

    # P13-X probe core for Y1 (classic renderer-meta snapshots): the pinned
    # P13-X classic call kwargs live inside the module (P13Y_CLASSIC_CALL_KWARGS).
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "p13y_probe_core", "/root/epic05_p13y/p13y_probe_core.py")
    probe_core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe_core)
    assert callable(getattr(probe_core, "classic_rasterize", None)), \
        "p13y_probe_core.classic_rasterize missing"
    print(f"  [oracle] probe_core classic call: "
          f"{probe_core.P13Y_CLASSIC_CALL_KWARGS} gsplat={probe_core.gsplat_file()}",
          flush=True)

    dataset = GTDataset(scene=args.scene, repo_root="/root/3dgs-renderer-benchmark",
                        resolution="1080p", device="cuda", background="black")
    print(f"  [oracle] scene={args.scene} cameras={len(dataset)}", flush=True)
    if args.mode == "parity":
        ok = mode_parity(args, adapter, dataset, probe_core)
        sys.exit(0 if ok else 3)
    mode_oracle(args, adapter, dataset, probe_core)


if __name__ == "__main__":
    main()
