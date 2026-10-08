#!/usr/bin/env python3
"""MSS0 E2 driver: feasibility on room + 2 scenes (short runs).

Guard-first. Runs dev-config selection then short runs of full MSS / no-dual /
local-LM / FULL_NATIVE on a contiguous crop per scene, measuring native-loss
drop, native-feasible accepts, reject rate, max certificate, peak memory, timing.
Fail-closed: E2.json status=PASS only if the frozen E2 gates hold.
"""
import json, os, sys, time, hashlib
from pathlib import Path
import torch

from mx_guard import require_guarded_process, sha256
require_guarded_process()
sys.path.insert(0, os.path.join(os.environ["MSS0_PACKAGE_ROOT"], "native"))
import mss_adapter as A
import mss_macro as X
from provenance import protocol_digest, reference_digest, implementation_digest

dev = "cuda"


def build_scene(scene_key, resolution, cam_idx, ch, cw):
    ckpt = os.environ.get(f"MSS0_E2_{scene_key.upper()}_CKPT", A.CKPT_DEFAULT)
    cam_path = os.environ.get(f"MSS0_E2_{scene_key.upper()}_CAMS",
                              os.path.join(A.CAMS_DEFAULT.replace("/room/", f"/{scene_key}/")))
    params = A.load_checkpoint(ckpt)
    cams_all = A.load_cameras(cam_path, resolution)
    cam = cams_all[cam_idx % len(cams_all)]
    H, W = cam["image_height"], cam["image_width"]
    ch = min(ch, H); cw = min(cw, W)
    cy0 = (H - ch) // 2; cx0 = (W - cw) // 2
    Y_full = X.load_gt_crop(scene_key, cam["image_name"], W, H).to(dev)
    Y_img = Y_full[:, cy0:cy0 + ch, cx0:cx0 + cw].unsqueeze(0).contiguous()
    pixels = X.crop_pixels(ch, cw, cy0, cx0, dev)
    return {"params": params, "cam": cam, "pixels": pixels, "Y_img": Y_img,
            "ch": ch, "cw": cw, "cy0": cy0, "cx0": cx0, "H": H, "W": W}


def run_mss(scene, cfg, ssim, n_macros, use_dual, seed):
    torch.manual_seed(seed)
    state = X.MSSState(scene["params"], dev, scene["params"]["active_sh_degree"]
                       if "active_sh_degree" in scene["params"] else 3)
    sh_deg = state.sh_degree
    cfg = dict(cfg); cfg["use_dual"] = use_dual
    cfg["cy0"] = scene["cy0"]; cfg["cx0"] = scene["cx0"]
    cfg["max_rays"] = int(os.environ.get("MSS0_E2_MAXRAYS", "48"))
    cfg["inner_sweeps"] = int(os.environ.get("MSS0_E2_INNER_SWEEPS", "1"))
    accepts = 0; rejects = 0; certs = []; losses = []
    theta_s = 0.0; n_changed = 0
    torch.cuda.reset_peak_memory_stats()
    t0 = time.monotonic()
    loss0 = None
    for m in range(n_macros):
        r = X.mss_macro(state, scene["cam"], scene["pixels"], scene["Y_img"], cfg, ssim,
                        scene["ch"], scene["cw"])
        if loss0 is None:
            loss0 = r["native_loss0"]
        losses.append(r["native_loss1"])
        certs.append(r["cert"]); theta_s += r["theta_seconds"]; n_changed += r["n_changed"]
        if r["accept"]:
            accepts += 1
        else:
            rejects += 1
        print(f"  [mss] macro {m}: loss={r['native_loss1']:.6f} d={r['loss_delta']:+.5f} accept={r['accept']} cert={r['cert']:.5f} nth={r['n_changed']} cap={r['t_cap']:.2f} img={r['t_img']:.2f} theta={r['theta_seconds']:.2f} r1={r['t_r1']:.2f}s", flush=True)
        if not torch.isfinite(torch.tensor(r["native_loss1"])):
            break
    wall = time.monotonic() - t0
    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    final = losses[-1] if losses else float("nan")
    drop = (final / loss0 - 1.0) if loss0 else float("nan")
    return {"variant": "mss_dual" if use_dual else "mss_nodual", "n_macros": n_macros,
            "accepts": accepts, "rejects": rejects,
            "reject_rate": rejects / max(1, accepts + rejects),
            "max_cert": max(certs) if certs else float("nan"),
            "loss_initial": loss0, "loss_final": final, "loss_drop_pct": drop * 100.0,
            "theta_seconds": theta_s, "wall_seconds": wall, "peak_gib": peak,
            "n_changed_total": n_changed, "losses": losses}


def run_baseline(scene, kind, n_steps, seed, lr, ssim):
    torch.manual_seed(seed)
    state = X.MSSState(scene["params"], dev, 3)
    sh_deg = state.sh_degree
    if kind == "adam":
        opt = torch.optim.Adam([state.p[k] for k in
                                ("xyz", "scaling", "rotation", "opacity", "shs")], lr=lr)
        step = lambda: X.full_native_step(state, scene["cam"], scene["pixels"], scene["Y_img"],
                                          opt, ssim, scene["ch"], scene["cw"],
                                          scene["cy0"], scene["cx0"])
    else:
        opt = torch.optim.LBFGS([state.p[k] for k in
                                 ("xyz", "scaling", "rotation", "opacity", "shs")],
                                lr=lr, max_iter=4, history_size=10)
        step = lambda: X.local_lm_step(state, scene["cam"], scene["pixels"], scene["Y_img"],
                                       opt, ssim, scene["ch"], scene["cw"],
                                       scene["cy0"], scene["cx0"])
    torch.cuda.reset_peak_memory_stats()
    t0 = time.monotonic()
    losses = []
    for _ in range(n_steps):
        l = step()
        losses.append(l)
        if not torch.isfinite(torch.tensor(l)):
            break
    wall = time.monotonic() - t0
    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    final = losses[-1] if losses else float("nan")
    drop = (final / losses[0] - 1.0) if losses and losses[0] else float("nan")
    return {"variant": kind, "n_steps": n_steps, "loss_initial": losses[0] if losses else float("nan"),
            "loss_final": final, "loss_drop_pct": drop * 100.0, "wall_seconds": wall,
            "peak_gib": peak, "losses": losses}


def main():
    resolution = os.environ.get("MSS0_E2_RES", "720p")
    ch = int(os.environ.get("MSS0_E2_CH", "160"))
    cw = int(os.environ.get("MSS0_E2_CW", "160"))
    cam_idx = int(os.environ.get("MSS0_E2_CAM", "150"))
    n_select = int(os.environ.get("MSS0_E2_NSELECT", "5"))
    n_macros = int(os.environ.get("MSS0_E2_NMACROS", "15"))
    seed = int(os.environ.get("MSS0_E2_SEED", "20261006"))
    base_lr = float(os.environ.get("MSS0_E2_LR", "0.005"))
    scenes_keys = os.environ.get("MSS0_E2_SCENES", "room,bonsai,counter").split(",")
    smoke = os.environ.get("MSS0_E2_SMOKE", "0") == "1"

    out = {"checks": {}, "values": {}, "scenes": {}, "config_selection": {}}
    outpath = Path(os.environ["MSS0_OUTPUT_DIR"]); outpath.mkdir(parents=True, exist_ok=True)
    import mss_solver as S
    ssim = S.SepSSIM(device=dev)


    # ---- config selection (room dev crop) ----
    CONFIGS = {
        "default": dict(beta=4.0, rho=4.0, image_steps=5, eta=0.05, mu=1e-2, max_gaussians=200),
        "b1r1":   dict(beta=1.0, rho=1.0, image_steps=5, eta=0.05, mu=1e-2, max_gaussians=200),
        "sweep1": dict(beta=4.0, rho=4.0, image_steps=1, eta=0.05, mu=1e-2, max_gaussians=200),
    }
    if smoke:
        n_select = 2; n_macros = 3
    scene0 = build_scene(scenes_keys[0], resolution, cam_idx, ch, cw)
    sh_deg = int(scene0["params"].get("active_sh_degree", 3))
    pix64 = scene0["pixels"][:64]
    gp0 = A.gaussian_params(scene0["params"])
    capL = X.capture_rays_light(gp0, scene0["cam"], sh_deg, pix64, dev)
    capR = A.capture_rays(gp0, scene0["cam"], sh_deg, pix64, dev)
    dK = int((capL["K"] != capR["K"]).sum().item())
    du1 = float((capL["u1"] - capR["u1"]).abs().max().item()) if capL["u1"].numel() else 0.0
    naccL = int(capL["K"].sum().item()); naccR = int(capR["K"].sum().item())
    print(f"[verify] capture_light vs capture_ref: K_mismatch={dK}/64 accL={naccL} accR={naccR} u1_maxdiff={du1:.6e}", flush=True)
    print(f"[start] smoke={smoke} n_select={n_select} n_macros={n_macros} scenes={scenes_keys} ch={ch} cw={cw} res={resolution}", flush=True)
    sel_results = {}
    for name, cfg in CONFIGS.items():
        r = run_mss(scene0, cfg, ssim, n_select, True, seed)
        sel_results[name] = {"loss_final": r["loss_final"], "wall_seconds": r["wall_seconds"],
                             "loss_drop_pct": r["loss_drop_pct"], "accepts": r["accepts"]}
        print(f"[select] {name}: loss={r['loss_final']:.6f} drop={r['loss_drop_pct']:.3f}% accept={r['accepts']} wall={r['wall_seconds']:.1f}s", flush=True)
        out["config_selection"][name] = sel_results[name]
    best_cfg_name = min(sel_results, key=lambda n: (sel_results[n]["loss_final"],
                                                     sel_results[n]["wall_seconds"]))
    best_cfg = CONFIGS[best_cfg_name]
    out["values"]["selected_config"] = best_cfg_name
    out["values"]["selected_config_sha256"] = hashlib.sha256(
        json.dumps(best_cfg, sort_keys=True).encode()).hexdigest()

    # ---- feasibility per scene ----
    feas = {}
    for sk in scenes_keys:
        sc = build_scene(sk, resolution, cam_idx, ch, cw)
        mss = run_mss(sc, best_cfg, ssim, n_macros, True, seed)
        nod = run_mss(sc, best_cfg, ssim, n_macros, False, seed)
        adam = run_baseline(sc, "adam", n_macros, seed, base_lr, ssim)
        lbfgs = run_baseline(sc, "lbfgs", max(3, n_macros // 3), seed, base_lr, ssim)
        feas[sk] = {"mss": mss, "no_dual": nod, "full_native": adam, "local_lm": lbfgs}
        out["scenes"][sk] = feas[sk]
        print(f"[scene {sk}] mss accept={mss['accepts']} drop={mss['loss_drop_pct']:.3f}% cert={mss['max_cert']:.5f} wall={mss['wall_seconds']:.1f}s | nodrop={nod['loss_drop_pct']:.3f}% | adam={adam['loss_drop_pct']:.3f}% | lbfgs={lbfgs['loss_drop_pct']:.3f}%", flush=True)

    # ---- gates ----
    CERT_TOL = 1.0 / 255.0 + 2e-5
    g = {}
    mss_all = [feas[sk]["mss"] for sk in scenes_keys]
    g["each_scene_accepts_ge_10"] = all(m["accepts"] >= 10 for m in mss_all)
    g["each_scene_loss_drop_ge_1pct"] = all(m["loss_drop_pct"] <= -1.0 for m in mss_all)
    g["mss_reject_rate_le_half"] = all(m["reject_rate"] <= 0.5 for m in mss_all)
    g["mss_max_cert_within_tol"] = all(m["max_cert"] <= CERT_TOL for m in mss_all)
    g["memory_within_32gib"] = all(max(feas[sk]["mss"]["peak_gib"],
                                       feas[sk]["no_dual"]["peak_gib"],
                                       feas[sk]["full_native"]["peak_gib"]) <= 32.0
                                   for sk in scenes_keys)
    out["checks"] = g
    out["values"]["cert_tol"] = CERT_TOL
    out["values"]["min_accepts"] = min(m["accepts"] for m in mss_all)
    out["values"]["worst_drop_pct"] = max(m["loss_drop_pct"] for m in mss_all)
    out["all_pass"] = all(g.values())

    raw = outpath / "E2_feasibility.json"
    raw.write_text(json.dumps(out, indent=2) + "\n")
    obj = {
        "schema": "mss0.e2.v1", "stage": "E2", "status": "PASS" if out["all_pass"] else "FAIL",
        "protocol_sha256": protocol_digest(), "reference_sha256": reference_digest(),
        "implementation_sha256": implementation_digest(),
        "selected_config_sha256": out["values"]["selected_config_sha256"],
        "scene_manifest_sha256": None, "source_patch_sha256": None,
        "raw_files": [{"path": raw.name, "sha256": sha256(raw), "kind": "measurements"}],
        "note": "E2 feasibility: dev-config selection + short runs of MSS/no-dual/local-LM/FULL_NATIVE on room+2 scenes.",
    }
    (outpath / "E2.json").write_text(json.dumps(obj, indent=2) + "\n")
    print(json.dumps({"status": obj["status"], "selected": best_cfg_name,
                      "checks": g,
                      "summary": {sk: {"accepts": feas[sk]["mss"]["accepts"],
                                       "drop_pct": feas[sk]["mss"]["loss_drop_pct"],
                                       "cert": feas[sk]["mss"]["max_cert"],
                                       "peak_gib": feas[sk]["mss"]["peak_gib"]}
                                   for sk in scenes_keys}}, indent=2))


if __name__ == "__main__":
    main()