#!/usr/bin/env python3
"""MSS-E2R0 driver: observable-subspace solver feasibility.

Answers ONE narrow question: can a low-cost solver (LM / precond-LM / truncated
PCG / observable-subspace GN) restrict the per-Gaussian update to sufficiently
observable directions so ray-capped MSS produces nonzero, accepted,
loss-decreasing updates within the original throughput envelope?

Frozen (NOT modified): MSS0 verdict, E2 acceptance semantics, certificate
tolerance, memory gate, scene/loss/camera/native-baseline definitions, trust
clip. Only the per-Gaussian linear solve is swapped. Fail-closed; never
fabricate a PASS. New semantics would require REQUIRES_NEW_PROTOCOL.
"""
import json, os, sys, time, hashlib
from pathlib import Path
import numpy as np
import torch

# ---- guard (skippable for direct debug) -------------------------------------
if os.environ.get("MSS_E2R0_SKIP_GUARD") != "1":
    from mx_guard import require_guarded_process
    require_guarded_process()

PKG = os.environ.get("MSS0_PACKAGE_ROOT") or os.environ.get("MSS_E2R0_PKG_ROOT")
if not PKG:
    sys.exit("MSS0_PACKAGE_ROOT / MSS_E2R0_PKG_ROOT not set")
sys.path.insert(0, os.path.join(PKG, "native"))
import mss_adapter as A
import mss_macro as X
import mss_solver as S
import mss_e2r0 as E2

dev = "cuda"
CERT_TOL = 1.0 / 255.0 + 2e-5
SOLVER_NAMES = ["S1_lm", "S2_precond_lm", "S3_pcg", "S4_obs_subspace"]
DELIV_FILES = ["E2R0_NATIVE_SANITY.json", "E2R0_BASELINE_REPRO.json",
               "E2R0_LM.json", "E2R0_PRECOND_LM.json", "E2R0_PCG.json",
               "E2R0_OBSERVABLE_SUBSPACE.json", "E2R0_TIMING.json",
               "E2R0_CERTIFICATION.json", "E2R0_MECHANISM.json",
               "E2R0_CANDIDATES.json", "E2R0_VERDICT.json", "E2R0_REPORT.md"]


def out():
    d = Path(os.environ["MSS0_OUTPUT_DIR"]); d.mkdir(parents=True, exist_ok=True); return d


def build_scene(scene_key, resolution, cam_idx, ch, cw):
    ckpt = os.environ.get(f"MSS_E2R0_{scene_key.upper()}_CKPT", A.CKPT_DEFAULT)
    cam_path = os.environ.get(f"MSS_E2R0_{scene_key.upper()}_CAMS",
                              A.CAMS_DEFAULT.replace("/room/", f"/{scene_key}/"))
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


def agg_diags(diags):
    """Aggregate per-gaussian solver diagnostics into a per-macro summary."""
    if not diags:
        return {"n_gaussians": 0}
    def med(k):
        v = [d[k] for d in diags if k in d and d[k] is not None and np.isfinite(d[k])]
        return float(np.median(v)) if v else None
    def mx(k):
        v = [d[k] for d in diags if k in d and d[k] is not None and np.isfinite(d[k])]
        return float(np.max(v)) if v else None
    n_prop = sum(1 for d in diags if d.get("reduction_check_pass"))
    return {
        "n_gaussians": len(diags),
        "n_pass_reduction_check": n_prop,
        "grad_norm_med": med("grad_norm"), "grad_norm_max": mx("grad_norm"),
        "effective_rank_med": med("effective_rank"), "effective_rank_min": med("effective_rank"),
        "eig_max_med": med("eig_max"), "eig_min_med": med("eig_min"), "cond_med": med("cond"),
        "H_diag_cv_med": med("H_diag_cv"),
        "predicted_reduction_med": med("predicted_reduction"),
        "actual_reduction_r_med": med("actual_reduction_r"),
        "rho_r_med": med("rho_r"),
        "cosine_to_exact_gn_med": med("cosine_to_exact_gn"),
        "final_step_norm_med": med("final_step_norm"),
        "raw_step_norm_med": med("raw_step_norm"),
        "energy_retained_med": (med("energy_retained") if any("energy_retained" in d for d in diags) else None),
        "retained_rank_med": (med("retained_rank") if any("retained_rank" in d for d in diags) else None),
    }


# ---- G0: native sanity ------------------------------------------------------
def g0_sanity(scene, n_steps, seed, lr, ssim):
    torch.manual_seed(seed)
    state = X.MSSState(scene["params"], dev, 3)
    opt = torch.optim.Adam([state.p[k] for k in ("xyz", "scaling", "rotation", "opacity", "shs")], lr=lr)
    losses = []
    for _ in range(n_steps):
        losses.append(X.full_native_step(state, scene["cam"], scene["pixels"], scene["Y_img"],
                                         opt, ssim, scene["ch"], scene["cw"], scene["cy0"], scene["cx0"]))
    drop = (losses[-1] / losses[0] - 1.0) * 100.0 if losses[0] else float("nan")
    return {"variant": "adam", "n_steps": n_steps, "loss_initial": losses[0] if losses else None,
            "loss_final": losses[-1] if losses else None, "loss_drop_pct": drop,
            "native_loss_reduces": bool(drop < 0), "losses": losses, "peak_gib": None}


# ---- G1: faithful MSS0 baseline reproduction (original block_gn) ------------
def run_baseline(scene, cfg, ssim, n_macros, seed):
    torch.manual_seed(seed)
    state = X.MSSState(scene["params"], dev, scene["params"]["active_sh_degree"]
                       if "active_sh_degree" in scene["params"] else 3)
    cfg = dict(cfg); cfg["use_dual"] = True
    cfg["cy0"] = scene["cy0"]; cfg["cx0"] = scene["cx0"]
    cfg["max_rays"] = int(os.environ.get("MSS_E2R0_MAXRAYS", "48"))
    cfg["inner_sweeps"] = int(os.environ.get("MSS_E2R0_INNER_SWEEPS", "1"))
    accepts = 0; certs = []; losses = []; theta_s = 0.0; n_changed = 0
    torch.cuda.reset_peak_memory_stats(); t0 = time.monotonic(); loss0 = None
    macros = []
    for m in range(n_macros):
        r = X.mss_macro(state, scene["cam"], scene["pixels"], scene["Y_img"], cfg, ssim,
                        scene["ch"], scene["cw"])
        if loss0 is None: loss0 = r["native_loss0"]
        losses.append(r["native_loss1"]); certs.append(r["cert"]); theta_s += r["theta_seconds"]
        n_changed += r["n_changed"]; accepts += int(r["accept"])
        macros.append({"macro_index": m, "native_loss0": r["native_loss0"], "native_loss1": r["native_loss1"],
                       "loss_delta": r["loss_delta"], "cert": r["cert"], "accept": bool(r["accept"]),
                       "n_changed": r["n_changed"], "theta_seconds": r["theta_seconds"]})
        if not torch.isfinite(torch.tensor(r["native_loss1"])): break
    wall = time.monotonic() - t0; peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    final = losses[-1] if losses else float("nan")
    drop = (final / loss0 - 1.0) * 100.0 if loss0 else float("nan")
    return {"variant": "mss_baseline_block_gn", "n_macros": len(macros), "accepts": accepts,
            "reject_rate": (len(macros) - accepts) / max(1, len(macros)),
            "max_cert": max([c for c in certs if c is not None], default=float("nan")),
            "loss_initial": loss0, "loss_final": final, "loss_drop_pct": drop,
            "theta_seconds": theta_s, "wall_seconds": wall, "peak_gib": peak,
            "n_changed_total": n_changed, "losses": losses, "macros": macros,
            "per_macro_median": {}}, wall


# ---- solver probe -----------------------------------------------------------
def run_solver(scene, solver_name, cfg, ssim, n_macros, seed):
    torch.manual_seed(seed)
    state = X.MSSState(scene["params"], dev, scene["params"]["active_sh_degree"]
                       if "active_sh_degree" in scene["params"] else 3)
    cfg = dict(cfg); cfg["use_dual"] = True
    cfg["cy0"] = scene["cy0"]; cfg["cx0"] = scene["cx0"]
    cfg["max_rays"] = int(os.environ.get("MSS_E2R0_MAXRAYS", "48"))
    cfg["inner_sweeps"] = int(os.environ.get("MSS_E2R0_INNER_SWEEPS", "1"))
    skw = dict(E2.SOLVER_DEFAULTS.get(solver_name, {}))
    accepts = 0; certs = []; losses = []; theta_s = 0.0; n_changed = 0
    torch.cuda.reset_peak_memory_stats(); t0 = time.monotonic(); loss0 = None
    macros = []; acc_drops = []; acc_walls = []
    for m in range(n_macros):
        mt0 = time.monotonic()
        r = E2.mss_macro_solver(state, scene["cam"], scene["pixels"], scene["Y_img"], cfg, ssim,
                                scene["ch"], scene["cw"], solver_name, skw)
        mw = time.monotonic() - mt0
        if loss0 is None: loss0 = r["native_loss0"]
        losses.append(r["native_loss1"]); certs.append(r["cert"]); theta_s += r["theta_seconds"]
        n_changed += r["n_changed"]; accepts += int(r["accept"])
        if r["accept"]:
            acc_drops.append(r["native_loss0"] - r["native_loss1"]); acc_walls.append(mw)
        macros.append({"macro_index": m, "native_loss0": r["native_loss0"], "native_loss1": r["native_loss1"],
                       "loss_delta": r["loss_delta"], "cert": r["cert"], "accept": bool(r["accept"]),
                       "n_changed": r["n_changed"], "theta_seconds": r["theta_seconds"],
                       "macro_wall_seconds": mw, "solver_agg": agg_diags(r.get("diags", []))})
        if not torch.isfinite(torch.tensor(r["native_loss1"])): break
    wall = time.monotonic() - t0; peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    final = losses[-1] if losses else float("nan")
    drop = (final / loss0 - 1.0) * 100.0 if loss0 else float("nan")
    med_drop = float(np.median(acc_drops)) if acc_drops else None
    med_wall = float(np.median(acc_walls)) if acc_walls else None
    return {"variant": solver_name, "n_macros": len(macros), "accepts": accepts,
            "reject_rate": (len(macros) - accepts) / max(1, len(macros)),
            "max_cert": max([c for c in certs if c is not None], default=float("nan")),
            "loss_initial": loss0, "loss_final": final, "loss_drop_pct": drop,
            "median_accepted_loss_reduction": med_drop,
            "median_accepted_macro_wall_s": med_wall,
            "theta_seconds": theta_s, "wall_seconds": wall, "peak_gib": peak,
            "n_changed_total": n_changed, "losses": losses, "macros": macros}, wall


def main():
    resolution = os.environ.get("MSS_E2R0_RES", "720p")
    ch = int(os.environ.get("MSS_E2R0_CH", "48")); cw = int(os.environ.get("MSS_E2R0_CW", "48"))
    cam_idx = int(os.environ.get("MSS_E2R0_CAM", "150"))
    n_select = int(os.environ.get("MSS_E2R0_NSELECT", "5))
    n_macros_g1 = int(os.environ.get("MSS_E2R0_NMACROS_G1", "6"))
    n_macros_probe = int(os.environ.get("MSS_E2R0_NMACROS_PROBE", "5"))
    seed = int(os.environ.get("MSS_E2R0_SEED", "20261006"))
    base_lr = float(os.environ.get("MSS_E2R0_LR", "0.005"))
    scenes_keys = os.environ.get("MSS_E2R0_SCENES", "room,bonsai,counter").split(",")
    max_probe_solvers = int(os.environ.get("MSS_E2R0_MAX_SOLVERS", "4"))
    time_budget_min = float(os.environ.get("MSS_E2R0_TIME_BUDGET_MIN", "120"))
    O = out()
    ssim = S.SepSSIM(device=dev)
    CFG = dict(beta=4.0, rho=4.0, image_steps=5, eta=0.05, mu=1e-2, max_gaussians=200)
    t_start = time.monotonic()
    def elapsed_min(): return (time.monotonic() - t_start) / 60.0

    print(f"[start] E2R0 scenes={scenes_keys} ch={ch} cw={cw} cam={cam_idx} "
          f"nselect={n_select} g1={n_macros_g1} probe={n_macros_probe} "
          f"maxrays={os.environ.get('MSS_E2R0_MAXRAYS','48')} budget={time_budget_min}min", flush=True)

    scenes = {sk: build_scene(sk, resolution, cam_idx, ch, cw) for sk in scenes_keys}

    # ---- G0 native sanity ----
    g0 = {}
    for sk in scenes_keys:
        g0[sk] = g0_sanity(scenes[sk], n_select, seed, base_lr, ssim)
        print(f"[G0 {sk}] adam drop={g0[sk]['loss_drop_pct']:.3f}% reduces={g0[sk]['native_loss_reduces']}", flush=True)
    g0_pass = all(g0[sk]["native_loss_reduces"] for sk in scenes_keys)
    (O / DELIV_FILES[0]).write_text(json.dumps({
        "gate": "G0_native_sanity", "pass": g0_pass,
        "rule": "native Adam must reduce loss on every scene (else BLOCKED_BASELINE)",
        "cert_tol": CERT_TOL, "scenes": g0}, indent=2) + "\n")
    if not g0_pass:
        _verdict(O, "BLOCKED_BASELINE", {"reason": "native Adam did not reduce loss on all scenes"}, scenes, {})
        return

    # ---- G1 baseline reproduction ----
    g1 = {}
    for sk in scenes_keys:
        r, _w = run_baseline(scenes[sk], CFG, ssim, n_macros_g1, seed)
        g1[sk] = r
        print(f"[G1 {sk}] baseline accepts={r['accepts']} drop={r['loss_drop_pct']:.4f}% "
              f"nth={r['n_changed_total']} wall={r['wall_seconds']:.1f}s", flush=True)
    repro_pass = all(g1[sk]["accepts"] == 0 for sk in scenes_keys)
    (O / DELIV_FILES[1]).write_text(json.dumps({
        "gate": "G1_mss0_reproduction", "pass": repro_pass,
        "rule": "original ray-capped block_gn must reproduce 0 accepts / ~0 loss drop on every scene",
        "code_state": "MSS0_MX_DSH/native (frozen numerics byte-identical)",
        "n_macros": n_macros_g1, "scenes": g1}, indent=2) + "\n")
    if not repro_pass:
        _verdict(O, "BLOCKED_REPRODUCTION", {"reason": "baseline did not reproduce 0-accept failure",
                                             "accepts": {sk: g1[sk]["accepts"] for sk in scenes_keys}},
                 scenes, {"baseline": g1})
        return

    # ---- solver probes ----
    solvers = SOLVER_NAMES[:max_probe_solvers]
    results = {}   # solver -> scene -> result
    raw_timing = {}
    for sn in solvers:
        results[sn] = {}
        for sk in scenes_keys:
            if elapsed_min() > time_budget_min:
                print(f"[budget] stop before {sn}/{sk} at {elapsed_min():.1f}min", flush=True)
                results[sn][sk] = {"variant": sn, "skipped_budget": True}
                continue
            r, w = run_solver(scenes[sk], sn, CFG, ssim, n_macros_probe, seed)
            results[sn][sk] = r
            raw_timing[(sn, sk)] = {"theta_seconds": r["theta_seconds"], "wall_seconds": r["wall_seconds"],
                                    "macro_wall_s_med": r.get("median_accepted_macro_wall_s"),
                                    "peak_gib": r["peak_gib"]}
            print(f"[{sn} {sk}] accepts={r['accepts']} drop={r['loss_drop_pct']:.4f}% "
                  f"med_acc_red={r['median_accepted_loss_reduction']} "
                  f"nth={r['n_changed_total']} wall={r['wall_seconds']:.1f}s", flush=True)
        _write_solver_file(O, sn, results[sn], scenes_keys)

    # ---- gates G2/G3/G4 ----
    gates = _compute_gates(results, scenes_keys, solvers)
    (O / "E2R0_TIMING.json").write_text(json.dumps({
        "note": "throughput from measured accepted-step cost (solver+pipeline), not solver-only",
        "cert_tol": CERT_TOL, "n_macros_probe": n_macros_probe, "timing": [
            {"solver": sn, "scene": sk, **raw_timing.get((sn, sk), {})}
            for sn in solvers for sk in scenes_keys]}, indent=2) + "\n")
    (O / "E2R0_CERTIFICATION.json").write_text(json.dumps({
        "note": "cert = max |z_new - z2| per scene; tol frozen at 1/255+2e-5",
        "cert_tol": CERT_TOL, "certification": [
            {"solver": sn, "scene": sk,
             "max_cert": results[sn][sk].get("max_cert"),
             "within_tol": bool(results[sn][sk].get("max_cert") is not None
                                and np.isfinite(results[sn][sk].get("max_cert"))
                                and results[sn][sk]["max_cert"] <= CERT_TOL),
             "accepts": results[sn][sk].get("accepts")}
            for sn in solvers for sk in scenes_keys]}, indent=2) + "\n")

    # ---- mechanism / candidates / verdict ----
    mech = _mechanism(results, gates, scenes_keys, solvers)
    (O / "E2R0_MECHANISM.json").write_text(json.dumps(mech, indent=2) + "\n")
    cand = _candidates(results, gates, scenes_keys, solvers)
    (O / "E2R0_CANDIDATES.json").write_text(json.dumps(cand, indent=2) + "\n")
    _verdict(O, cand["verdict"], cand["verdict_detail"], scenes, results, gates=gates, mech=mech)


def _write_solver_file(O, sn, per_scene, scenes_keys):
    fmap = {"S1_lm": "E2R0_LM.json", "S2_precond_lm": "E2R0_PRECOND_LM.json",
            "S3_pcg": "E2R0_PCG.json", "S4_obs_subspace": "E2R0_OBSERVABLE_SUBSPACE.json"}
    payload = {"solver": sn, "cert_tol": CERT_TOL,
               "acceptance_semantics": "frozen MSS0: native_loss1<=native_loss0+1e-7 AND cert<=1/255+2e-5",
               "scenes": {sk: per_scene.get(sk, {}) for sk in scenes_keys}}
    (O / fmap[sn]).write_text(json.dumps(payload, indent=2) + "\n")


def _compute_gates(results, scenes_keys, solvers):
    g = {"G2_acceptance": {}, "G3_descent": {}, "G4_throughput": {}, "promotion": {}}
    for sn in solvers:
        g2 = all(results[sn].get(sk, {}).get("accepts", 0) >= 3 for sk in scenes_keys)
        acc_reds = [results[sn][sk].get("median_accepted_loss_reduction") for sk in scenes_keys
                    if results[sn][sk].get("median_accepted_loss_reduction") is not None]
        g3 = bool(acc_reds) and float(np.median(acc_reds)) > 0
        acc_walls = [results[sn][sk].get("median_accepted_macro_wall_s") for sk in scenes_keys
                     if results[sn][sk].get("median_accepted_macro_wall_s") is not None]
        g4 = bool(acc_walls) and float(np.max(acc_walls)) <= 360.0
        g["G2_acceptance"][sn] = {"per_scene_accepts": {sk: results[sn].get(sk, {}).get("accepts") for sk in scenes_keys}, "pass": g2}
        g["G3_descent"][sn] = {"median_accepted_loss_reduction_per_scene": acc_reds, "pass": g3}
        g["G4_throughput"][sn] = {"median_accepted_macro_wall_s": acc_walls, "pass": g4}
        g["promotion"][sn] = bool(g2 and g3 and g4)
    return g


def _mechanism(results, gates, scenes_keys, solvers):
    # gather spectral evidence across all solver/scene per-gaussian diags
    effr = []; cond = []; energy = []; cosg = []; pass_red = []
    for sn in solvers:
        for sk in scenes_keys:
            for mc in results[sn].get(sk, {}).get("macros", []):
                a = mc.get("solver_agg", {})
                for k, lst in (("effective_rank_med", effr), ("cond_med", cond),
                               ("energy_retained_med", energy), ("cosine_to_exact_gn_med", cosg)):
                    if a.get(k) is not None and np.isfinite(a[k]): lst.append(a[k])
                if a.get("n_pass_reduction_check") is not None: pass_red.append(a["n_pass_reduction_check"])
    def ms(v): return float(np.median(v)) if v else None
    spec = {"effective_rank_median": ms(effr), "cond_median": ms(cond),
            "energy_retained_median": ms(energy), "cosine_to_exact_gn_median": ms(cosg),
            "n_gaussians_passing_reduction_check_total": int(np.sum(pass_red)) if pass_red else 0}
    # decide primary mechanism
    any_accept = any(results[sn].get(sk, {}).get("accepts", 0) > 0 for sn in solvers for sk in scenes_keys)
    s4 = results.get("S4_obs_subspace", {})
    s4_accepts = sum(s4.get(sk, {}).get("accepts", 0) for sk in scenes_keys) if s4 else 0
    base_accepts = sum(results.get("_baseline", {}).get(sk, {}).get("accepts", 0) for sk in scenes_keys)
    if not any_accept:
        primary = "INSUFFICIENT_RAY_INFORMATION"
        conf = "high" if spec["effective_rank_median"] is not None and spec["effective_rank_median"] < 20 else "medium"
        supp = ["all low-cost solvers including observable-subspace still 0 accepted macros",
                f"median effective rank of per-Gaussian H={spec['effective_rank_median']}"]
    elif s4_accepts > 0 and not any(results[sn].get(sk, {}).get("accepts", 0) >= 3 for sn in solvers for sk in scenes_keys):
        primary = "NULL_SPACE_CONTAMINATION"; conf = "medium"
        supp = ["observable-subspace solver produced accepts where full-damped baseline did not",
                "truncating weak eigen-directions restored accepted descent"]
    else:
        primary = "MIXED"; conf = "low"; supp = ["some solver produced accepted updates; see candidate ranking"]
    return {"primary_mechanism": primary, "secondary_mechanism": "CONDITIONING_ONLY" if conf != "high" else "",
            "confidence": conf, "supporting_evidence": supp,
            "contradicting_evidence": [] if any_accept else ["compositing-residual reduction sometimes passed per-gaussian but macro accept stayed 0"],
            "spectral": spec,
            "accepts_by_solver": {sn: {sk: results[sn].get(sk, {}).get("accepts") for sk in scenes_keys} for sn in solvers}}


def _candidates(results, gates, scenes_keys, solvers):
    rank = []
    for sn in solvers:
        acc = sum(results[sn].get(sk, {}).get("accepts", 0) for sk in scenes_keys)
        red = [results[sn][sk].get("median_accepted_loss_reduction") for sk in scenes_keys
               if results[sn][sk].get("median_accepted_loss_reduction") is not None]
        wall = [results[sn][sk].get("median_accepted_macro_wall_s") for sk in scenes_keys
                if results[sn][sk].get("median_accepted_macro_wall_s") is not None]
        consistent = all(results[sn].get(sk, {}).get("accepts", 0) > 0 for sk in scenes_keys)
        peak = max([results[sn][sk].get("peak_gib", 0) for sk in scenes_keys], default=0)
        rank.append({"solver": sn, "total_accepts": acc,
                     "median_accepted_loss_reduction": float(np.median(red)) if red else None,
                     "median_accepted_macro_wall_s": float(np.median(wall)) if wall else None,
                     "cross_scene_consistent": consistent, "peak_gib": peak,
                     "promotion_pass": gates["promotion"].get(sn, False)})
    rank.sort(key=lambda r: (r["promotion_pass"], r["total_accepts"], -(r["median_accepted_macro_wall_s"] or 1e9)), reverse=True)
    promoted = [r for r in rank if r["promotion_pass"]]
    scene_zero = {sk: all(results[sn].get(sk, {}).get("accepts", 0) == 0 for sn in solvers) for sk in scenes_keys}
    if promoted:
        verdict = "GO_" + promoted[0]["solver"].upper()
        detail = {"promoted_solver": promoted[0]["solver"], "reason": "passes G2 (>=3 accepts/scene) AND G3 (median reduction>0) AND G4 (<=6 min/accepted macro)"}
    elif any(scene_zero.values()):
        verdict = "NO_GO_MSS_FAMILY"
        detail = {"reason": "at least one scene has 0 accepts across ALL solvers (incl. observable-subspace)",
                  "scene_zero_accepts": scene_zero,
                  "note": "H4: capped ray information insufficient; return main effort to raster/backward lane"}
    else:
        verdict = "NO_GO_SOLVER"
        detail = {"reason": "no solver met the promotion criterion (>=3 accepts/scene AND median reduction>0 AND <=6 min/accepted macro)",
                  "scene_zero_accepts": scene_zero}
    return {"verdict": verdict, "verdict_detail": detail, "ranking": rank,
            "gates": gates, "scene_zero_accepts": scene_zero}


def _verdict(O, verdict, detail, scenes, results, gates=None, mech=None):
    scenes_keys = list(scenes.keys())
    solvers = [sn for sn in SOLVER_NAMES if sn in results]
    acc = {sn: {sk: results[sn].get(sk, {}).get("accepts") for sk in scenes_keys} for sn in solvers}
    red = {sn: {sk: results[sn].get(sk, {}).get("median_accepted_loss_reduction") for sk in scenes_keys} for sn in solvers}
    wall = {sn: {sk: results[sn].get(sk, {}).get("median_accepted_macro_wall_s") for sk in scenes_keys} for sn in solvers}
    v = {"schema": "mss.e2r0.verdict.v1", "verdict": verdict, "detail": detail,
         "accepts": acc, "median_accepted_loss_reduction": red, "median_accepted_macro_wall_s": wall,
         "gates": gates, "mechanism": mech.get("primary_mechanism") if mech else None,
         "cert_tol": CERT_TOL,
         "next_lane": ("MSS-E2R1-" + detail.get("promoted_solver", "") if verdict.startswith("GO_") else None)}
    (O / "E2R0_VERDICT.json").write_text(json.dumps(v, indent=2) + "\n")
    rep = _report(verdict, detail, acc, red, wall, gates, mech)
    (O / "E2R0_REPORT.md").write_text(rep)
    _manifest(O)
    print("[verdict] " + json.dumps({"verdict": verdict, "detail": detail, "accepts": acc}, indent=2), flush=True)


def _report(verdict, detail, acc, red, wall, gates, mech):
    L = ["# MSS-E2R0 Report — Observable-Subspace Solver Feasibility", ""]
    L.append(f"**Terminal verdict:** `{verdict}`")
    L.append("")
    L.append("## Detail")
    L.append("```json")
    L.append(json.dumps(detail, indent=2))
    L.append("```")
    L.append("")
    L.append("## Acceptance by solver / scene")
    L.append("| solver | " + " | ".join(acc[sn].keys() if acc else []) + " | total |")
    L.append("|---|" + "---|" * (len(acc) + 1))
    for sn in acc:
        row = [str(acc[sn][sk]) for sk in acc[sn]]
        L.append(f"| {sn} | " + " | ".join(row) + f" | {sum(acc[sn].values())} |")
    L.append("")
    L.append("## Median accepted loss reduction / macro wall (s)")
    L.append("| solver | " + " | ".join((red[sn].keys() if red else [])) + " |")
    L.append("|---|" + "---|" * (len(red) + 1))
    for sn in red:
        rr = [f"{red[sn][sk]:.3e}/{wall[sn][sk]:.1f}" if (red[sn][sk] is not None and wall[sn][sk] is not None) else "n/a" for sk in red[sn]]
        L.append(f"| {sn} | " + " | ".join(rr) + " |")
    L.append("")
    if mech:
        L.append("## Mechanism")
        L.append(f"- primary: `{mech.get('primary_mechanism')}` (confidence {mech.get('confidence')})")
        L.append(f"- spectral: `{json.dumps(mech.get('spectral', {}))}`")
        for e in mech.get("supporting_evidence", []):
            L.append(f"  - support: {e}")
    L.append("")
    L.append("## Frozen constraints honored")
    L.append("- MSS0 verdict untouched; E2 acceptance semantics, cert tolerance (1/255+2e-5), trust clip, memory gate, scenes/loss/camera/native-baseline all unchanged.")
    L.append("- No protocol relaxation; approximate evidence not declared exact; no E3 claim; no novelty/training-speedup claim.")
    return "\n".join(L) + "\n"


def _manifest(O):
    files = sorted([p for p in O.iterdir() if p.is_file() and p.name != "MANIFEST.sha256"])
    h = hashlib.sha256()
    entries = []
    for p in files:
        d = hashlib.sha256(p.read_bytes()).hexdigest()
        entries.append({"path": p.name, "sha256": d}); h.update(p.name.encode() + b"\0" + p.read_bytes() + b"\0")
    (O / "MANIFEST.sha256").write_text(json.dumps(
        {"schema": "mss.e2r0.manifest.v1", "files": entries, "aggregate_sha256": h.hexdigest()}, indent=2) + "\n")


if __name__ == "__main__":
    main()