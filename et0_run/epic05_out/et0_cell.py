#!/usr/bin/env python3
"""ET0 per-cell driver (ET0_PREREG.md, FROZEN). Executes on remote EPIC-05 A100
(NEVER local GPU). Reuses frozen C0 primitives via a0_lib (A0/IR0 instrumentation);
renderer driven read-only through the same runtime capture patch. No production
CUDA modified. No densification.

Modes:
  (default)          phase-1 sweep: per source cam -> g, native Adam d, alpha
                     sweep of L_src / C_tile / C_order / H / E / sign-fail,
                     rho, AUC, alpha_C. Dumps per-probe curves + H(alpha=1) for
                     cohort tau_H.
  --safestep TAU_H   phase-2: recompute d, render source+7peer+8holdout at
                     alpha_C/alpha_H/1.0, G_C + holdout PSNR delta.
  --aggregate        read all per-cell dumps -> ET0_SUMMARY.json (schema cells).
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, "/root/epic05_moonshot/a0")
import a0_lib as L
import et0_core as E
import et0_shm; et0_shm.install()

CKPT_HASH = {
    ("room", 10000): "9aa00f0ddf28b6c097e3dc8f41dbb107cd9bf175750503dc330aa1d6ab579fc4",
    ("room", 15000): "b276b338f5a444d8ea1be390d21fd3e67679b7d862652f0babd131db9434e5dd",
    ("room", 20000): "c4e41cfef79e50980406fc1fa148af8d04bc8b37b801c36da5cddb2ce8058718",
    ("bicycle", 10000): "26bf0b959f10f0ce526c496d0dc7ca8e62b0ba93efc83d58b7f9440d58c2f4e9",
    ("bicycle", 15000): "273971d3867c1d2a08b7246803265539d74756b8eca4e3f01c6a8d149b7fd6f7",
    ("bicycle", 20000): "96f136cb331a19900a41a516d53314d522b4de5a3250ddbba8a232b610b398f0",
}
PREREG_PIN_C0 = "d26cccf46c7932bfb5fd44ee05e553af27e17c93eaffb7dacf3197258a513f11"
PREREG_PIN_CORE = "935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e"


def jdump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=L.json_default)
    print(f"[wrote] {path}", flush=True)


def median(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.median(v)) if v else None


def p10(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.percentile(v, 10)) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--ckpt", type=int, required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--mode", choices=["sweep", "safestep", "aggregate"], default="sweep")
    ap.add_argument("--tau-h", type=float, default=None)
    ap.add_argument("--cells", default="")
    args = ap.parse_args()

    if args.mode == "aggregate":
        return aggregate(args)

    scene, it = args.scene, args.ckpt
    cell_id = f"{scene}_{it}"
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    LOG = lambda *a: print(f"[{time.time()-t0:8.1f}s]", *a, flush=True)

    # ---- identity + environment gate (fail-closed) ----
    ident_env = L.bootstrap_c0()
    assert ident_env["so_sha256"] == PREREG_PIN_C0, "C0 EXT MISMATCH"
    assert ident_env["core_sha256"] == PREREG_PIN_CORE, "CORE SO MISMATCH"
    ds = L.load_dataset(scene, device="cuda")
    m, opt_state, ident = L.load_model(scene, it)
    ckpt_sha = ident["ckpt_sha256"]
    assert ckpt_sha == CKPT_HASH[(scene, it)], f"CKPT HASH MISMATCH {scene}@{it}"
    N = int(ident["num_points"])
    n_cams = len(ds)
    n, src_cams, hold_cams, train_cams, hold_all = E.source_and_holdout(ds)
    LOG(f"cell={cell_id} N={N} n_cams={n_cams} src={src_cams} hold={hold_cams}")

    env_doc = {
        "cell": cell_id, "arm": ident_env["arm"],
        "so_sha256": ident_env["so_sha256"], "core_sha256": ident_env["core_sha256"],
        "ckpt_path": ident["ckpt_path"], "ckpt_sha256": ckpt_sha, "scene": scene,
        "ckpt_iter": ident["ckpt_iter"], "num_points": N,
        "active_sh_degree": ident["active_sh_degree"],
        "spatial_lr_scale": ident["spatial_lr_scale"], "group_order": ident["group_order"],
        "n_cams": n_cams, "train_pool_size": len(train_cams), "holdout_size": len(hold_all),
        "source_cams": src_cams, "holdout_cams": hold_cams,
        "camera_rule": "train: i%8!=0 midpoint-stratified(8); holdout: i%8==0 midpoint-stratified(8)",
        "adam": {"beta1": E.BETA1, "beta2": E.BETA2, "eps": E.EPS_ADAM,
                 "per_group_lr": {g: opt_state[g]["lr"] for g in E.GROUPS},
                 "per_group_step": {g: opt_state[g]["step"] for g in E.GROUPS}},
        "python": sys.version.split()[0], "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        env_doc["gpu"] = torch.cuda.get_device_name(0)
    jdump(env_doc, os.path.join(outdir, f"ET0_ENVIRONMENT_{cell_id}.json"))

    patch_info, uninstall = L.install_capture_patch()
    SH = E.SH_DEGREE

    def set_params(theta):
        for g in E.GROUPS:
            m[g].data = theta[g]

    def render_loss(cam, gt):
        frame, _, _, _ = L.render_c0(m, cam, SH)
        return float(L.train_loss(frame, gt).item()), frame

    def capture_state(cam, gt):
        L.A0_CAPTURE["enabled"] = True
        L.A0_CAPTURE["records"].clear()
        frame, _, _, _ = L.render_c0(m, cam, SH)
        loss_t = L.train_loss(frame, gt)
        loss_t.backward()
        cap = L.parse_capture(L.A0_CAPTURE["records"][0])
        L.A0_CAPTURE["records"].clear()
        L.A0_CAPTURE["enabled"] = False
        return float(loss_t.item()), cap

    def zero_grads():
        for g in E.GROUPS:
            m[g].grad = None

    n_src = 2 if args.smoke else 8
    src_cams = src_cams[:n_src]
    alphas = E.ALPHAS

    if args.mode == "sweep":
        return run_sweep(args, scene, it, cell_id, outdir, ds, m, opt_state, ident,
                         src_cams, hold_cams, alphas, set_params, render_loss,
                         capture_state, zero_grads, N, LOG, n_cams, uninstall)

    if args.mode == "safestep":
        assert args.tau_h is not None, "safestep needs --tau-h"
        return run_safestep(args, scene, it, cell_id, outdir, ds, m, opt_state,
                            src_cams, hold_cams, alphas, args.tau_h, set_params,
                            render_loss, capture_state, zero_grads, N, LOG, uninstall)


def _forward_g_and_capture(m, opt_state, ds, ci, set_params, capture_state, zero_grads):
    """theta_0 native gradient g (source cam) + theta_0 capture state, in ONE
    backward pass (trusted IR0 pattern: capture records saved state during the
    same _native_backward that produces g)."""
    cam, gt = ds.get_item(ci)
    zero_grads()
    L.A0_CAPTURE["enabled"] = True
    L.A0_CAPTURE["records"].clear()
    frame, _, _, _ = L.render_c0(m, cam, E.SH_DEGREE)
    loss = L.train_loss(frame, gt)
    loss.backward()
    g = {grp: (m[grp].grad.detach().clone() if m[grp].grad is not None
               else torch.zeros_like(m[grp])) for grp in E.GROUPS}
    cap0 = L.parse_capture(L.A0_CAPTURE["records"][0])
    L.A0_CAPTURE["records"].clear()
    L.A0_CAPTURE["enabled"] = False
    return g, float(loss.item()), cap0

def run_sweep(args, scene, it, cell_id, outdir, ds, m, opt_state, ident,
              src_cams, hold_cams, alphas, set_params, render_loss,
              capture_state, zero_grads, N, LOG, n_cams, uninstall):
    smoke = args.smoke
    probes = []
    for k, ci in enumerate(src_cams):
        cam, gt = ds.get_item(ci)
        # snapshot theta_0
        theta0 = {g: m[g].detach().clone() for g in E.GROUPS}
        g, L0, cap0 = _forward_g_and_capture(m, opt_state, ds, ci, set_params,
                                             capture_state, zero_grads)
        BASE = N + 1  # fixed key multiplier (total Gaussians + 1)
        ts0 = E.tile_state(cap0, BASE)
        # native Adam displacement d = theta_adam - theta_0
        d = {grp: E.adam_update(g[grp], opt_state[grp]) for grp in E.GROUPS}
        gd = float(sum((g[grp] * d[grp]).sum().item() for grp in E.GROUPS))
        set_params(theta0)  # restore exactly

        rows = []
        H1 = None
        for alpha in alphas:
            theta = {grp: theta0[grp] + alpha * d[grp] for grp in E.GROUPS}
            set_params(theta)
            # validity: finite quats/scales/opacity
            valid = bool(torch.isfinite(theta["rotation"]).all()
                         and torch.isfinite(theta["scaling"]).all()
                         and torch.isfinite(theta["opacity"]).all())
            La, capA = capture_state(cam, gt)
            tsA = E.tile_state(capA, BASE)
            C_tile = E.c_tile(ts0[0], tsA[0])
            C_order, insdel = E.c_order(ts0, tsA, BASE)
            C = 0.5 * C_tile + 0.5 * C_order
            H = float(E.hellinger_sq(theta0["xyz"], theta0["rotation"],
                                     theta0["scaling"], theta["xyz"],
                                     theta["rotation"], theta["scaling"]).mean().item())
            pred = -alpha * gd
            real = L0 - La
            denom = abs(pred) + abs(real) + 1e-12
            Eerr = abs(pred - real) / denom
            sign_fail = bool(pred > 0 and real <= 0)
            rows.append({"alpha": alpha, "L_src": La, "pred": pred, "real": real,
                         "E": Eerr, "sign_failure": sign_fail, "C_tile": C_tile,
                         "C_order": C_order, "C": C, "H": H, "insdel": insdel,
                         "valid": valid})
            if alpha == 1.0:
                H1 = H
            del capA, tsA
            torch.cuda.empty_cache()
        set_params(theta0)

        # rho / AUC over the 8 alpha points
        Ea = np.array([r["E"] for r in rows])
        Ca = np.array([r["C"] for r in rows])
        Ha = np.array([r["H"] for r in rows])
        labels = np.array([1 if (r["E"] >= 0.50 or r["sign_failure"]) else 0 for r in rows])
        rho_EC = E.spearman(Ea, Ca)
        rho_EH = E.spearman(Ea, Ha)
        delta_rho = rho_EC - rho_EH if not (np.isnan(rho_EC) or np.isnan(rho_EH)) else float("nan")
        AUC_C = E.auc(labels, Ca)
        AUC_H = E.auc(labels, Ha)
        delta_AUC = AUC_C - AUC_H if not (np.isnan(AUC_C) or np.isnan(AUC_H)) else float("nan")
        # alpha_C = max alpha with C<=tau_C
        okC = [r["alpha"] for r in rows if r["C"] <= E.TAU_C and r["valid"]]
        alpha_C = max(okC) if okC else None
        probe = {
            "probe": k, "cam": int(ci), "L_src_theta0": L0, "gd": gd,
            "rows": rows, "rho_EC": rho_EC, "rho_EH": rho_EH,
            "delta_rho": delta_rho, "AUC_C": AUC_C, "AUC_H": AUC_H,
            "delta_AUC": delta_AUC, "alpha_C": alpha_C, "H_at_alpha1": H1,
            "n_positive_labels": int(labels.sum()),
        }
        probes.append(probe)
        LOG(f"cam {k} (id {ci}) L0={L0:.5f} gd={gd:.4e} rhoEC={rho_EC:.3f} "
            f"rhoEH={rho_EH:.3f} dAUC={delta_AUC:.3f} alpha_C={alpha_C} H1={H1:.4e}")
        del cap0, ts0
        torch.cuda.empty_cache()

    uninstall()
    dump = {"cell": cell_id, "scene": scene, "ckpt": it, "mode": "sweep",
            "smoke": smoke, "tau_C": E.TAU_C, "alphas": alphas, "probes": probes}
    jdump(dump, os.path.join(outdir, f"ET0_SWEEP_{cell_id}.json"))
    h1s = [p["H_at_alpha1"] for p in probes if p["H_at_alpha1"] is not None]
    jdump({"cell": cell_id, "H_at_alpha1": h1s,
           "cohort_tau_H_input": True},
          os.path.join(outdir, f"ET0_H1_{cell_id}.json"))
    LOG("sweep done")


def run_safestep(args, scene, it, cell_id, outdir, ds, m, opt_state,
                 src_cams, hold_cams, alphas, tau_H, set_params,
                 render_loss, capture_state, zero_grads, N, LOG, uninstall):
    """ET0 S9: safe-step opportunity. G_C uses source losses already in the
    sweep (free). Renders 7 peer-train + 8 true-holdout at alpha_C/alpha_H/1.0."""
    sweep = json.load(open(os.path.join(outdir, f"ET0_SWEEP_{cell_id}.json")))
    smap = {p["cam"]: p for p in sweep["probes"]}
    peers_all = src_cams
    probes = []
    for k, ci in enumerate(src_cams):
        cam, gt = ds.get_item(ci)
        theta0 = {g: m[g].detach().clone() for g in E.GROUPS}
        g, L0, cap0 = _forward_g_and_capture(m, opt_state, ds, ci, set_params,
                                             capture_state, zero_grads)
        d = {grp: E.adam_update(g[grp], opt_state[grp]) for grp in E.GROUPS}
        set_params(theta0)
        sp = smap.get(ci, {})
        rows = {r["alpha"]: r for r in sp.get("rows", [])}
        alpha_C = sp.get("alpha_C")
        okH = [a for a, r in rows.items() if r["H"] <= tau_H and r["valid"]]
        alpha_H = max(okH) if okH else None
        peers = [c for c in peers_all if c != ci]

        def render_set(cam_list, alpha):
            if alpha is None:
                return {}
            th = {grp: theta0[grp] + alpha * d[grp] for grp in E.GROUPS}
            set_params(th)
            out = {}
            for cj in cam_list:
                camj, gtj = ds.get_item(cj)
                frame, _, _, _ = L.render_c0(m, camj, E.SH_DEGREE)
                Lj = float(L.train_loss(frame, gtj).item())
                mse = float(((frame - gtj) ** 2).mean().item())
                psnr = float(20 * np.log10(1.0 / np.sqrt(mse))) if mse > 1e-10 else 100.0
                out[cj] = {"loss": Lj, "psnr": psnr}
            return out

        evC_peer = render_set(peers, alpha_C)
        evH_peer = render_set(peers, alpha_H)
        evC_hold = render_set(hold_cams, alpha_C)
        evH_hold = render_set(hold_cams, alpha_H)
        ev1_hold = render_set(hold_cams, 1.0)
        set_params(theta0)
        # G_C from sweep source losses (free): real(a)=L0-L_src(a)
        G_C = None
        if alpha_C is not None and alpha_C in rows and 1.0 in rows:
            real_C = L0 - rows[alpha_C]["L_src"]
            real_1 = L0 - rows[1.0]["L_src"]
            if real_1 > 0:
                G_C = real_C / real_1
        # holdout PSNR delta (alpha_C - alpha1), median over 8 holdout cams
        hdeltas = []
        if alpha_C is not None and alpha_C in evC_hold and 1.0 in ev1_hold:
            for cj in hold_cams:
                hdeltas.append(evC_hold[cj]["psnr"] - ev1_hold[cj]["psnr"])
        hold_psnr_delta = float(np.median(hdeltas)) if hdeltas else None
        probes.append({"probe": k, "cam": int(ci), "alpha_C": alpha_C,
                       "alpha_H": alpha_H, "G_C": G_C,
                       "holdout_psnr_delta": hold_psnr_delta,
                       "holdout_psnr_delta_per_cam": hdeltas,
                       "n_peers": len(peers), "n_holdout": len(hold_cams)})
        LOG(f"cam {k} alpha_C={alpha_C} alpha_H={alpha_H} G_C={G_C} "
            f"hold_dPSNR={hold_psnr_delta}")
        torch.cuda.empty_cache()
    uninstall()
    jdump({"cell": cell_id, "tau_H": tau_H, "probes": probes},
          os.path.join(outdir, f"ET0_SAFESTEP_{cell_id}.json"))
    LOG("safestep done")


def aggregate(args):
    cells = [c for c in args.cells.split(",") if c]
    out = {}
    for cell in cells:
        sw = json.load(open(os.path.join(args.outdir, f"ET0_SWEEP_{cell}.json")))
        ss = json.load(open(os.path.join(args.outdir, f"ET0_SAFESTEP_{cell}.json")))
        smap = {p["cam"]: p for p in ss["probes"]}
        rho_ECs, drhos, AUC_Cs, dAUCs, G_Cs, hold_d, invalids = [], [], [], [], [], [], 0
        n_probes = len(sw["probes"])
        for p in sw["probes"]:
            sp = smap.get(p["cam"], {})
            G_C = sp.get("G_C"); hold = sp.get("holdout_psnr_delta")
            rho_ok = not np.isnan(p["rho_EC"])
            auc_ok = not np.isnan(p["AUC_C"])
            g_ok = G_C is not None
            if rho_ok:
                rho_ECs.append(p["rho_EC"])
            if not np.isnan(p["delta_rho"]):
                drhos.append(p["delta_rho"])
            if auc_ok:
                AUC_Cs.append(p["AUC_C"])
            if not np.isnan(p["delta_AUC"]):
                dAUCs.append(p["delta_AUC"])
            if g_ok:
                G_Cs.append(G_C)
            if hold is not None:
                hold_d.append(hold)
            if (not rho_ok) or np.isnan(p["delta_rho"]) or (not g_ok):
                invalids += 1
        out[cell.replace("_", "@")] = {
            "valid_rho_count": len(rho_ECs),
            "rho_EC_median": median(rho_ECs),
            "delta_rho_median": median(drhos),
            "valid_auc_count": len(AUC_Cs),
            "AUC_C_median": median(AUC_Cs),
            "delta_AUC_median": median(dAUCs),
            "G_C_median": median(G_Cs),
            "G_C_p10": p10(G_Cs),
            "truehold_psnr_delta_median_db": median(hold_d),
            "invalid_fraction": float(invalids) / n_probes if n_probes else 1.0,
        }
    summary = {"cells": out, "_agg": {"tau_C": E.TAU_C, "cells": cells}}
    jdump(summary, os.path.join(args.outdir, "ET0_SUMMARY.json"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()







