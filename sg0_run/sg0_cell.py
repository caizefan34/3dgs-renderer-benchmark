"""SplatMG SG0 per-cell driver (scene x checkpoint).

Runs the frozen native-Adam rollouts (K=4, K=8), builds the deterministic
kd-tree hierarchy, performs the future-displacement oracle projection for
{ L32, L128 } x { SIM13, AFF22 }, evaluates each oracle jump on the original
full-resolution objective, and writes a compact per-cell JSON + raw log.

Frozen gates (SG0_PREREG sections 3, 8, 9) are enforced exactly; no threshold
tuning, no per-cell selection (that belongs to the analyzer).
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, "/root/epic05_sg0")
import sg0_lib as G

SH_DEGREE = G.SH_DEGREE


def jdump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=G.L.json_default)


def cyclic_offsets(cams4):
    a, b, c, d = cams4
    return [[a, b, c, d], [b, c, d, a], [c, d, a, b], [d, a, b, c]]


def identity_control(raw0, disp, log):
    """Identity/per-Gaussian basis control: projecting onto the identity basis
    must reconstruct the represented future displacement to tolerance, i.e. the
    coordinate round-trip raw0 -> disp -> reconstruct -> disp is exact."""
    oracle = G.reconstruct_oracle(raw0, disp)
    back = G.future_displacement(raw0, oracle)
    errs = {}
    for f in G.FAMILIES:
        errs[f] = float((back[f] - disp[f]).abs().max())
    max_err = max(errs.values())
    ok = max_err < 1e-5
    log(f"[identity] max round-trip err={max_err:.3e} per-family={errs} pass={ok}")
    if not ok:
        raise RuntimeError(f"SG0_PROJECTION_BUG identity control failed: {errs}")
    return {"max_abs_err": max_err, "per_family": errs, "pass": bool(ok)}


def project_displacement(disp, raw0, clusters, basis):
    """Project full displacement dict onto the coarse basis for a partition."""
    proj = {f: torch.zeros_like(disp[f]) for f in G.FAMILIES}
    xyz0 = raw0["xyz"]
    for members in clusters:
        members = members if isinstance(members, torch.Tensor) else torch.as_tensor(members, device=xyz0.device, dtype=torch.long)
        means0_c = xyz0[members]
        for f in G.FAMILIES:
            d_c = disp[f][members]
            proj[f][members] = G.cluster_projection(d_c, basis, means0_c, f)
    return proj


def agg_metrics(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return {"p_hold_median": None, "p_hold_p10": None, "n_valid": 0}
    a = np.array(vals, dtype=np.float64)
    return {"p_hold_median": float(np.median(a)),
            "p_hold_p10": float(np.percentile(a, 10)),
            "n_valid": int(len(a))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--ckpt", type=int, required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--log", default="")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    logf = open(args.log, "a") if args.log else None
    t0 = time.time()
    def log(*a):
        msg = f"[{time.time()-t0:8.1f}s] " + " ".join(str(x) for x in a)
        print(msg, flush=True)
        if logf:
            logf.write(msg + "\n"); logf.flush()

    torch.manual_seed(0)
    np.random.seed(0)

    # ---------------- bootstrap frozen C0 renderer / data ----------------
    ident_env = G.L.bootstrap_c0()
    ds = G.L.load_dataset(args.scene, device="cuda")
    raw0, opt_state, ident = G.L.load_model(args.scene, args.ckpt)
    opt_raw = G.load_raw_opt_state(args.scene, args.ckpt)
    assert ident["active_sh_degree"] == SH_DEGREE, ident["active_sh_degree"]
    n_cams = len(ds)
    cam_ids = G.L.frozen_camera_ids(n_cams)  # c0..c7
    cell = f"{args.scene}@{args.ckpt}"
    log(f"scene={args.scene} ckpt={args.ckpt} N={ident['num_points']} "
        f"n_cams={n_cams} frozen_cams={cam_ids}")

    # 8 frozen camera items
    cam_items = [ds.get_item(ci) for ci in cam_ids]
    # fold definitions (positions into cam_ids)
    folds = {
        "A": {"design": [0, 1, 2, 3], "holdout": [4, 5, 6, 7]},
        "B": {"design": [4, 5, 6, 7], "holdout": [0, 1, 2, 3]},
    }
    # build the 8 rollout design-camera orders
    rollout_design_orders = []  # list of (fold, [positions])
    for fk, fdef in folds.items():
        base = fdef["design"]
        for off in cyclic_offsets(base):
            rollout_design_orders.append((fk, off))
    assert len(rollout_design_orders) == 8

    # ---------------- theta_0 evaluation on all 8 cameras ----------------
    ev0 = G.evaluate(raw0, cam_items, ds)
    L0 = [p["loss"] for p in ev0["per_camera"]]
    log(f"theta_0 loss/8cam={['%.5f'%x for x in L0]}")

    # ---------------- hierarchy (from theta_0 means only) ----------------
    means0 = raw0["xyz"].detach().cpu().numpy().astype(np.float64)
    t_h = time.time()
    clusters_by_level = G.build_hierarchy(means0, levels=(0, 2))
    hsummary = G.hierarchy_summary(clusters_by_level)
    for lvl, s in hsummary.items():
        log(f"hierarchy {lvl}: {s}")
    log(f"hierarchy built in {time.time()-t_h:.1f}s")

    # ---------------- per-rollout, per-K, per-config ----------------
    # results[config][K] = dict(p_hold list, p_design list, oracle_improve list, base_non_descent list)
    results = {G.cfg_name(l, b): {"K4": {"p_hold": [], "p_design": [],
               "oracle_improve": [], "base_non_descent": []},
               "K8": {"p_hold": [], "p_design": [],
               "oracle_improve": [], "base_non_descent": []}}
              for (l, b) in G.CONFIGS}
    random_ctrl = {"K4": {"p_hold": []}, "K8": {"p_hold": []}}
    identity_result = None
    n_rollout_records = []

    for ridx, (fk, order) in enumerate(rollout_design_orders):
        design_pos = order
        hold_pos = folds[fk]["holdout"]
        design_items = [cam_items[p] for p in design_pos]
        L0_des = float(np.mean([L0[p] for p in design_pos]))
        L0_hold = float(np.mean([L0[p] for p in hold_pos]))
        for K in (4, 8):
            rawK, trace = G.native_adam_rollout(raw0, opt_raw, design_items, ds, K)
            evK = G.evaluate(rawK, cam_items, ds)
            LK = [p["loss"] for p in evK["per_camera"]]
            LK_des = float(np.mean([LK[p] for p in design_pos]))
            LK_hold = float(np.mean([LK[p] for p in hold_pos]))
            base_non_descent = not (LK_hold < L0_hold)
            disp = G.future_displacement(raw0, rawK)

            if identity_result is None:
                identity_result = identity_control(raw0, disp, log)

            rec = {"rollout": ridx, "fold": fk, "K": K,
                   "design_positions": design_pos, "holdout_positions": hold_pos,
                   "L0_design": L0_des, "L0_holdout": L0_hold,
                   "LK_design": LK_des, "LK_holdout": LK_hold,
                   "base_non_descent": bool(base_non_descent),
                   "train_loss_trace": trace}
            n_rollout_records.append(rec)

            for (lvl, basis) in G.CONFIGS:
                cfg = G.cfg_name(lvl, basis)
                clusters = clusters_by_level[lvl]
                proj = project_displacement(disp, raw0, clusters, basis)
                oracle = G.reconstruct_oracle(raw0, proj)
                evO = G.evaluate(oracle, cam_items, ds)
                LO = [p["loss"] for p in evO["per_camera"]]
                LO_des = float(np.mean([LO[p] for p in design_pos]))
                LO_hold = float(np.mean([LO[p] for p in hold_pos]))
                den_h = L0_hold - LK_hold
                den_d = L0_des - LK_des
                p_hold = (L0_hold - LO_hold) / den_h if den_h > 0 else None
                p_design = (L0_des - LO_des) / den_d if den_d > 0 else None
                oracle_improve = bool(LO_hold < L0_hold)
                energy = G.energy_capture(disp, proj)
                R = results[cfg][f"K{K}"]
                R["p_hold"].append(p_hold)
                R["p_design"].append(p_design)
                R["oracle_improve"].append(oracle_improve)
                R["base_non_descent"].append(base_non_descent)
                rec.setdefault("configs", {})[cfg] = {
                    "p_hold": p_hold, "p_design": p_design,
                    "oracle_improve": oracle_improve,
                    "LO_design": LO_des, "LO_holdout": LO_hold,
                    "energy_total": energy["total"],
                    "energy_per_family": {f: energy[f] for f in G.FAMILIES},
                }

            # random-cluster diagnostic (SIM13 only, descriptive)
            for K2 in (K,):
                rc = G.random_partition_like(clusters_by_level["L32"], seed=20260930)
                proj_r = project_displacement(disp, raw0, rc, "SIM13")
                oracle_r = G.reconstruct_oracle(raw0, proj_r)
                evOr = G.evaluate(oracle_r, cam_items, ds)
                LOr = [p["loss"] for p in evOr["per_camera"]]
                LO_hold = float(np.mean([LOr[p] for p in hold_pos]))
                den_h = L0_hold - LK_hold
                p_hold = (L0_hold - LO_hold) / den_h if den_h > 0 else None
                random_ctrl[f"K{K}"]["p_hold"].append(p_hold)

            del rawK, disp
            torch.cuda.empty_cache()
            log(f"rollout {ridx} fold {fk} K={K} done: LK_hold={LK_hold:.5f} "
                f"L0_hold={L0_hold:.5f} non_descent={base_non_descent}")

    # ---------------- aggregate ----------------
    N = int(ident["num_points"])
    fine_dof = N * G.REP_DOF_PER_G
    out_cell = {
        "cell": cell, "scene": args.scene, "ckpt": args.ckpt,
        "num_points": N, "n_cameras_total": n_cams,
        "frozen_camera_ids": cam_ids,
        "identity_control": identity_result,
        "hierarchy": hsummary,
        "fine_dof": fine_dof,
        "rollouts": n_rollout_records,
    }
    for (lvl, basis) in G.CONFIGS:
        cfg = G.cfg_name(lvl, basis)
        ncl = len(clusters_by_level[lvl])
        cdof = G.coarse_dof(lvl, basis, ncl)
        ratio = cdof / fine_dof
        cell_cfg = {"coarse_dof": cdof, "coarse_dof_ratio": ratio,
                    "n_clusters": ncl, "K4": {}, "K8": {}}
        for K in (4, 8):
            R = results[cfg][f"K{K}"]
            am = agg_metrics(R["p_hold"])
            pdes = [v for v in R["p_design"] if v is not None and np.isfinite(v)]
            cell_cfg[f"K{K}"] = {
                "p_hold_median": am["p_hold_median"],
                "p_hold_p10": am["p_hold_p10"],
                "p_design_median": float(np.median(pdes)) if pdes else None,
                "oracle_improve_count": int(sum(1 for x in R["oracle_improve"] if x)),
                "base_non_descent_count": int(sum(1 for x in R["base_non_descent"] if x)),
                "n_rollouts": 8,
                "p_hold_valid_n": am["n_valid"],
            }
        out_cell.setdefault("configs", {})[cfg] = cell_cfg

    # random-cluster diagnostic aggregate
    out_cell["random_cluster_control"] = {
        "K4": agg_metrics(random_ctrl["K4"]["p_hold"]),
        "K8": agg_metrics(random_ctrl["K8"]["p_hold"]),
        "seed": 20260930,
        "note": "descriptive only; does not enter the verdict",
    }

    # baseline stability stop check (K=8, any config's base_non_descent same across configs)
    bnd8 = results["L32/SIM13"]["K8"]["base_non_descent"]
    out_cell["base_non_descent_K8_count"] = int(sum(1 for x in bnd8 if x))

    jdump(out_cell, os.path.join(args.outdir, f"SG0_CELL_{cell}.json"))
    log(f"wrote SG0_CELL_{cell}.json  base_non_descent_K8={out_cell['base_non_descent_K8_count']}")


if __name__ == "__main__":
    main()
