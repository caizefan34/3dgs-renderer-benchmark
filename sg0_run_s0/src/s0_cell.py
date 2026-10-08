"""SecantGS S0 per-cell driver (scene x checkpoint).

Runs the 4 frozen native-Adam rollouts (offsets 0/2/4/6, K=16). For each rollout,
performs the oracle temporal-subspace projection for m in {2,4} (previous-m executed
Adam deltas span the next executed Adam delta), evaluates each projected state on
the source camera + 7 peer train views + 8 true-holdout views, and writes a compact
per-cell JSON + raw log.

Frozen gates (S0_PREREG sections 2,3,4,5,7,8) are enforced exactly; no threshold
tuning, no per-cell selection (the analyzer owns the verdict).
"""
import argparse
import os
import sys
import json
import time
import numpy as np
import torch

sys.path.insert(0, "/root/epic05_s0")
import s0_lib as G

L = G.L
SH_DEGREE = G.SH_DEGREE
OFFSETS = (0, 2, 4, 6)
K = 16


def _dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=L.json_default)


def pctl(a, q):
    a = [x for x in a if x is not None and np.isfinite(x)]
    if not a:
        return None
    return float(np.percentile(np.array(a, dtype=np.float64), q))


def median(a):
    a = [x for x in a if x is not None and np.isfinite(x)]
    if not a:
        return None
    return float(np.median(np.array(a, dtype=np.float64)))


def measure_timing(raw0, opt_raw, design_items, ds, log):
    """Descriptive phase timing (prereg section 11). CUDA-event based, one rollout's
    worth of steps; NOT a speedup claim."""
    import torch.nn.functional as F
    import fused_ssim
    params = {
        "xyz": torch.nn.Parameter(raw0["xyz"].clone().detach().float().requires_grad_(True)),
        "shs": torch.nn.Parameter(raw0["shs"].clone().detach().float().requires_grad_(True)),
        "scaling": torch.nn.Parameter(raw0["scaling"].clone().detach().float().requires_grad_(True)),
        "rotation": torch.nn.Parameter(raw0["rotation"].clone().detach().float().requires_grad_(True)),
        "opacity": torch.nn.Parameter(raw0["opacity"].clone().detach().float().requires_grad_(True)),
    }
    opt = G.build_optimizer(params, opt_raw["param_groups"])
    n = 10
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    torch.cuda.synchronize()
    # full native step time
    t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
    cam, gt = design_items[0][1]
    torch.cuda.synchronize(); t0.record()
    for _ in range(n):
        opt.zero_grad(set_to_none=True)
        frame, _, _, _ = L.render_c0(params, cam, SH_DEGREE)
        loss = L.train_loss(frame, gt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(params.values()), 1.0)
        opt.step()
    t1.record(); torch.cuda.synchronize()
    T_full = t0.elapsed_time(t1) / n  # ms per native step (incl forward)
    # forward-only validation-like pass (render + loss, no backward)
    torch.cuda.synchronize(); t0.record()
    for _ in range(n):
        with torch.no_grad():
            frame, _, _, _ = L.render_c0(params, cam, SH_DEGREE)
            b = frame.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
            g = gt.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
            _ = F.l1_loss(b, g)
    t1.record(); torch.cuda.synchronize()
    T_fwd = t0.elapsed_time(t1) / n
    out = {"n_steps": n, "T_full_step_ms": T_full, "T_forward_val_like_ms": T_fwd,
           "note": "descriptive Amdahl ceiling inputs; T(q)=T_full+(q-1)*T_fwd; NOT a speedup"}
    for q in (2, 4, 8):
        out[f"T_q{q}_ms"] = T_full + (q - 1) * T_fwd
    log(f"timing: T_full={T_full:.2f}ms T_fwd={T_fwd:.2f}ms")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--ckpt", type=int, required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--log", default="")
    ap.add_argument("--offsets", default="0,2,4,6")
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
    offsets = tuple(int(x) for x in args.offsets.split(","))

    # ---------------- bootstrap frozen C0 renderer / data ----------------
    L.bootstrap_c0()
    ds = L.load_dataset(args.scene, device="cuda")
    raw0, opt_state, ident = L.load_model(args.scene, args.ckpt)
    opt_raw = G.load_raw_opt_state(args.scene, args.ckpt)  # raw optimizer_state_dict
    assert ident["active_sh_degree"] == SH_DEGREE, ident["active_sh_degree"]
    n_cams = len(ds)
    train_pool, holdout_pool, train_panel, holdout_panel = G.corrected_panels(n_cams)
    cell = f"{args.scene}@{args.ckpt}"
    log(f"scene={args.scene} ckpt={args.ckpt} N={ident['num_points']} n_cams={n_cams}")
    log(f"train_panel={train_panel}")
    log(f"holdout_panel={holdout_panel}")

    train_cams = [ds.get_item(ci) for ci in train_panel]
    holdout_cams = [ds.get_item(ci) for ci in holdout_panel]
    group_lrs = G.frozen_group_lrs(opt_raw)
    log(f"group_lrs={group_lrs}")

    ev0h = G.evaluate(raw0, holdout_cams, ds)
    ev0t = G.evaluate(raw0, train_cams, ds)
    L0_holdout = ev0h["loss_mean"]; L0_train = ev0t["loss_mean"]
    log(f"L0 holdout={L0_holdout:.5f} train={L0_train:.5f}")

    # timing (descriptive)
    c0, g0 = train_cams[0]
    design_items_t = [(train_panel[0], (c0, g0))]
    timing = measure_timing(raw0, opt_raw, design_items_t, ds, log)

    # accumulators per memory m
    acc = {m: {"E_span": [], "P_src": [], "peer_win": [], "thold_psnr_delta": [],
               "thold_loss_delta": [], "valid_sd": 0, "invalid": 0, "total": 0}
           for m in G.MEMORIES}
    rollout_records = []
    proj_records = []
    gram_records = []
    loss_probe_records = []
    holdout_records = []

    for ridx, off in enumerate(offsets):
        design_idx = [train_panel[(off + j) % len(train_panel)] for j in range(K)]
        design_items = []
        for ci in design_idx:
            cam, gt = ds.get_item(ci)
            design_items.append((ci, (cam, gt)))
        states, src_before, src_after, src_cam, lrs = \
            G.native_adam_rollout_capture(raw0, opt_raw, design_items, ds, K=K)
        log(f"rollout {ridx} off={off} captured: src_after={[round(x,4) for x in src_after]}")

        # native theta_t evals (holdout + train + peers) cached per step t
        nat = {}
        for t in range(1, K + 1):
            theta_t = states[t]
            evh = G.evaluate(theta_t, holdout_cams, ds)
            evt = G.evaluate(theta_t, train_cams, ds)
            src_t = src_cam[t - 1]
            peer_pos = [p for p in range(len(train_panel)) if train_panel[p] != src_t]
            evp = G.evaluate(theta_t, [train_cams[p] for p in peer_pos], ds)
            nat[t] = {"Lnat_holdout": evh["loss_mean"], "Pnat_holdout": evh["psnr_mean"],
                      "Lnat_train": evt["loss_mean"], "Lnat_peer": evp["loss_mean"],
                      "peer_pos": peer_pos}

        # oracle temporal projections
        for m in G.MEMORIES:
            mi = int(m)
            for t in range(1, K + 1):
                if t < mi + 1:
                    continue
                # delta_k = states[k] - states[k-1]; basis = delta_{t-mi}..delta_{t-1}; target = delta_t
                d_of = lambda k: {g: states[k][g] - states[k - 1][g] for g in G.GROUPS}
                basis = [d_of(t - mi + j) for j in range(mi)]
                target = d_of(t)
                lrs_t = lrs[t - 1]
                acc[m]["total"] += 1
                D_native_src = src_before[t - 1] - src_after[t - 1]
                orec = {"rollout": ridx, "offset": off, "m": m, "t": t,
                        "src_cam": src_cam[t - 1], "D_native_src": D_native_src}
                proj = G.oracle_projection(basis, target, lrs_t)
                gram_records.append({"rollout": ridx, "m": m, "t": t,
                                     "trace_G": proj.get("trace_G"),
                                     "lambda": proj.get("lambda"),
                                     "target_norm_W2": proj.get("target_norm_W2"),
                                     "proj_norm_W2": proj.get("proj_norm_W2")})
                if not proj["valid"]:
                    acc[m]["invalid"] += 1
                    orec["valid"] = False; orec["reason"] = proj.get("reason")
                    proj_records.append(orec)
                    continue
                theta_prev = states[t - 1]
                theta_proj = G.apply_delta_raw(theta_prev, proj["delta_proj"])
                src_t = src_cam[t - 1]
                sp = train_panel.index(src_t)
                peer_pos = [p for p in range(len(train_panel)) if p != sp]
                cam_s, gt_s = ds.get_item(src_t)
                ev_src = G.evaluate(theta_proj, [(cam_s, gt_s)], ds)
                L_proj_src = ev_src["loss_mean"]
                ev_pp = G.evaluate(theta_proj, [train_cams[p] for p in peer_pos], ds)
                L_proj_peer = ev_pp["loss_mean"]
                ev_hp = G.evaluate(theta_proj, holdout_cams, ds)
                L_proj_hold = ev_hp["loss_mean"]; P_proj_hold = ev_hp["psnr_mean"]
                finite = all(np.isfinite(x) for x in
                             [L_proj_src, L_proj_peer, L_proj_hold, P_proj_hold,
                              proj["E_span"], proj["residual_energy"]])
                D_proj_src = src_before[t - 1] - L_proj_src
                P_src = (D_proj_src / D_native_src) if D_native_src > 0 else None
                peer_win = bool(L_proj_peer <= nat[t]["Lnat_peer"])
                thold_psnr_delta = P_proj_hold - nat[t]["Pnat_holdout"]
                thold_loss_delta = L_proj_hold - nat[t]["Lnat_holdout"]
                orec.update({"valid": bool(finite), "E_span": proj["E_span"],
                             "residual_energy": proj["residual_energy"],
                             "P_src": P_src, "peer_win": peer_win,
                             "thold_psnr_delta": thold_psnr_delta,
                             "thold_loss_delta": thold_loss_delta,
                             "L_proj_src": L_proj_src, "L_proj_peer": L_proj_peer,
                             "L_proj_holdout": L_proj_hold,
                             "P_proj_holdout": P_proj_hold,
                             "coefficients": proj["coefficients"]})
                proj_records.append(orec)
                loss_probe_records.append({"rollout": ridx, "m": m, "t": t,
                                           "src_cam": src_t, "L_src_before": src_before[t - 1],
                                           "L_src_after": src_after[t - 1],
                                           "D_native_src": D_native_src,
                                           "L_proj_src": L_proj_src, "D_proj_src": D_proj_src,
                                           "P_src": P_src})
                holdout_records.append({"rollout": ridx, "m": m, "t": t,
                                        "Lnat_holdout": nat[t]["Lnat_holdout"],
                                        "L_proj_holdout": L_proj_hold,
                                        "thold_loss_delta": thold_loss_delta,
                                        "Pnat_holdout": nat[t]["Pnat_holdout"],
                                        "P_proj_holdout": P_proj_hold,
                                        "thold_psnr_delta": thold_psnr_delta})
                if finite:
                    acc[m]["E_span"].append(proj["E_span"])
                    if P_src is not None:
                        acc[m]["P_src"].append(P_src)
                    acc[m]["peer_win"].append(1.0 if peer_win else 0.0)
                    acc[m]["thold_psnr_delta"].append(thold_psnr_delta)
                    acc[m]["thold_loss_delta"].append(thold_loss_delta)
                    if D_native_src > 0:
                        acc[m]["valid_sd"] += 1
                else:
                    acc[m]["invalid"] += 1
                del theta_proj
        del states, nat
        torch.cuda.empty_cache()
        rollout_records.append({"rollout": ridx, "offset": off,
                                "design_idx": design_idx,
                                "src_cam": src_cam,
                                "src_before": src_before, "src_after": src_after})

    # ---------------- aggregate per memory ----------------
    out_cell = {"cell": cell, "scene": args.scene, "ckpt": args.ckpt,
                "num_points": ident["num_points"], "n_cameras_total": n_cams,
                "frozen_camera_ids": train_panel, "true_holdout_ids": holdout_panel,
                "group_lrs": group_lrs, "identity": ident,
                "L0_holdout": L0_holdout, "L0_train": L0_train,
                "timing": timing, "rollouts": rollout_records}
    for m in G.MEMORIES:
        a = acc[m]
        n_tot = a["total"]
        out_cell[m] = {
            "total_probes": n_tot,
            "valid_source_descent_count": a["valid_sd"],
            "invalid_count": a["invalid"],
            "invalid_fraction": (a["invalid"] / n_tot) if n_tot else None,
            "E_span_median": median(a["E_span"]),
            "E_span_p10": pctl(a["E_span"], 10),
            "P_src_median": median(a["P_src"]),
            "P_src_p10": pctl(a["P_src"], 10),
            "peer_loss_win_rate": (float(np.mean(a["peer_win"])) if a["peer_win"] else None),
            "thold_psnr_delta_median_db": median(a["thold_psnr_delta"]),
            "thold_psnr_delta_p10_db": pctl(a["thold_psnr_delta"], 10),
            "thold_loss_delta_median": median(a["thold_loss_delta"]),
        }
    _dump(out_cell, os.path.join(args.outdir, f"S0_CELL_{cell}.json"))
    _dump(proj_records, os.path.join(args.outdir, f"S0_ORACLE_PROJECTIONS_{cell}.json"))
    _dump(gram_records, os.path.join(args.outdir, f"S0_UPDATE_GRAMS_{cell}.json"))
    _dump(loss_probe_records, os.path.join(args.outdir, f"S0_LOSS_PROBES_{cell}.json"))
    _dump(holdout_records, os.path.join(args.outdir, f"S0_TRUE_HOLDOUT_{cell}.json"))
    _dump({"cell": cell, "timing": timing}, os.path.join(args.outdir, f"S0_TIMING_CEILING_{cell}.json"))
    log(f"wrote S0_CELL_{cell}.json  m4 valid_sd={out_cell['4']['valid_source_descent_count']} "
        f"m2 valid_sd={out_cell['2']['valid_source_descent_count']}")


if __name__ == "__main__":
    main()
