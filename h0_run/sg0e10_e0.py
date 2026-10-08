#!/usr/bin/env python3
"""SG0-E10 E0 baseline-viability driver (per cell).

Runs ONLY the frozen native-Adam K=4/K=8 rollouts on the canonical 10K
checkpoint and reports base_non_descent_K8_count. NO hierarchy, NO oracle.
This is the cheap viability gate of SG0-E10_PREREG section C1: if native K=8
single-camera Adam does not reduce held-out loss (BASE_NON_DESCT > 2/8) on ANY
cell, SplatMG is CLOSED immediately (SPLATMG_CLOSED) and the oracle is NOT run.

Reuses sg0_lib primitives verbatim (a0_lib frozen renderer/loss/cameras/model
loader + native_adam_rollout). No renderer change. Decision-independent.
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

    ident_env = G.L.bootstrap_c0()
    ds = G.L.load_dataset(args.scene, device="cuda")
    raw0, opt_state, ident = G.L.load_model(args.scene, args.ckpt)
    opt_raw = G.load_raw_opt_state(args.scene, args.ckpt)
    assert ident["active_sh_degree"] == SH_DEGREE, ident["active_sh_degree"]
    n_cams = len(ds)
    cam_ids = G.L.frozen_camera_ids(n_cams)
    cell = f"{args.scene}@{args.ckpt}"
    log(f"scene={args.scene} ckpt={args.ckpt} N={ident['num_points']} "
        f"n_cams={n_cams} frozen_cams={cam_ids} ckpt_sha={ident['ckpt_sha256'][:16]}")

    cam_items = [ds.get_item(ci) for ci in cam_ids]
    folds = {
        "A": {"design": [0, 1, 2, 3], "holdout": [4, 5, 6, 7]},
        "B": {"design": [4, 5, 6, 7], "holdout": [0, 1, 2, 3]},
    }
    rollout_design_orders = []
    for fk, fdef in folds.items():
        base = fdef["design"]
        for off in cyclic_offsets(base):
            rollout_design_orders.append((fk, off))
    assert len(rollout_design_orders) == 8

    ev0 = G.evaluate(raw0, cam_items, ds)
    L0 = [p["loss"] for p in ev0["per_camera"]]
    log(f"theta_0 loss/8cam={['%.5f'%x for x in L0]}")

    n_rollout_records = []
    bnd8 = []
    bnd4 = []
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
            rec = {"rollout": ridx, "fold": fk, "K": K,
                   "design_positions": design_pos, "holdout_positions": hold_pos,
                   "L0_design": L0_des, "L0_holdout": L0_hold,
                   "LK_design": LK_des, "LK_holdout": LK_hold,
                   "base_non_descent": bool(base_non_descent),
                   "train_loss_trace": trace}
            n_rollout_records.append(rec)
            if K == 8:
                bnd8.append(bool(base_non_descent))
            else:
                bnd4.append(bool(base_non_descent))
            log(f"rollout {ridx} fold {fk} K={K}: L0_hold={L0_hold:.6f} "
                f"LK_hold={LK_hold:.6f} non_descent={base_non_descent}")
            del rawK
            torch.cuda.empty_cache()

    out_cell = {
        "cell": cell, "scene": args.scene, "ckpt": args.ckpt,
        "num_points": ident["num_points"], "n_cameras_total": n_cams,
        "frozen_camera_ids": cam_ids,
        "ckpt_sha256": ident["ckpt_sha256"],
        "stage": "E0",
        "base_non_descent_K4_count": int(sum(bnd4)),
        "base_non_descent_K8_count": int(sum(bnd8)),
        "rollouts": n_rollout_records,
    }
    jdump(out_cell, os.path.join(args.outdir, f"SG0E10_E0_{cell}.json"))
    log(f"wrote SG0E10_E0_{cell}.json  base_non_descent_K8={out_cell['base_non_descent_K8_count']} "
        f"base_non_descent_K4={out_cell['base_non_descent_K4_count']}")


if __name__ == "__main__":
    main()