"""P14 Section 5: ORACLE_ONLY active-set selector (per cell).

At the 16K initial state, one EXACT dense backward per cluster camera.
Per-Gaussian gradient energy, normalized per (camera, group) by the
median of nonzero per-Gaussian norms, squared, summed over cameras and
groups -> S_i. Active set = top-S_i rows (tie-break Gaussian ID).
Saves active sets for the prereg q grid + E(q) curve. Selector compute
is NOT counted toward any speedup (oracle-only).
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p14_common as C

Q_CURVE = [round(0.025 * k, 4) for k in range(1, 21)]  # 0.025..0.5
Q_CURVE += [0.6, 0.7, 0.8, 0.9, 1.0]


def k_of_q(q, N):
    return int(np.floor(q * N + 0.5))  # round half up, deterministic


def selector_for_cell(ctx, cam_ids):
    """Returns S_i [N] float64, per-camera records, and E(q) data."""
    N = ctx.N
    S = np.zeros(N, dtype=np.float64)
    cam_recs = []
    for ci in cam_ids:
        cam, gt = ctx.cam_gt(ci)
        ctx.model.optimizer.zero_grad(set_to_none=True)
        image = ctx.render_train(cam)
        loss = ctx.fused_loss(image, gt)
        loss.backward()
        g = {}
        g["means"] = ctx.model._xyz.grad.detach()          # [N,3]
        g["scale"] = ctx.model._scaling.grad.detach()      # [N,3]
        g["quat"] = ctx.model._rotation.grad.detach()      # [N,4]
        g["opacity"] = ctx.model._opacity.grad.detach()    # [N,1]
        g["sh0"] = ctx.model._shs.grad[:, 0, :].detach()   # [N,3]
        g["shn"] = ctx.model._shs.grad[:, 1:, :].detach()  # [N,15,3]
        rec = {"camera": int(ci), "loss": float(loss.detach()),
               "groups": {}}
        for name, t in g.items():
            nrm = t.reshape(t.shape[0], -1).float().norm(dim=-1) \
                .cpu().numpy().astype(np.float64)
            nz = nrm[nrm > 0]
            med = float(np.median(nz)) if nz.size else None
            if med is None or med == 0.0:
                rec["groups"][name] = {"nonzero": int(nz.size),
                                       "median": None, "skipped": True}
                continue
            contrib = (nrm / med) ** 2
            S += contrib
            rec["groups"][name] = {
                "nonzero": int(nz.size), "median": med,
                "energy_share": float(
                    contrib.sum() / max(S.sum(), 1e-30))}
        torch.cuda.synchronize()
        ctx.model.optimizer.zero_grad(set_to_none=True)
        cam_recs.append(rec)
        print(f"    cam {ci}: loss={rec['loss']:.5f}", flush=True)
    total = float(S.sum())
    order = np.argsort(-S, kind="stable")  # ties keep id order
    cum = np.cumsum(S[order]) / total
    e_curve = {}
    for q in Q_CURVE + list(C.Q_LIST_STAGE_A):
        k = k_of_q(q, N)
        e_curve[str(q)] = float(cum[k - 1]) if k >= 1 else 0.0
    return S, order, cam_recs, e_curve, total


def main():
    import json
    gpu = "4"
    clusters = json.load(open(os.path.join(
        C.P14_ROOT, "artifacts", "P14_CAMERA_CLUSTERS.json")))
    out = {"phase": "P14", "created_utc": time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "scenes": {}}
    for scene in C.SCENES:
        t0 = time.time()
        ctx = C.SceneCtx(scene, gpu=gpu)
        ctx.restore_initial()
        sc = {"scene": scene, "N": int(ctx.N),
              "cells": [], "active_rule":
              ("top-K by S_i desc, K = floor(q*N + 0.5); tie-break "
               "Gaussian ID ascending (stable sort)"),
              "normalization": ("per (camera, group): ||g_i^group|| / "
                                "median_nonzero(||g_i^group||), squared, "
                                "summed over the 8 cluster cameras and "
                                "6 groups (means/scale/quat/opacity/"
                                "SH0/SHN)")}
        npz_path = os.path.join(C.P14_ROOT, "raw",
                                f"P14_ACTIVE_{scene}.npz")
        npz = {"N": np.int64(ctx.N)}
        for ci_cell, cell in enumerate(clusters["scenes"][scene]["cells"]):
            print(f"[p14:{scene}] selector cell {ci_cell} "
                  f"(seed {cell['seed_camera']})", flush=True)
            S, order, cam_recs, e_curve, total = selector_for_cell(
                ctx, cell["cluster_cameras"])
            cell_rec = {
                "cell": ci_cell, "seed_camera": cell["seed_camera"],
                "cluster_cameras": cell["cluster_cameras"],
                "per_camera": cam_recs,
                "S_total": total, "S_sha256": C.sha256_bytes(
                    S.tobytes()),
                "E_q": e_curve,
                "active_counts": {},
            }
            for q in C.Q_LIST_STAGE_A:
                k = k_of_q(q, ctx.N)
                act = np.sort(order[:k]).astype(np.int64)
                npz[f"cell{ci_cell}_q{int(round(q*100)):02d}"] = act
                cell_rec["active_counts"][str(q)] = int(k)
                cell_rec.setdefault("active_sha256", {})[str(q)] = \
                    C.sha256_bytes(act.tobytes())
            sc["cells"].append(cell_rec)
            print(f"[p14:{scene}] cell {ci_cell}: E(q) = "
                  + ", ".join(f"{q}:{e_curve[str(q)]:.4f}"
                              for q in C.Q_LIST_STAGE_A), flush=True)
        np.savez_compressed(npz_path, **npz)
        sc["active_npz"] = {"path": npz_path,
                            "sha256": C.sha256_file(npz_path)}
        out["scenes"][scene] = sc
        print(f"[p14:{scene}] selector done in {time.time()-t0:.0f}s",
              flush=True)
        del ctx
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    C.jwrite(os.path.join(C.P14_ROOT, "artifacts", "P14_SELECTOR.json"),
             out)
    print("[p14] P14_SELECTOR.json written (ORACLE_ONLY, not counted "
          "toward speedup)")


if __name__ == "__main__":
    main()
