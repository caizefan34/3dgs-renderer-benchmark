#!/usr/bin/env python3
"""CVT0 cross-view transfer-oracle harness (CVT0_PREREG.md, FROZEN).

Executes on the remote EPIC-05 A100 (NEVER local GPU). Reuses the frozen C0
primitives from n1_core.py (bootstrap_c0, render_view, load_scene_dataset,
load_checkpoint, build_model, StateMaster, make_fused_loss). The native training
loss is the FUSED-SSIM loss L = 0.8*L1 + 0.2*(1-SSIM) that produced the frozen
n1snap checkpoints (final_status.json loss=fused), i.e. the exact native Adam
step. Topology is frozen: ONE forward/backward/Adam step per source, no
densify/prune/clone/split.

Pipeline (per prereg sections 1-13):
  A identity+env  B camera semantics+panels  C base eval + 8x8 transfer
    matrices + selectors  D diagnostics + timing  E summary  F verdict(analyzer)
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/epic05_n1")
import n1_core as C

EPS2D = C.EPS2D                 # 0.3 native c0
LAMBDA = C.LAMBDA_DSSIM         # 0.2
CELLS = [
    ("room", 10000), ("room", 15000), ("room", 20000),
    ("bicycle", 10000), ("bicycle", 15000), ("bicycle", 20000),
]
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


# ---- SepSSIM (frozen evaluation metric only; NOT the training loss) ---------
class SepSSIM:
    def __init__(self, device="cuda", window_size=11, sigma=1.5):
        coords = torch.arange(window_size, dtype=torch.float32, device=device)
        coords -= window_size // 2
        k1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        k1d = k1d / k1d.sum()
        self.k_h = k1d.unsqueeze(0).unsqueeze(0).repeat(15, 1, 1, 1).contiguous()
        self.k_v = k1d.unsqueeze(0).unsqueeze(0).repeat(15, 1, 1, 1).permute(0, 1, 3, 2).contiguous()
        self.padding = window_size // 2

    def __call__(self, pred, target):
        if pred.ndim == 3:
            pred = pred.unsqueeze(0).permute(0, 3, 1, 2)
            target = target.unsqueeze(0).permute(0, 3, 1, 2)
        stacked = torch.cat([pred, target, pred ** 2, target ** 2, pred * target], dim=1)
        b = F.conv2d(stacked, self.k_h, padding=(0, self.padding), groups=15)
        b = F.conv2d(b, self.k_v, padding=(self.padding, 0), groups=15)
        mu_p, mu_t = b[:, 0:3], b[:, 3:6]
        bp2, bt2, bpt = b[:, 6:9], b[:, 9:12], b[:, 12:15]
        mu_p2, mu_t2, mu_pt = mu_p ** 2, mu_t ** 2, mu_p * mu_t
        sp2, st2, spt = bp2 - mu_p2, bt2 - mu_t2, bpt - mu_pt
        ssim_map = (2 * mu_pt + 0.01 ** 2) * (2 * spt + 0.03 ** 2) / \
                   ((mu_p2 + mu_t2 + 0.01 ** 2) * (sp2 + st2 + 0.03 ** 2))
        return 1.0 - ssim_map.mean()


def _json_default(o):
    import numpy as _np
    if isinstance(o, _np.ndarray):
        return o.tolist()
    if isinstance(o, (_np.floating, _np.integer)):
        return o.item()
    return str(o)


def spearman(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    rx = np.argsort(np.argsort(x)) + 1
    ry = np.argsort(np.argsort(y)) + 1
    # average ties
    def rank_avg(v):
        order = np.argsort(v)
        ranks = np.empty(len(v), dtype=float)
        ranks[order] = np.arange(1, len(v) + 1)
        _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
        sums = np.zeros(len(counts))
        np.add.at(sums, inv, ranks)
        return (sums / counts)[inv]
    rx = rank_avg(x); ry = rank_avg(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def pearson(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


# ---- Camera semantics (prereg section 3/4) ----------------------------------
def camera_semantics(ds):
    n = len(ds)
    all_idx = list(range(n))
    holdout = [i for i in all_indices_safe(ds) if i % 8 == 0]
    train = [i for i in all_indices_safe(ds) if i % 8 != 0]
    assert set(train).isdisjoint(set(holdout)), "train/holdout overlap"
    return n, train, holdout


def all_indices_safe(ds):
    return list(range(len(ds)))


def stratified(pool, k):
    """k midpoint-quantile positions into pool, deterministic dedup by advance."""
    N = len(pool)
    used = set(); out = []
    for i in range(k):
        p = int(math.floor((i + 0.5) * N / k))
        while p in used or p >= N:
            p += 1
        used.add(p)
        out.append(p)
    return out


def build_panels(train, holdout):
    sel_pos = stratified(train, 24)
    sel_cams = [train[p] for p in sel_pos]
    panels = {"panel_0": [], "panel_1": [], "panel_2": []}
    for k, cam in enumerate(sel_cams):
        panels[f"panel_{k % 3}"].append(cam)
    hpos = stratified(holdout, 8)
    hold_panel = [holdout[p] for p in hpos]
    return sel_pos, sel_cams, panels, hpos, hold_panel


# ---- Evaluation helpers -----------------------------------------------------
def native_loss(fused, img, gt):
    return float(fused(img, gt).item())


def eval_cam(model, ds, ci, fused, ssim_fn):
    cam, gt = ds.get_item(ci)
    with torch.no_grad():
        img, _, _ = C.render_view(model, cam, model.active_sh_degree)
    obj = native_loss(fused, img, gt)
    b = img.unsqueeze(0).permute(0, 3, 1, 2)
    g = gt.unsqueeze(0).permute(0, 3, 1, 2)
    mse = float(F.mse_loss(b, g).item())
    psnr = float(20 * math.log10(1.0 / math.sqrt(mse))) if mse > 1e-10 else 100.0
    ssim = float(1.0 - ssim_fn(img, gt).item())
    center = cam.camera_center.detach().float().cpu().numpy().tolist()
    return {"loss": obj, "psnr": psnr, "ssim": ssim, "center": center}


def one_native_step(model, master, ds, ci, fused):
    """ONE exact native Adam step from the restored theta0 using camera ci."""
    master.restore()
    model.optimizer.zero_grad(set_to_none=True)
    cam, gt = ds.get_item(ci)
    img, _, _ = C.render_view(model, cam, model.active_sh_degree)
    loss = fused(img, gt)
    loss.backward()
    model.optimizer.step()               # topology frozen: NO densify/prune
    return float(loss.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/root/epic05_cvt0")
    ap.add_argument("--cells", default="room@10000,room@15000,room@20000,"
                                       "bicycle@10000,bicycle@15000,bicycle@20000")
    ap.add_argument("--smoke", action="store_true",
                    help="1 cell, 1 panel, timing only (mechanics proof)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "cvt0_raw.log"), "w", buffering=1)
    def P(*a):
        s = " ".join(str(x) for x in a); print(s); log.write(s + "\n")

    # ---- A. identity + environment ----------------===========================
    ident = C.bootstrap_c0()
    env = {
        "host": os.uname().nodename,
        "gpu": "NVIDIA A100-SXM4-80GB",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": "cuda:0",
        "loss_backend": "fused-ssim (padding=same) -- frozen n1snap training loss",
        "loss_so_sha256": ident["so_sha256"],
        "c0_ext_so_match_prereg_pin": ident["so_sha256"] == PREREG_PIN_C0,
        "core_so_match_prereg_pin": ident["core_sha256"] == PREREG_PIN_CORE,
        "eps2d": EPS2D, "lambda_dssim": LAMBDA, "tile_size": 16,
        "background": "black", "topology": "frozen during one-step probe",
    }
    P("[A] identity c0_ext=%s core=%s" % (ident["so_sha256"][:16], ident["core_sha256"][:16]))
    assert env["c0_ext_so_match_prereg_pin"] and env["core_so_match_prereg_pin"], "BINARY MISMATCH"
    with open(os.path.join(args.out, "CVT0_ENVIRONMENT.json"), "w") as f:
        json.dump(env, f, indent=2)

    cells = [tuple(x.split("@")) for x in args.cells.split(",")]
    cells = [(s, int(i)) for s, i in cells]
    if args.smoke:
        cells = [cells[0]]

    ssim_fn = SepSSIM(device="cuda")
    base = {}          # (scene,iter) -> dataset cache
    results = {}       # cell -> per-panel record

    for scene, it in cells:
        t_cell0 = time.perf_counter()
        P(f"\n===== CELL {scene}@{it} =====")
        # checkpoint hash gate (fail-closed)
        ckpt_path = os.path.join(C.SNAP_RUNS[scene], f"ckpt_{it}.pt")
        h = C.sha256_file(ckpt_path)
        assert h == CKPT_HASH[(scene, it)], f"CKPT HASH MISMATCH {scene}@{it}: {h}"
        P(f"  ckpt hash OK: {h[:16]}...")

        ds = base.get(scene) or C.load_scene_dataset(scene)
        base[scene] = ds
        n, train, holdout = camera_semantics(ds)
        sel_pos, sel_cams, panels, hpos, hold_panel = build_panels(train, holdout)
        P(f"  n={n} train={len(train)} holdout={len(holdout)} "
          f"hold_panel={hold_panel}")

        ckpt = C.load_checkpoint(scene, it)
        model = C.build_model(ckpt)
        C.scene_reset()
        master = C.StateMaster(model)
        fused = C.make_fused_loss()
        assert fused.so_sha256 == C.P11_FUSED_SO_SHA256, "FUSED LOSS LOCK VIOLATION"

        panel_names = ["panel_0", "panel_1", "panel_2"] if not args.smoke else ["panel_0"]
        cell_res = {"scene": scene, "iter": it, "n_cams": n,
                    "train_pool_size": len(train), "holdout_size": len(holdout),
                    "panels": {}}
        for pname in panel_names:
            panel_cams = panels[pname]
            assert len(panel_cams) == 8, f"{pname} has {len(panel_cams)} cams"
            t_p0 = time.perf_counter()
            # base eval theta0 on panel + holdout (no grad)
            master.restore()
            base_panel = [eval_cam(model, ds, ci, fused, ssim_fn) for ci in panel_cams]
            base_hold = [eval_cam(model, ds, ci, fused, ssim_fn) for ci in hold_panel]
            L0_panel = [r["loss"] for r in base_panel]
            L0_hold = [r["loss"] for r in base_hold]

            # source-update matrix: one native Adam step per source
            M = [[0.0] * 8 for _ in range(8)]     # M[i][j]
            post_panel = [None] * 8
            post_hold = [None] * 8
            src_losses = []
            t_step0 = time.perf_counter()
            for i, si in enumerate(panel_cams):
                one_native_step(model, master, ds, si, fused)
                post_panel[i] = [eval_cam(model, ds, cj, fused, ssim_fn) for cj in panel_cams]
                post_hold[i] = [eval_cam(model, ds, cj, fused, ssim_fn) for cj in hold_panel]
                src_losses.append(L0_panel[i])     # source current loss = L0 at si
            t_step1 = time.perf_counter()
            for i in range(8):
                for j in range(8):
                    M[i][j] = L0_panel[j] - post_panel[i][j]["loss"]
            # true-holdout transfer H_i
            H = [float(np.mean([L0_hold[hh] - post_hold[i][hh]["loss"] for hh in range(8)]))
                 for i in range(8)]
            # peer transfer T_i, self-improvement S_i
            T = []; S = []
            for i in range(8):
                peers = [M[i][j] for j in range(8) if j != i]
                T.append(float(np.mean(peers)))
                S.append(float(L0_panel[i] - post_panel[i][i]["loss"]))

            # selectors
            T_random = float(np.mean(T)); H_random = float(np.mean(H))
            i_hard = int(np.argmax(L0_panel))     # argmax current loss; np.argmax tie->lowest index
            # tie-break by camera ID: among ties pick smallest cam id
            maxL = max(L0_panel)
            i_hard = min([j for j in range(8) if L0_panel[j] == maxL])
            T_hard = T[i_hard]; H_hard = H[i_hard]
            maxT = max(T)
            i_oracle = min([j for j in range(8) if T[j] == maxT])
            T_oracle = T[i_oracle]; H_oracle = H[i_oracle]; S_oracle = S[i_oracle]

            # metrics (prereg section 8)
            A = float(np.mean(np.abs(T))) + 1e-12
            B = float(np.mean(np.abs(H))) + 1e-12
            G_random = (T_oracle - T_random) / A
            G_hard = (T_oracle - T_hard) / A
            G_true = (H_oracle - H_random) / B
            F_pos = float(np.mean([1.0 if M[i_oracle][j] > 0 else 0.0
                                   for j in range(8) if j != i_oracle]))
            rho_TH = spearman(T, H)
            R_random = (T_oracle / T_random) if T_random > 0 else None
            R_hard = (T_oracle / T_hard) if T_hard > 0 else None
            R_true = (H_oracle / H_random) if H_random > 0 else None

            nan_inf = int(sum(1 for i in range(8) for j in range(8)
                              if not math.isfinite(M[i][j]))
                        + sum(1 for v in T if not math.isfinite(v))
                        + sum(1 for v in H if not math.isfinite(v)))

            # diagnostics (prereg section 10, report-only)
            centers = np.array([base_panel[j]["center"] for j in range(8)])
            pose_d = [[float(np.linalg.norm(centers[i] - centers[j]))
                       for j in range(8)] for i in range(8)]
            iu = [(i, j) for i in range(8) for j in range(8) if i != j]
            pose_flat = [pose_d[i][j] for i, j in iu]
            M_flat = [M[i][j] for i, j in iu]
            diag = {
                "pose_dist_vs_Mij_pearson": pearson(pose_flat, M_flat),
                "source_current_loss_vs_Ti_pearson": pearson(src_losses, T),
                "source_self_improvement_Si_vs_Ti_pearson": pearson(S, T),
                "note": "report-only; determines whether transfer is geometry proximity or hard-view selection",
            }

            t_p1 = time.perf_counter()
            rec = {
                "panel": pname, "panel_cams": panel_cams, "holdout_cams": hold_panel,
                "L0_panel": L0_panel, "L0_holdout": L0_hold,
                "base_panel_eval": base_panel, "base_holdout_eval": base_hold,
                "post_panel_eval": post_panel, "post_holdout_eval": post_hold,
                "M_panel": M, "T_i": T, "H_i": H, "S_i": S,
                "selectors": {"i_hard": i_hard, "T_hard": T_hard, "H_hard": H_hard,
                              "i_oracle": i_oracle, "T_oracle": T_oracle,
                              "H_oracle": H_oracle, "S_oracle": S_oracle,
                              "T_random": T_random, "H_random": H_random},
                "metrics": {"A": A, "B": B, "G_random": G_random, "G_hard": G_hard,
                            "G_true": G_true, "F_pos": F_pos, "rho_TH": rho_TH,
                            "R_random": R_random, "R_hard": R_hard, "R_true": R_true},
                "nan_inf_count": nan_inf,
                "diagnostics": diag,
                "timing": {"panel_s": t_p1 - t_p0, "matrix_build_s": t_step1 - t_step0,
                           "per_step_ms": (t_step1 - t_step0) / 8 * 1000.0},
            }
            cell_res["panels"][pname] = rec
            P(f"  {pname}: T_oracle={T_oracle:.6f} G_r={G_random:.4f} "
              f"G_h={G_hard:.4f} G_t={G_true:.4f} F_pos={F_pos:.3f} "
              f"rho={rho_TH:.4f} H_oracle={H_oracle:.6f} nan={nan_inf} "
              f"({t_p1-t_p0:.1f}s)")
        cell_res["cell_s"] = time.perf_counter() - t_cell0
        results[f"{scene}@{it}"] = cell_res

    with open(os.path.join(args.out, "CVT0_RESULTS.json"), "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
    P(f"\nwrote CVT0_RESULTS.json ({len(results)} cells)")
    log.close()


if __name__ == "__main__":
    main()
