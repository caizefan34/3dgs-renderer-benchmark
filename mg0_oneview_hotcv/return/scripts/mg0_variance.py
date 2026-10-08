#!/usr/bin/env python3
"""MG0 stages 2-4 — hot controls + variance / estimator evaluation.

Usage: mg0_variance.py <room|bicycle> <15000|20000|25000>

Stage 15000 (prereg §4):
  Pass A: streaming fp64 cross-view score in Adam-preconditioned coordinates
          score_j = mean_i ||P_15K(g_i,j) - mean_v P_15K(g_v,j)||^2,
          P_t(x) = x/(sqrt(v_t)+eps) with the checkpoint's per-group eps.
          Freeze nested K=2048/8192/32768 by descending score, ties by row
          index (stable argsort). -> mg0_hotset_*_15000.json
  Pass B: store, for H_32768 only, each of the 64 views' RAW gradient rows in
          FP16 + the FP32 exact mean c_bar_K of those same stored values.

Stage 20000 (prereg §5):
  Pass A: FP32 64-view reference G_20K (mean of fresh grads) + 20K hot ranking
          (same frozen rule). -> mg0_hotset_*_20000.json
  Pass B: per-view estimator metrics for every K using the frozen 15K controls
          v_i,K = g_i,20K - c_i,15K,K + c_bar_15K,K (zero outside the hot set),
          computed without storing all current gradients; also stores the 20K
          H_32768 controls in FP16 for the 25K test.

Stage 25000 (prereg §6):
  Same as 20K with the frozen 20K controls/hot sets; no new store. The 25K
  ranking is recorded for completeness only.

Telemetry only: no renderer change, no optimizer step, no densification, no
loss/resolution change anywhere in this script.
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mg0_core as C


def _agg(vals):
    a = np.asarray(vals, dtype=np.float64)
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "p10": float(np.percentile(a, 10)), "min": float(a.min()),
            "max": float(a.max())}


def _new_metrics():
    m = {"cos_g": {g: [] for g, _, _ in C.GROUPS}, "cos_g_global": [],
         "g_sq": {g: 0.0 for g, _, _ in C.GROUPS}, "g_sq_global": 0.0,
         "dot_gG": {g: 0.0 for g, _, _ in C.GROUPS}, "dot_gG_global": 0.0,
         "raw_ss": {g: 0.0 for g, _, _ in C.GROUPS}, "raw_ss_global": 0.0,
         "adam_raw_ss": {g: 0.0 for g, _, _ in C.GROUPS}, "adam_raw_ss_global": 0.0,
         "pos_raw": {g: [] for g, _, _ in C.GROUPS},
         "raw_ss_view": [], "adam_raw_ss_view": [],
         "g_norm_view": [], "pos_raw_view": []}
    for K in C.K_LIST:
        m[K] = {"v_sq": {g: 0.0 for g, _, _ in C.GROUPS}, "v_sq_global": 0.0,
                "dot_vG": {g: 0.0 for g, _, _ in C.GROUPS}, "dot_vG_global": 0.0,
                "vc_ss": {g: 0.0 for g, _, _ in C.GROUPS}, "vc_ss_global": 0.0,
                "adam_vc_ss": {g: 0.0 for g, _, _ in C.GROUPS},
                "adam_vc_ss_global": 0.0,
                "cos_v": {g: [] for g, _, _ in C.GROUPS}, "cos_v_global": [],
                "pos_v": {g: [] for g, _, _ in C.GROUPS},
                "vc_ss_view": [], "adam_vc_ss_view": [],
                "v_norm_view": [], "pos_v_view": []}
    return m


def _accumulate_view(m, i, g, G, denom, G_sq, controls):
    """One fresh 20K/25K view gradient; update all metric accumulators.

    controls = None (no metrics at 15K) or dict(K -> (rows_K, c_all_fp32, cbar_K))
    with c_all_fp32 [64,K,59] cuda fp32 (indexed by the view's position i).
    """
    raw_delta = g - G
    raw = C.group_sumsq(raw_delta)
    adam_raw = C.group_sumsq(raw_delta / denom)
    gsq = C.group_sumsq(g)
    dot = C.group_dot(g, G)
    raw_global = sum(raw.values())
    adam_raw_global = sum(adam_raw.values())
    gsq_global = sum(gsq.values())
    dot_global = dot["global"]
    m["raw_ss_global"] += raw_global
    m["adam_raw_ss_global"] += adam_raw_global
    m["g_sq_global"] += gsq_global
    m["dot_gG_global"] += dot_global
    m["raw_ss_view"].append(raw_global)
    m["adam_raw_ss_view"].append(adam_raw_global)
    m["g_norm_view"].append(math.sqrt(gsq_global))
    m["pos_raw_view"].append(1.0 if dot_global > 0 else 0.0)
    for grp, _, _ in C.GROUPS:
        m["raw_ss"][grp] += raw[grp]
        m["adam_raw_ss"][grp] += adam_raw[grp]
        m["g_sq"][grp] += gsq[grp]
        m["dot_gG"][grp] += dot[grp]
        cos = dot[grp] / math.sqrt(gsq[grp] * G_sq[grp]) \
            if gsq[grp] > 0 and G_sq[grp] > 0 else float("nan")
        m["cos_g"][grp].append(cos)
        m["pos_raw"][grp].append(1.0 if dot[grp] > 0 else 0.0)
    m["cos_g_global"].append(
        dot_global / math.sqrt(gsq_global * sum(G_sq.values()))
        if gsq_global > 0 else float("nan"))

    if controls is None:
        return
    for K in C.K_LIST:
        rows, c_all, cbK = controls[K]
        cK = c_all[i]                            # this view's stale control rows
        vb = g.clone()
        vb[rows] = (g[rows] - cK) + cbK          # exact estimator semantics
        vd = vb - G
        vcs = C.group_sumsq(vd)
        adam_vcs = C.group_sumsq(vd / denom)
        vsq = C.group_sumsq(vb)
        dotv = C.group_dot(vb, G)
        vc_global = sum(vcs.values())
        adam_vc_global = sum(adam_vcs.values())
        vsq_global = sum(vsq.values())
        dotv_global = dotv["global"]
        kk = m[K]
        kk["vc_ss_global"] += vc_global
        kk["adam_vc_ss_global"] += adam_vc_global
        kk["v_sq_global"] += vsq_global
        kk["dot_vG_global"] += dotv_global
        kk["vc_ss_view"].append(vc_global)
        kk["adam_vc_ss_view"].append(adam_vc_global)
        kk["v_norm_view"].append(math.sqrt(vsq_global))
        kk["pos_v_view"].append(1.0 if dotv_global > 0 else 0.0)
        for grp, _, _ in C.GROUPS:
            kk["vc_ss"][grp] += vcs[grp]
            kk["adam_vc_ss"][grp] += adam_vcs[grp]
            kk["v_sq"][grp] += vsq[grp]
            kk["dot_vG"][grp] += dotv[grp]
            cos = dotv[grp] / math.sqrt(vsq[grp] * G_sq[grp]) \
                if vsq[grp] > 0 and G_sq[grp] > 0 else float("nan")
            kk["cos_v"][grp].append(cos)
            kk["pos_v"][grp].append(1.0 if dotv[grp] > 0 else 0.0)
        kk["cos_v_global"].append(
            dotv_global / math.sqrt(vsq_global * sum(G_sq.values()))
            if vsq_global > 0 else float("nan"))


def _finalize_metrics(m, n_views, G_sq):
    out = {}
    gsq_total = sum(G_sq.values())
    raw_var = m["raw_ss_global"] / n_views
    adam_raw_var = m["adam_raw_ss_global"] / n_views
    gnorms = np.asarray(m["g_norm_view"])
    gains_global = {}
    out["raw"] = {
        "raw_var": raw_var,
        "adam_raw_var": adam_raw_var,
        "mean_cos_g": float(np.mean(m["cos_g_global"])),
        "median_cos_g": float(np.median(m["cos_g_global"])),
        "positive_dot_fraction_raw": float(np.mean(m["pos_raw_view"])),
        "mean_norm_ratio_to_G": float(gnorms.mean() / math.sqrt(gsq_total)),
        "median_norm_ratio_to_G": float(np.median(gnorms) / math.sqrt(gsq_total)),
        "G_norm": math.sqrt(gsq_total),
        "per_group": {},
    }
    for grp, _, _ in C.GROUPS:
        out["raw"]["per_group"][grp] = {
            "raw_var": m["raw_ss"][grp] / n_views,
            "adam_raw_var": m["adam_raw_ss"][grp] / n_views,
            "median_cos_g": float(np.median(m["cos_g"][grp])),
            "G_norm": math.sqrt(G_sq[grp]),
        }
    for K in C.K_LIST:
        kk = m[K]
        vc_var = kk["vc_ss_global"] / n_views
        adam_vc_var = kk["adam_vc_ss_global"] / n_views
        vnorms = np.asarray(kk["v_norm_view"])
        gain_g = np.asarray(kk["cos_v_global"]) - np.asarray(m["cos_g_global"])
        rec = {
            "raw_var": raw_var,
            "hotcv_var": vc_var,
            "raw_var_ratio": vc_var / raw_var,
            "adam_raw_var": adam_raw_var,
            "adam_hotcv_var": adam_vc_var,
            "adam_var_ratio": adam_vc_var / adam_raw_var,
            "median_cosine_gain": float(np.median(gain_g)),
            "p10_cosine_gain": float(np.percentile(gain_g, 10)),
            "mean_cosine_gain": float(gain_g.mean()),
            "mean_cos_g": float(np.mean(m["cos_g_global"])),
            "mean_cos_v": float(np.mean(kk["cos_v_global"])),
            "positive_dot_fraction": float(np.mean(kk["pos_v_view"])),
            "positive_dot_fraction_raw": float(np.mean(m["pos_raw_view"])),
            "positive_dot_fraction_not_decreased": bool(
                float(np.mean(kk["pos_v_view"])) >= float(np.mean(m["pos_raw_view"]))),
            "mean_norm_ratio_to_G": float(vnorms.mean() / math.sqrt(gsq_total)),
            "median_norm_ratio_to_G": float(np.median(vnorms) / math.sqrt(gsq_total)),
            "raw_norm_ratio_to_G": float(gnorms.mean() / math.sqrt(gsq_total)),
            "per_group": {},
            "per_view": {
                "cos_g": [float(x) for x in m["cos_g_global"]],
                "cos_v": [float(x) for x in kk["cos_v_global"]],
                "cosine_gain": [float(x) for x in gain_g],
                "raw_ss": [float(x) for x in m["raw_ss_view"]],
                "hotcv_ss": [float(x) for x in kk["vc_ss_view"]],
                "adam_raw_ss": [float(x) for x in m["adam_raw_ss_view"]],
                "adam_hotcv_ss": [float(x) for x in kk["adam_vc_ss_view"]],
            },
        }
        for grp, _, _ in C.GROUPS:
            gainp = np.asarray(kk["cos_v"][grp]) - np.asarray(m["cos_g"][grp])
            rec["per_group"][grp] = {
                "raw_var": m["raw_ss"][grp] / n_views,
                "hotcv_var": kk["vc_ss"][grp] / n_views,
                "raw_var_ratio": (kk["vc_ss"][grp] / m["raw_ss"][grp])
                if m["raw_ss"][grp] > 0 else float("nan"),
                "adam_raw_var": m["adam_raw_ss"][grp] / n_views,
                "adam_hotcv_var": kk["adam_vc_ss"][grp] / n_views,
                "adam_var_ratio": (kk["adam_vc_ss"][grp] / m["adam_raw_ss"][grp])
                if m["adam_raw_ss"][grp] > 0 else float("nan"),
                "median_cosine_gain": float(np.median(gainp)),
                "p10_cosine_gain": float(np.percentile(gainp, 10)),
                "mean_cos_g": float(np.mean(m["cos_g"][grp])),
                "mean_cos_v": float(np.mean(kk["cos_v"][grp])),
                "positive_dot_fraction": float(np.mean(kk["pos_v"][grp])),
                "positive_dot_fraction_raw": float(np.mean(m["pos_raw"][grp])),
                "positive_dot_fraction_not_decreased": bool(
                    float(np.mean(kk["pos_v"][grp])) >= float(np.mean(m["pos_raw"][grp]))),
            }
        out[str(K)] = rec
    return out


def main():
    scene = sys.argv[1]
    stage = int(sys.argv[2])
    assert scene in C.SCENES and stage in C.STAGE_CKPTS
    t_start = time.time()
    cams = C.load_camera_set(scene)
    views = [int(v) for v in cams["camera_ids"]]
    assert len(views) == C.N_VIEWS
    print(f"[mg0_variance] scene={scene} stage={stage} views={views}", flush=True)

    C.bootstrap_c0()
    fused_loss = C.make_fused_loss()
    dataset = C.load_scene_dataset(scene)
    ck = C.load_checkpoint(scene, stage)
    model = C.build_model(ck)
    C.scene_reset()
    del ck
    torch.cuda.empty_cache()
    N = int(model._xyz.shape[0])
    denom = C.flat_denom(model)
    print(f"[mg0_variance] N={N} mem={C.gpu_mem_gb():.1f}GB", flush=True)

    # ---------------- Pass A1: score mean + (stage>15K) FP32 reference -------
    t0 = time.time()
    S_P = torch.zeros((N, C.ROW_DIM), dtype=torch.float64, device="cuda")
    S_G = (torch.zeros((N, C.ROW_DIM), dtype=torch.float64, device="cuda")
           if stage > 15000 else None)
    view_losses = []
    for v in views:
        cam, gt = dataset.get_item(v)
        loss, g = C.view_grad_and_loss(model, cam, gt, fused_loss)
        view_losses.append(loss)
        S_P += (g / denom).double()
        if S_G is not None:
            S_G += g.double()
        del g
    m_P = S_P / len(views)
    G = (S_G / len(views)).float() if S_G is not None else None
    del S_P, S_G
    t_a1 = time.time() - t0
    print(f"[mg0_variance] Pass A1 done in {t_a1:.0f}s mem={C.gpu_mem_gb():.1f}GB", flush=True)

    # ---------------- Pass A2: score (Q accumulation) ------------------------
    t0 = time.time()
    Q = torch.zeros((N, C.ROW_DIM), dtype=torch.float64, device="cuda")
    for v in views:
        cam, gt = dataset.get_item(v)
        loss, g = C.view_grad_and_loss(model, cam, gt, fused_loss)
        Q += (g / denom - m_P) ** 2
        del g
    score = (Q.sum(dim=1) / len(views))          # fp64 per-row score
    del Q, m_P
    rank = torch.argsort(score, descending=True, stable=True).to(torch.int64)
    hot_ids_top = rank[:C.K_MAIN].cpu()
    t_a2 = time.time() - t0
    print(f"[mg0_variance] Pass A2 done in {t_a2:.0f}s mem={C.gpu_mem_gb():.1f}GB", flush=True)

    score_np = score.float().cpu().numpy()
    score_path = os.path.join(C.ARTIFACTS, f"MG0_SCORE_{scene}_{stage}.npy")
    np.save(score_path, score_np)
    score_stats = {"sum": float(score.double().sum()), "mean": float(score.mean()),
                   "max": float(score.max()), "min": float(score.min()),
                   "n_zero": int((score == 0).sum()),
                   "n_nonfinite": int((~torch.isfinite(score)).sum())}
    hot_json = {"campaign": "MG0_OneView_GS_HotCV", "scene": scene,
                "source_checkpoint": stage, "n_views": len(views), "N_rows": N,
                "rule": "score_j = mean_i ||P_t(g_i,j) - mean_v P_t(g_v,j)||^2; "
                        "P_t(x)=x/(sqrt(v_t)+eps), per-group eps from checkpoint; "
                        "two-pass fp64 streaming; descending; stable argsort "
                        "(ties by row index)",
                "score_stats": score_stats, "score_file": os.path.basename(score_path),
                "score_file_sha256": C.sha256_file(score_path),
                "K": {}}
    score_cpu = score.cpu()
    for K in C.K_LIST:
        ids_K = hot_ids_top[:K]
        hot_json["K"][str(K)] = {
            "ids": [int(x) for x in ids_K],
            "min_score_kept": float(score_cpu[ids_K[-1]]),
            "max_score_kept": float(score_cpu[ids_K[0]]),
            "score_sum": float(score_cpu[ids_K].double().sum()),
        }
    C.json_dump(hot_json, os.path.join(C.ARTIFACTS,
                                       f"mg0_hotset_{scene}_{stage}.json"))

    # ---------------- Pass B ------------------------------------------------
    t0 = time.time()
    controls = None
    if stage > 15000:
        ctrl = C.load_controls(scene, stage - C.N_STALE_LAG)
        assert ctrl["row_dim"] == C.ROW_DIM
        ctrl_ids = ctrl["hot_ids"].to("cuda")
        ctrl_c = ctrl["c_fp16"]
        controls = {}
        for K in C.K_LIST:
            cbarK = ctrl["cbar"][K].to("cuda")
            # c_all: [64, K, 59] cuda fp32 — view i's stale control rows are
            # c_all[i] (indexed by the view's position in the frozen 64 list).
            controls[K] = (ctrl_ids[:K], ctrl_c[:, :K, :].to("cuda").float(),
                           cbarK)
        print(f"[mg0_variance] loaded {stage - C.N_STALE_LAG} controls: "
              f"hot_ids={tuple(ctrl['hot_ids'][:4].tolist())}...", flush=True)

    store = stage in (15000, 20000)
    c_fp16 = None
    cbar_acc = None
    if store:
        c_fp16 = torch.empty((len(views), C.K_MAIN, C.ROW_DIM), dtype=torch.float16)
        cbar_acc = {K: torch.zeros((K, C.ROW_DIM), dtype=torch.float64)
                    for K in C.K_LIST}

    metrics = _new_metrics() if stage > 15000 else None
    G_sq = C.group_sumsq(G) if G is not None else None

    for i, v in enumerate(views):
        cam, gt = dataset.get_item(v)
        loss, g = C.view_grad_and_loss(model, cam, gt, fused_loss)
        if store:
            c_fp16[i] = g.index_select(0, hot_ids_top.to("cuda")).to(
                "cpu", torch.float16)
            for K in C.K_LIST:
                cbar_acc[K] += c_fp16[i][:K].double()
        if metrics is not None:
            _accumulate_view(metrics, i, g, G, denom, G_sq, controls)
        del g
        if (i + 1) % 16 == 0:
            print(f"[mg0_variance] Pass B {i+1}/{len(views)} "
                  f"mem={C.gpu_mem_gb():.1f}GB", flush=True)
    t_b = time.time() - t0
    print(f"[mg0_variance] Pass B done in {t_b:.0f}s", flush=True)

    # ---- store controls (15K/20K) ------------------------------------------
    ctrl_meta = None
    if store:
        cbar = {K: (cbar_acc[K] / len(views)).float() for K in C.K_LIST}
        ctrl_meta = {
            "scene": scene, "stage": stage, "n_views": len(views),
            "views": views, "N_rows": N, "K_main": C.K_MAIN,
            "row_layout": ["xyz3", "scale3", "quat4", "opacity1", "sh_dc3", "sh_rest45"],
            "cbar_fp32_exact_mean_of_stored_fp16": True,
        }
        path = C.save_controls(scene, stage, hot_ids_top.clone(), c_fp16, cbar,
                               ctrl_meta)
        ctrl_meta["controls_path"] = path
        ctrl_meta["controls_file_sha256"] = C.sha256_file(path)
        ctrl_meta["cbar_stats"] = {str(K): {"sum": float(cbar[K].double().sum()),
                                            "absmax": float(cbar[K].abs().max())}
                                   for K in C.K_LIST}
        C.json_dump(ctrl_meta, os.path.join(
            C.ARTIFACTS, f"mg0_controls_meta_{scene}_{stage}.json"))
        print(f"[mg0_variance] saved controls -> {path}", flush=True)

    # ---- finalize -----------------------------------------------------------
    out = {"campaign": "MG0_OneView_GS_HotCV", "scene": scene,
           "checkpoint": stage, "N_rows": N, "n_views": len(views),
           "views": views, "view_fused_losses": view_losses,
           "loss_mean": float(np.mean(view_losses)),
           "G_norm": float(math.sqrt(sum(G_sq.values()))) if G_sq else None,
           "timing": {"pass_a1_s": t_a1, "pass_a2_s": t_a2, "pass_b_s": t_b,
                      "total_s": time.time() - t_start},
           "memory_gb_max": C.gpu_mem_gb(),
           "gpu_uuid": os.environ.get("MG0_GPU_UUID", "unknown"),
           "hotset_file": f"mg0_hotset_{scene}_{stage}.json",
           "controls": ctrl_meta}
    if metrics is not None:
        out["metrics"] = _finalize_metrics(metrics, len(views), G_sq)
        out["metrics_SOURCE"] = ("controls from checkpoint "
                                 f"{stage - C.N_STALE_LAG}; preconditioner P_t "
                                 f"uses the v_t of checkpoint {stage}")
    out["script_sha256"] = {"mg0_variance.py": C.sha256_file(os.path.abspath(__file__)),
                            "mg0_core.py": C.sha256_file(os.path.join(
                                os.path.dirname(os.path.abspath(__file__)), "mg0_core.py"))}
    C.json_dump(out, os.path.join(C.ARTIFACTS, f"mg0_variance_{scene}_{stage}_raw.json"))
    print(f"[mg0_variance] {scene}@{stage} COMPLETE in "
          f"{time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
