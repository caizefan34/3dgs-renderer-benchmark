"""P14 O1 (Sections 4, 6, 7, 8) + inline O2 raw-unit collection (11-14).

Stage A: per cell D1(H=32) + F(q,32) for q in {0.10,0.20,0.25,0.30,0.50};
         per scene D0(32) on the original P12 schedule (external ref).
Stage B: per cell D1(H) + F(q,H) for q in {0.10,0.20,0.25}, H in {8,128}.
Every arm starts from a bitwise-identical clone of the 16K model+Adam
state. F arms freeze G_F (params+exp_avg+exp_avg_sq) with a per-step
bitwise guard. Inline O2 units (stream metrics pre/post + rebuild) are
collected for every F arm with q<=0.25 (gate eligibility is decided
later by the aggregators, never by this script).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p14_common as C


def active_from_npz(npz, cell_i, q, N):
    act = npz[f"cell{cell_i}_q{int(round(q * 100)):02d}"].astype(np.int64)
    act_bool = np.zeros(N, dtype=bool)
    act_bool[act] = True
    frozen = np.setdiff1d(np.arange(N, dtype=np.int64), act)
    return act, act_bool, frozen


def o2_unit(ctx, scene, cell_i, cell, q, H, pre_pieces, act, act_bool,
            arm_rec):
    """Collect compiled-stream metrics for one F(q,H) arm (post state)."""
    post_pieces = C.collect_pieces(ctx, cell["cluster_cameras"])
    pre_lab = C.label_cluster(pre_pieces, act_bool)
    post_lab = C.label_cluster(post_pieces, act_bool)
    rb = C.rebuild_comparison(pre_lab, post_lab)
    tot_pre, tot_post = pre_lab["totals"], post_lab["totals"]
    n_seg = max(tot_pre["n_frozen_segments"],
                tot_post["n_frozen_segments"])
    unit = {
        "scene": scene, "cell": cell_i, "seed_camera": cell["seed_camera"],
        "q": q, "H": H, "cameras": cell["cluster_cameras"],
        "state_pre": "16K initial (P13-X ckpt)",
        "state_post": f"after F({q},{H}) arm",
        "pre": tot_pre, "post": tot_post,
        "R_stream_pre": tot_pre["R_stream"],
        "R_stream_post": tot_post["R_stream"],
        "R_stream_max": max(tot_pre["R_stream"], tot_post["R_stream"]),
        "n_frozen_segments_max": n_seg,
        "cache_bytes_fp32_max": n_seg * C.CACHE_SEG_BYTES_FP32,
        "cache_bytes_fp16_max": n_seg * C.CACHE_SEG_BYTES_FP16,
        "cache_gb_fp32_max": n_seg * C.CACHE_SEG_BYTES_FP32 / 1e9,
        "segment_bytes_fp32": C.CACHE_SEG_BYTES_FP32,
        "segment_bytes_fp16": C.CACHE_SEG_BYTES_FP16,
        "rebuild": rb,
        "frozen_bitwise_unchanged": bool(
            arm_rec["freeze_ok"] and arm_rec["finite_ok"]),
        "active_sha256": arm_rec["active_param_sha256_after"],
        "note": ("unit carries raw measured data only; gate eligibility "
                 "is decided by P14_O2_AGG (surviving q per O1)"),
    }
    path = os.path.join(
        C.o2_dir(),
        f"{scene}_cell{cell_i}_q{int(round(q*100)):02d}_H{H}.json")
    C.jwrite(path, unit)
    return unit


def run_scene(scene, gpu, stage):
    t_start = time.time()
    ctx = C.SceneCtx(scene, gpu=gpu)
    clusters = json.load(open(os.path.join(
        C.P14_ROOT, "artifacts", "P14_CAMERA_CLUSTERS.json")))
    npz = np.load(os.path.join(C.P14_ROOT, "raw",
                               f"P14_ACTIVE_{scene}.npz"))
    N = ctx.N
    cells = clusters["scenes"][scene]["cells"]

    # ---------------- D0: original P12 schedule (external ref) --------
    H_A = C.H_STAGE_A
    if stage == "A" and not C.SMOKE:
        ctx.restore_initial()
        seq = C.p12_camera_sequence(ctx.n_cameras, ctx.it0 + H_A, seed=42)
        d0_sched = seq[ctx.it0:ctx.it0 + H_A]
        d0_eval_before = {}
        for ci_cell, cell in enumerate(cells):
            cams = cell["cluster_cameras"] + cell["audit_cameras"]
            d0_eval_before[ci_cell] = ctx.eval_cameras(cams)
        timg = C.run_arm(ctx, d0_sched, H_A, freeze=None)
        d0_eval_after = {}
        for ci_cell, cell in enumerate(cells):
            cams = cell["cluster_cameras"] + cell["audit_cameras"]
            d0_eval_after[ci_cell] = ctx.eval_cameras(cams)
        d0_unit = {
            "scene": scene, "arm": "d0", "H": H_A,
            "schedule": "original P12 camera schedule (seed 42, steps "
                        f"{ctx.it0+1}..{ctx.it0+H_A})",
            "schedule_cameras": [int(x) for x in d0_sched],
            "role": "external reference only (not used in F vs D1 gates)",
            "eval_before_by_cell": d0_eval_before,
            "eval_after_by_cell": d0_eval_after,
            "timings_ms": timg, "finite_ok": C.params_finite(ctx),
        }
        C.jwrite(os.path.join(C.o1_dir(), f"{scene}_d0.json"), d0_unit)
        print(f"[p14:{scene}] D0 done", flush=True)

    # ---------------- cells -------------------------------------------
    cells = clusters["scenes"][scene]["cells"]
    if C.SMOKE:
        cells = cells[:1]
    for ci_cell, cell in enumerate(cells):
        cc = cell["cluster_cameras"]
        au = cell["audit_cameras"]
        ctx.restore_initial()
        ev_before = ctx.eval_cameras(cc + au)
        # split eval: cluster vs audit
        ev_c = {"loss": float(np.mean(ev_before["losses"][:len(cc)])),
                "psnr": float(np.mean(ev_before["psnrs"][:len(cc)]))}
        ev_a = {"loss": float(np.mean(ev_before["losses"][len(cc):])),
                "psnr": float(np.mean(ev_before["psnrs"][len(cc):]))}
        pre_pieces = C.collect_pieces(ctx, cc)

        if stage == "A":
            q_list = [0.20] if C.SMOKE else C.Q_LIST_STAGE_A
            arms = [("d1", None, C.H_STAGE_A)] + [
                ("f", q, C.H_STAGE_A) for q in q_list]
        else:
            q_list = [0.20] if C.SMOKE else C.Q_LIST_LE25
            arms = [("d1", None, 8), ("d1", None, 128)] + [
                ("f", q, h) for h in C.H_STAGE_B for q in q_list]

        for kind, q, H in arms:
            ctx.restore_initial()
            tag = "d1" if kind == "d1" else f"f_q{int(round(q*100)):02d}"
            t0 = time.time()
            if kind == "d1":
                timg = C.run_arm(ctx, cc, H, freeze=None)
                ev_after = ctx.eval_cameras(cc + au)
                ev_after_c = {
                    "loss": float(np.mean(ev_after["losses"][:len(cc)])),
                    "psnr": float(np.mean(ev_after["psnrs"][:len(cc)]))}
                ev_after_a = {
                    "loss": float(np.mean(ev_after["losses"][len(cc):])),
                    "psnr": float(np.mean(ev_after["psnrs"][len(cc):]))}
                unit = {
                    "scene": scene, "cell": ci_cell,
                    "seed_camera": cell["seed_camera"], "arm": "d1",
                    "H": H, "q": None,
                    "cluster_cameras": cc, "audit_cameras": au,
                    "schedule": "cluster cameras cycled in camera-ID order",
                    "L_cluster_before": ev_c["loss"],
                    "L_cluster_after": ev_after_c["loss"],
                    "L_audit_before": ev_a["loss"],
                    "L_audit_after": ev_after_a["loss"],
                    "psnr_cluster_before": ev_c["psnr"],
                    "psnr_cluster_after": ev_after_c["psnr"],
                    "psnr_audit_before": ev_a["psnr"],
                    "psnr_audit_after": ev_after_a["psnr"],
                    "eval_before_full": ev_before,
                    "eval_after_full": ev_after,
                    "G_train": ev_c["loss"] - ev_after_c["loss"],
                    "timings_ms": timg,
                    "finite_ok": C.params_finite(ctx),
                }
            else:
                act, act_bool, frozen = active_from_npz(npz, ci_cell, q, N)
                act_idx = torch.as_tensor(act, dtype=torch.long,
                                          device="cuda")
                fr_idx = torch.as_tensor(frozen, dtype=torch.long,
                                         device="cuda")
                fr_param_sha0 = C.rows_sha256(ctx, fr_idx, "param")
                fr_m_sha0 = C.rows_sha256(ctx, fr_idx, "exp_avg")
                fr_v_sha0 = C.rows_sha256(ctx, fr_idx, "exp_avg_sq")
                timg = C.run_arm(ctx, cc, H, freeze=(act_idx, fr_idx))
                fr_param_sha1 = C.rows_sha256(ctx, fr_idx, "param")
                fr_m_sha1 = C.rows_sha256(ctx, fr_idx, "exp_avg")
                fr_v_sha1 = C.rows_sha256(ctx, fr_idx, "exp_avg_sq")
                act_sha1 = C.rows_sha256(ctx, act_idx, "param")
                freeze_ok = (fr_param_sha0 == fr_param_sha1
                             and fr_m_sha0 == fr_m_sha1
                             and fr_v_sha0 == fr_v_sha1
                             and timg["freeze_checks"] == H)
                ev_after = ctx.eval_cameras(cc + au)
                ev_after_c = {
                    "loss": float(np.mean(ev_after["losses"][:len(cc)])),
                    "psnr": float(np.mean(ev_after["psnrs"][:len(cc)]))}
                ev_after_a = {
                    "loss": float(np.mean(ev_after["losses"][len(cc):])),
                    "psnr": float(np.mean(ev_after["psnrs"][len(cc):]))}
                unit = {
                    "scene": scene, "cell": ci_cell,
                    "seed_camera": cell["seed_camera"], "arm": f"f",
                    "q": q, "H": H,
                    "cluster_cameras": cc, "audit_cameras": au,
                    "schedule": "cluster cameras cycled in camera-ID order",
                    "n_active": int(len(act)), "n_frozen": int(len(frozen)),
                    "L_cluster_before": ev_c["loss"],
                    "L_cluster_after": ev_after_c["loss"],
                    "L_audit_before": ev_a["loss"],
                    "L_audit_after": ev_after_a["loss"],
                    "psnr_cluster_before": ev_c["psnr"],
                    "psnr_cluster_after": ev_after_c["psnr"],
                    "psnr_audit_before": ev_a["psnr"],
                    "psnr_audit_after": ev_after_a["psnr"],
                    "eval_after_full": ev_after,
                    "G_train": ev_c["loss"] - ev_after_c["loss"],
                    "frozen_param_sha256_before": fr_param_sha0,
                    "frozen_param_sha256_after": fr_param_sha1,
                    "frozen_exp_avg_sha256_before": fr_m_sha0,
                    "frozen_exp_avg_sha256_after": fr_m_sha1,
                    "frozen_exp_avg_sq_sha256_before": fr_v_sha0,
                    "frozen_exp_avg_sq_sha256_after": fr_v_sha1,
                    "active_param_sha256_after": act_sha1,
                    "freeze_ok": bool(freeze_ok),
                    "freeze_checks": timg["freeze_checks"],
                    "timings_ms": timg,
                    "finite_ok": C.params_finite(ctx),
                }
            path = os.path.join(
                C.o1_dir(),
                f"{scene}_cell{ci_cell}_{tag}_h{H}.json")
            C.jwrite(path, unit)
            print(f"[p14:{scene}] cell {ci_cell} {tag} h{H} "
                  f"({time.time()-t0:.0f}s): G_train="
                  f"{unit['G_train']:.6f} "
                  f"L_cluster {ev_c['loss']:.5f}->"
                  f"{ev_after_c['loss']:.5f} "
                  f"L_audit {ev_a['loss']:.5f}->{ev_after_a['loss']:.5f}"
                  + (f" freeze_ok={unit['freeze_ok']}"
                     if kind == "f" else ""), flush=True)
            # inline O2 units for q<=0.25 F arms
            if kind == "f" and q <= 0.25:
                o2_unit(ctx, scene, ci_cell, cell, q, H, pre_pieces,
                        act, act_bool, unit)
        # cell done
    print(f"[p14:{scene}] stage {stage} complete in "
          f"{time.time()-t_start:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, choices=C.SCENES)
    ap.add_argument("--stage", required=True, choices=("A", "B"))
    ap.add_argument("--gpu", default="4")
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    print(f"[p14] O1 {args.scene} stage {args.stage} on GPU {args.gpu}; "
          f"gpu snapshot:\n{C.gpu_snapshot()}", flush=True)
    run_scene(args.scene, args.gpu, args.stage)
    print(f"[p14] final gpu snapshot:\n{C.gpu_snapshot()}", flush=True)


if __name__ == "__main__":
    main()
