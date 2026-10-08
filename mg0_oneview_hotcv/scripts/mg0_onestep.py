#!/usr/bin/env python3
"""MG0 Stage 6 — one-step optimizer integration test (prereg §9).

Runs ONLY if MG0_GATE.json selected a passing K* (strong or actionable).

Cells: room@25000, bicycle@20000, bicycle@25000.
Per cell: 32 frozen anchor views (from the frozen 64-camera set; their stale
controls exist for every anchor) + the disjoint frozen 16-camera internal
training-view evaluation set.

Per anchor, from the exact restored checkpoint + Adam state:
  B1 (native): the anchor's ordinary single-view gradient + ordinary Adam step.
  B2 (HotCV):  the SAME fresh gradient, replaced by the K* HotCV estimator
               (v = g - c_i + c_bar on the hot rows; unchanged elsewhere),
               then the unchanged Adam step.
Evaluation: original full-resolution frozen loss (0.8*L1+0.2*(1-SSIM)) on the
16 internal views before and after the update. DeltaHeldoutLoss = L_before -
L_after. Paired B2-B1 differences and the fraction B2 > B1 are reported.
A deterministic 25% anchor subset (stride-4) repeats B1 three times total to
estimate numerical noise.

No second current view. No loss modification. No LR modification.
"""
import json
import math
import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mg0_core as C

CELLS = (("room", 25000), ("bicycle", 20000), ("bicycle", 25000))
NOISE_STRIDE = 4   # anchors[0::4] -> 8 of 32 = 25%


def apply_rows_to_grads(model, g, N):
    """Copy a [N,59] row buffer into the model's param .grad tensors exactly.

    Materializes .grad when StateMaster.restore() left it None
    (zero_grad(set_to_none=True)).
    """
    with torch.no_grad():
        for n in C.PARAM_NAMES:
            p = getattr(model, n)
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        model._xyz.grad.copy_(g[:, 0:3])
        model._scaling.grad.copy_(g[:, 3:6])
        model._rotation.grad.copy_(g[:, 6:10])
        model._opacity.grad.copy_(g[:, 10:11].reshape(-1))
        model._shs.grad[:, 0, :].copy_(g[:, 11:14])
        model._shs.grad[:, 1:, :].copy_(g[:, 14:C.ROW_DIM].reshape(N, 15, 3))


@torch.no_grad()
def eval_internal(model, pairs, fused_loss):
    tot = 0.0
    for cam, gt in pairs:
        img, _, _ = C.render_view(model, cam, model.active_sh_degree)
        tot += float(fused_loss(img, gt).item())
    return tot / len(pairs)


def check_params_finite(model):
    for n in C.PARAM_NAMES:
        p = getattr(model, n)
        if not bool(torch.isfinite(p).all()):
            return False
    return True


def main():
    gate = json.load(open(os.path.join(C.ARTIFACTS, "MG0_GATE.json")))
    if not gate["run_one_step"]:
        print(f"[mg0_onestep] gate={gate['verdict']} -> stage 6 NOT RUN", flush=True)
        return
    Kstar = int(gate["K_star"])
    print(f"[mg0_onestep] gate={gate['verdict']} K*={Kstar}", flush=True)

    C.bootstrap_c0()
    fused_loss = C.make_fused_loss()
    all_rows = []
    noise = {}
    summaries = {}

    for scene, stage in CELLS:
        t0 = time.time()
        cams = C.load_camera_set(scene)
        views = [int(v) for v in cams["camera_ids"]]
        pos_of = {v: i for i, v in enumerate(views)}
        anchors = [int(v) for v in cams["anchors_32"]]
        internal = [int(v) for v in cams["internal_eval_16"]]

        dataset = C.load_scene_dataset(scene)
        ck = C.load_checkpoint(scene, stage)
        model = C.build_model(ck)
        C.scene_reset()
        del ck
        torch.cuda.empty_cache()
        N = int(model._xyz.shape[0])
        sm = C.StateMaster(model)

        ctrl = C.load_controls(scene, stage - C.N_STALE_LAG)
        rows = ctrl["hot_ids"][:Kstar].to("cuda")
        internal_pairs = [dataset.get_item(v) for v in internal]

        cell_rows = []
        for ai, av in enumerate(anchors):
            pos = pos_of[av]
            c_i = ctrl["c_fp16"][pos, :Kstar, :].to("cuda").float()
            cb = ctrl["cbar"][Kstar].to("cuda")

            sm.restore()
            L_before = eval_internal(model, internal_pairs, fused_loss)
            cam, gt = dataset.get_item(av)
            loss_anchor, g = C.view_grad_and_loss(model, cam, gt, fused_loss)

            # ---- B1: ordinary single-view gradient + ordinary Adam step
            model.optimizer.step()
            L_b1 = eval_internal(model, internal_pairs, fused_loss)
            norms_b1 = sm.param_update_norm()
            finite_b1 = check_params_finite(model)
            sm.restore()

            # ---- B2: same gradient replaced by the K* HotCV estimator
            g2 = g.clone()
            g2[rows] = (g[rows] - c_i) + cb
            apply_rows_to_grads(model, g2, N)
            model.optimizer.step()
            L_b2 = eval_internal(model, internal_pairs, fused_loss)
            norms_b2 = sm.param_update_norm()
            finite_b2 = check_params_finite(model)
            sm.restore()
            del g2

            d_b1 = L_before - L_b1
            d_b2 = L_before - L_b2
            rec = {
                "scene": scene, "checkpoint": stage, "K": Kstar,
                "anchor_index": ai, "anchor_view": av,
                "anchor_fused_loss": loss_anchor,
                "L_before": L_before, "L_after_b1": L_b1, "L_after_b2": L_b2,
                "delta_b1": d_b1, "delta_b2": d_b2,
                "diff_b2_minus_b1": d_b2 - d_b1,
                "b2_better": bool(d_b2 > d_b1),
                "params_finite_b1": finite_b1, "params_finite_b2": finite_b2,
                "is_noise_subset": bool(ai % NOISE_STRIDE == 0),
            }
            for gname, v in norms_b1.items():
                rec[f"upd_b1_{gname}"] = v
            for gname, v in norms_b2.items():
                rec[f"upd_b2_{gname}"] = v
            cell_rows.append(rec)
            if rec["is_noise_subset"]:
                key = f"{scene}@{stage}#{ai}"
                reps = [d_b1]
                for _ in range(2):
                    sm.restore()
                    apply_rows_to_grads(model, g, N)
                    model.optimizer.step()
                    Lr = eval_internal(model, internal_pairs, fused_loss)
                    reps.append(L_before - Lr)
                    sm.restore()
                noise[key] = {"scene": scene, "checkpoint": stage,
                              "anchor_index": ai, "anchor_view": av,
                              "K": Kstar, "b1_repeats": reps,
                              "span": max(reps) - min(reps),
                              "std": float(np.std(reps, ddof=0))}
            if (ai + 1) % 8 == 0:
                print(f"[mg0_onestep] {scene}@{stage} anchors {ai+1}/{len(anchors)} "
                      f"mem={C.gpu_mem_gb():.1f}GB", flush=True)

        spans = [v["span"] for v in noise.values() if v["scene"] == scene
                 and v["checkpoint"] == stage]
        all_rows.extend(cell_rows)
        noise_level = float(max(spans)) if spans else float("nan")
        diffs = np.asarray([r["diff_b2_minus_b1"] for r in cell_rows],
                           dtype=np.float64)
        d_b1s = np.asarray([r["delta_b1"] for r in cell_rows], dtype=np.float64)
        d_b2s = np.asarray([r["delta_b2"] for r in cell_rows], dtype=np.float64)
        median_diff = float(np.median(diffs))
        frac = float(np.mean([bool(r["b2_better"]) for r in cell_rows]))
        # mechanical instability criterion (operationalized; see
        # MG0_SPEC_OPERATIONALIZATION.json): non-finite params OR any group
        # median B2 update norm > 3x median B1 update norm.
        group_flags = {}
        for gname in [g for g, _, _ in C.GROUPS] + ["global"]:
            b1m = float(np.median([r[f"upd_b1_{gname}"] for r in cell_rows]))
            b2m = float(np.median([r[f"upd_b2_{gname}"] for r in cell_rows]))
            r = b2m / max(b1m, 1e-30)
            group_flags[gname] = {"median_ratio_b2_over_b1": r,
                                  "median_b1": b1m, "median_b2": b2m,
                                  "instability": bool(r > 3.0)}
        unstable = (not all(r["params_finite_b1"] and r["params_finite_b2"]
                            for r in cell_rows)) \
            or any(v["instability"] for v in group_flags.values())
        summaries[f"{scene}@{stage}"] = {
            "scene": scene, "checkpoint": stage, "K": Kstar,
            "n_anchors": len(cell_rows),
            "frac_b2_gt_b1": frac,
            "median_paired_diff": median_diff,
            "mean_paired_diff": float(diffs.mean()),
            "median_delta_b1": float(np.median(d_b1s)),
            "median_delta_b2": float(np.median(d_b2s)),
            "mean_delta_b1": float(d_b1s.mean()),
            "mean_delta_b2": float(d_b2s.mean()),
            "b1_noise_level": noise_level,
            "b1_noise_spans": spans,
            "median_positive_beyond_noise": bool(median_diff > noise_level),
            "no_catastrophic_group_instability": bool(not unstable),
            "group_update_ratios": group_flags,
            "elapsed_s": time.time() - t0,
        }
        del model, sm, ctrl, dataset, internal_pairs
        torch.cuda.empty_cache()
        print(f"[mg0_onestep] {scene}@{stage}: frac={frac:.3f} "
              f"median_diff={median_diff:.3e} noise={noise_level:.3e} "
              f"({time.time()-t0:.0f}s)", flush=True)

    table = pa.Table.from_pylist(all_rows)
    pq.write_table(table, os.path.join(C.ARTIFACTS, "MG0_ONE_STEP.parquet"),
                   compression="zstd")
    C.json_dump({"campaign": "MG0_OneView_GS_HotCV", "stage": "6_one_step",
                 "K_star": Kstar, "gate_verdict": gate["verdict"],
                 "noise_protocol": "deterministic 25% anchor subset (anchors[0::4], "
                                   "8 of 32 per cell) repeats B1 three times total",
                 "noise_level_definition": "max over the subset of (max-min) of the "
                                           "three B1 DeltaHeldoutLoss repeats",
                 "cells": summaries},
                os.path.join(C.ARTIFACTS, "mg0_onestep_summary.json"))
    C.json_dump({"campaign": "MG0_OneView_GS_HotCV", "stage": "6_b1_noise",
                 "K_star": Kstar, "anchors": noise},
                os.path.join(C.ARTIFACTS, "MG0_B1_NOISE.json"))
    print("[mg0_onestep] COMPLETE", flush=True)


if __name__ == "__main__":
    main()
