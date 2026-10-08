#!/usr/bin/env python3
"""MG0 Stage 5 — isolated control-maintenance cost probe (prereg §7).

Per scene, per checkpoint (15000/20000/25000), per K in {2048, 8192, 32768},
on the real checkpoint tensors and the real stored control tables:

  isolated op sequence (no renderer / no optimizer step):
    1. gather/read stored c_i rows (FP16 -> FP32)
    2. subtract c_i from the current gradient rows
    3. add c_bar to those same rows
    4. refresh c_bar by the exact mean delta when c_i is replaced by the
       current selected rows (new FP16-stored value minus old, / n_cameras)
    5. store the refreshed camera control in FP16

  10 warmups + 1000 timed repeats (CUDA-event pairs + host wall).
  Reports bytes read/written, peak additional GPU memory, and normalizes
  against the already-measured native A1 iteration wall at the same
  checkpoint (N1_ACTION_TIMING_*.json).

Also writes MG0_MEMORY.json: peak additional GPU memory per cell +
projection of the control table to the FULL number of training cameras.
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mg0_core as C

WARMUP = 10
REPEATS = 1000


def control_source_stage(stage):
    return stage - C.N_STALE_LAG if stage > 15000 else 15000


def run_cell(grad_buf, rows, c_row_orig, cbar_orig, n_stored_cams):
    """Timed isolated op sequence; returns summary dict."""
    dev = grad_buf.device
    c_row = c_row_orig.to(dev).clone()          # fp16 working copy [K,59]
    cbar = cbar_orig.to(dev).clone()            # fp32 working copy [K,59]
    M = float(n_stored_cams)
    out = None

    def op():
        nonlocal out
        ci = c_row.to(torch.float32)                       # 1 read
        gv = grad_buf.index_select(0, rows)                # current rows
        out = (gv - ci) + cbar                             # 2 subtract, 3 add
        cn = gv.to(torch.float16)                          # 5a convert
        c_row.copy_(cn)                                    # 5b store FP16
        cbar.add_(cn.to(torch.float32).sub_(ci).div_(M))   # 4 mean-delta refresh

    for _ in range(WARMUP):
        op()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    mem_base = torch.cuda.memory_allocated()

    pairs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
             for _ in range(REPEATS)]
    t_host0 = time.perf_counter()
    for s, e in pairs:
        s.record()
        op()
        e.record()
    torch.cuda.synchronize()
    host_wall = time.perf_counter() - t_host0
    dur = np.array([s.elapsed_time(e) for s, e in pairs], dtype=np.float64)
    peak_extra = torch.cuda.max_memory_allocated() - mem_base
    _ = float(out.abs().sum().item())  # keep the result live

    K, D = c_row.shape
    n = n_stored_cams
    return {
        "K": int(K), "n_stored_cameras": int(n),
        "event_ms": {"median": float(np.median(dur)), "mean": float(dur.mean()),
                     "std": float(dur.std()), "min": float(dur.min()),
                     "max": float(dur.max()), "p90": float(np.percentile(dur, 90))},
        "host_wall_ms_total": host_wall * 1e3,
        "host_wall_ms_per_repeat": host_wall * 1e3 / REPEATS,
        "repeats": REPEATS, "warmups": WARMUP,
        "bytes": {
            "read_fp16_control_rows": K * D * 2,
            "read_fp32_cbar": K * D * 4,
            "read_fp32_current_grad_rows": K * D * 4,
            "write_fp32_temp_corrected_rows": K * D * 4,
            "write_fp16_control_store": K * D * 2,
            "readwrite_fp32_cbar_update": 2 * K * D * 4,
        },
        "peak_extra_gpu_mem_mb": peak_extra / (1024 ** 2),
        "analytic_gpu_resident_mb": (K * D * (2 + 4 + 4 + 4 + 4 + 2) + K * 8)
                                    / (1024 ** 2),
    }


def main():
    cells = {}
    mem = {"campaign": "MG0_OneView_GS_HotCV", "stage": "5_isolated_cost",
           "protocol": {"warmups": WARMUP, "repeats": REPEATS,
                        "timing": "CUDA-event pairs per repeat + host wall",
                        "normalization": "ratio vs native A1 iteration wall median "
                                         "(N1_ACTION_TIMING_*.json)",
                        "note": "isolated control-buffer ops ONLY; no renderer, "
                                "no optimizer step, no param update"},
           "cells": {}, "projected_table_memory": {}}
    projections = {}

    for scene in C.SCENES:
        n_cams = None
        try:
            cs = C.load_camera_set(scene)
            n_cams = int(cs["N_cameras"])
        except Exception:
            pass
        if n_cams is None:
            n_cams = len(json.load(open(os.path.join(
                C.REPO, "data", "official", "mipnerf360", scene, "cameras.json"))))
        projections[scene] = {"N_cameras": n_cams, "K": {}}
        for K in C.K_LIST:
            table = n_cams * K * C.ROW_DIM * 2
            cbar = K * C.ROW_DIM * 4
            ids = K * 8
            total = table + cbar + ids
            projections[scene]["K"][str(K)] = {
                "fp16_camera_table_bytes": table,
                "fp32_cbar_bytes": cbar,
                "int64_hot_ids_bytes": ids,
                "total_bytes": total,
                "total_gib": total / (1024 ** 3),
            }

        C.bootstrap_c0()
        fused_loss = C.make_fused_loss()
        dataset = C.load_scene_dataset(scene)
        cams = C.load_camera_set(scene)
        views = [int(v) for v in cams["camera_ids"]]

        for stage in C.STAGE_CKPTS:
            t0 = time.time()
            ck = C.load_checkpoint(scene, stage)
            model = C.build_model(ck)
            C.scene_reset()
            del ck
            torch.cuda.empty_cache()
            N = int(model._xyz.shape[0])

            src = control_source_stage(stage)
            ctrl = C.load_controls(scene, src)
            hot_ids = ctrl["hot_ids"]

            # one fresh gradient at this checkpoint -> "current gradient rows"
            cam, gt = dataset.get_item(views[0])
            _, grad_buf = C.view_grad_and_loss(model, cam, gt, fused_loss)
            model.optimizer.zero_grad(set_to_none=True)

            for K in C.K_LIST:
                key = f"{scene}|{stage}|{K}"
                rows = hot_ids[:K].to("cuda")
                res = run_cell(grad_buf, rows,
                               ctrl["c_fp16"][0, :K, :], ctrl["cbar"][K],
                               n_stored_cams=len(views))
                a1 = C.a1_median_ms(scene, stage)
                res.update({"scene": scene, "checkpoint": stage,
                            "control_source_checkpoint": src,
                            "a1_median_ms": a1,
                            "a1_source_field": "N1_ACTION_TIMING total_wall_ms.median",
                            "control_maintenance_ratio": res["event_ms"]["median"] / a1,
                            "control_maintenance_ratio_host_wall":
                                res["host_wall_ms_per_repeat"] / a1})
                cells[key] = res
                print(f"[mg0_cost] {key}: ev_med={res['event_ms']['median']*1000:.1f}us "
                      f"ratio={res['control_maintenance_ratio']:.4f} "
                      f"peak_extra={res['peak_extra_gpu_mem_mb']:.1f}MB", flush=True)
            mem["cells"].update({f"{scene}|{stage}|{K}": {
                "peak_extra_gpu_mem_mb": cells[f"{scene}|{stage}|{K}"]["peak_extra_gpu_mem_mb"]}
                for K in C.K_LIST})
            del model, grad_buf, ctrl
            torch.cuda.empty_cache()
            print(f"[mg0_cost] {scene}@{stage} done in {time.time()-t0:.0f}s", flush=True)

    out = {"campaign": "MG0_OneView_GS_HotCV", "stage": "5_isolated_cost",
           "frozen_date": "2026-09-29", "cells": cells,
           "a1_source": "N1_ACTION_TIMING_{scene}.json cells '{ckpt}_A1' total_wall_ms.median",
           "gpu_uuid": os.environ.get("MG0_GPU_UUID", "unknown"),
           "script_sha256": {"mg0_cost.py": C.sha256_file(os.path.abspath(__file__))}}
    C.json_dump(out, os.path.join(C.ARTIFACTS, "MG0_COST.json"))
    mem["projected_table_memory"] = projections
    mem["projection_note"] = ("bytes = N_cameras*K*59*2 (FP16 table) + K*59*4 (FP32 "
                              "cbar) + K*8 (int64 hot ids); FP16 table scaled to the "
                              "FULL training-camera count of the scene")
    C.json_dump(mem, os.path.join(C.ARTIFACTS, "MG0_MEMORY.json"))


if __name__ == "__main__":
    main()
