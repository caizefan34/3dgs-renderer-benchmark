#!/usr/bin/env python3
"""R17-P3: 580K controlled parameter/camera attribution experiment.

8 cells on the exact R8/R15 580K subset (seed 42, sorted choice) at 540p:
  A  raw/raw/raw        + identity camera   (reproduces the R8/R15/R16 protocol)
  B  sigmoid/raw/raw    + identity camera
  B2 raw/exp/raw        + identity camera
  B3 raw/raw/normalize  + identity camera
  C  sigmoid/exp/normalize + identity camera
  C2 raw/raw/raw        + real cam0 (training protocol)
  D  sigmoid/exp/normalize + real cam0
  D2 sigmoid/raw/raw    + real cam0

Per cell: frozen HiGS forward+backward on the UNMODIFIED binary, with the
isect_tiles interceptor recording the real full intersection counts, a seeded
1000-sample CPU-oracle comparison, tile coverage, radius distribution, fwd/bwd
timing, peak VRAM and NaN checks.

Cell A carries the R15 exact-reproduction gate (same seed/sample rule).

Outputs: R17_580K_CONTROLLED_COMPARISON.csv, R17_INTERSECTION_COVERAGE.csv,
         raw/<cell>_r15_style_capture.pt, oracle/<cell>/, raw/<cell>_cell.json
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.cuda import Event

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (EXPECTED, H540, OUT, PROJECT, W540, add_project_to_syspath,
                    build_580k_indices, build_cell_state, ensure_out_dirs,
                    gpu_mem_snapshot, load_checkpoint, load_real_cameras,
                    nan_report, np_stats, read_common_constants, sha256_file,
                    write_json)

CELLS = {
    "A": dict(opacity="raw", scale="raw", quat="raw", camera="identity",
              desc="R8/R15/R16 protocol reproduction"),
    "B": dict(opacity="sigmoid", scale="raw", quat="raw", camera="identity",
              desc="opacity activation only"),
    "B2": dict(opacity="raw", scale="exp", quat="raw", camera="identity",
               desc="scale activation only"),
    "B3": dict(opacity="raw", scale="raw", quat="normalize", camera="identity",
               desc="quaternion normalization only"),
    "C": dict(opacity="sigmoid", scale="exp", quat="normalize", camera="identity",
              desc="all-activated params, synthetic camera"),
    "C2": dict(opacity="raw", scale="raw", quat="raw", camera="real",
               desc="raw params, real cam0"),
    "D": dict(opacity="sigmoid", scale="exp", quat="normalize", camera="real",
              desc="all-activated params, real cam0 (fully correct protocol)"),
    "D2": dict(opacity="sigmoid", scale="raw", quat="raw", camera="real",
               desc="opacity activation only, real cam0"),
}

CSV_FIELDS = [
    "cell", "desc", "opacity_mode", "scale_mode", "quat_mode", "camera_mode",
    "n_gaussians", "n_visible", "culling_ratio", "first_call_full_n", "n_qualifying_calls",
    "sum_counts_first_call", "sum_counts_all_calls", "n_isects_metadata", "n_isects_full_metadata",
    "count_mean", "count_median", "count_p99", "count_max", "frac_counts_gt_tile_bound",
    "radius_mean", "radius_p50", "radius_p99", "radius_max", "frac_radius_zero",
    "coverage_tiles_hit", "coverage_frac", "n_images_first_call",
    "fwd_status", "fwd_ms", "bwd_status", "bwd_ms", "peak_vram_gb",
    "nan_frame", "nan_alpha", "nan_grads", "oracle_mismatch_pairs", "oracle_mismatch_ratio",
    "oracle_verdict", "r15_gate", "error",
]

MIN_FREE_GB = 20.0


def jsonable_meta(meta: dict) -> dict:
    out = {}
    if not isinstance(meta, dict):
        return {"raw": str(meta)}
    for k, v in meta.items():
        if isinstance(v, torch.Tensor):
            out[k] = {"shape": list(v.shape), "dtype": str(v.dtype)}
        elif isinstance(v, (int, float, str, bool, type(None))):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def run_cell(letter, cfg, ckpt, cams540, indices, rasterize_higs, alpha, extend,
             out_dir: Path, skip_backward=False, do_oracle=True, min_free_gb=MIN_FREE_GB):
    from capture_intersections import IsectRecorder, install, uninstall, SafeAbort
    from oracle_higs import run_oracle

    row = {k: None for k in CSV_FIELDS}
    row.update({"cell": letter, "desc": cfg["desc"], "opacity_mode": cfg["opacity"],
                "scale_mode": cfg["scale"], "quat_mode": cfg["quat"],
                "camera_mode": cfg["camera"], "n_gaussians": int(indices.size),
                "r15_gate": "", "error": ""})
    cell_pre = gpu_mem_snapshot(f"{letter}_pre")
    if cell_pre.get("free_gb", 0) < min_free_gb:
        row.update({"fwd_status": "OUT_OF_HEADROOM", "bwd_status": "OUT_OF_HEADROOM"})
        row["error"] = f"free_gb={cell_pre.get('free_gb')} < {min_free_gb}"
        write_json(out_dir / "raw" / f"{letter}_cell.json",
                   {"status": "OUT_OF_HEADROOM", "row": row})
        return row, cell_pre

    if cfg["camera"] == "real" and not cams540:
        row.update({"fwd_status": "ERROR", "bwd_status": "NOT_RUN",
                    "error": "Real cameras.json unavailable; cell not executed"})
        write_json(out_dir / "raw" / f"{letter}_cell.json",
                   {"status": "NOT_EXECUTED_CAMERAS_JSON_MISSING", "row": row})
        return row, cell_pre

    state = build_cell_state(ckpt, indices, cfg["opacity"], cfg["scale"], cfg["quat"],
                             device="cuda")
    if cfg["camera"] == "identity":
        K = torch.tensor([[W540, 0, W540 / 2], [0, H540, H540 / 2], [0, 0, 1]],
                         dtype=torch.float32, device="cuda").unsqueeze(0)
        viewmat = torch.eye(4, device="cuda", dtype=torch.float32).unsqueeze(0)
        cam_desc = "identity origin, fx=960 fy=540"
    else:
        cam = cams540[0]
        K = cam["K"].to("cuda").unsqueeze(0)
        viewmat = cam["viewmatrix"].to("cuda").unsqueeze(0)
        cam_desc = f"real cam0 {cam['img_name']} 540p training protocol"

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    cell_json = {"cell": letter, "config": cfg, "camera": cam_desc,
                 "param_stats_gpu": {}, "pre_mem": cell_pre}

    # GPU-side param summaries (cheap; sample to bound cost)
    with torch.no_grad():
        s = slice(None, None, 41)
        cell_json["param_stats_gpu"] = {
            "opacity": np_stats(state["opacities"][s].detach().cpu().numpy(), "op_"),
            "scales": np_stats(state["scales"][s].detach().cpu().numpy(), "sc_"),
            "quat_norm": np_stats(
                torch.linalg.norm(state["quats"][s], dim=-1).detach().cpu().numpy(), "q_"),
        }

    recorder = IsectRecorder(None, mode="stats", save_dir=out_dir / "raw", tag=letter)
    orig = install(recorder)
    fwd_status, bwd_status = "NOT_RUN", "NOT_RUN"
    fwd_ms = bwd_ms = None
    result = None  # forward may raise before returning (OOM/diagnostic abort)
    frame = render_alpha = None
    result_meta = {}
    err_text = ""
    try:
        start = Event(enable_timing=True); end = Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        result = rasterize_higs(
            state["means"], state["quats"], state["scales"], state["opacities"], state["shs"],
            viewmats=viewmat, Ks=K, width=W540, height=H540,
            sh_degree=3, enable_culling=True, use_higs_culling=True)
        end.record()
        torch.cuda.synchronize()
        fwd_ms = start.elapsed_time(end)
        fwd_status = "PASS"
        frame = result.get("frame", result.get("render_colors"))
        render_alpha = result.get("alpha", result.get("render_alphas"))
        result_meta = {k: jsonable_meta(v) for k, v in result.items() if k != "densification_info"}
        dens = result.get("densification_info") or {}
        if isinstance(dens, dict):
            for key in ("gaussian_ids", "visible_gaussian_ids", "radii", "means2d"):
                v = dens.get(key)
                if torch.is_tensor(v):
                    result_meta[f"dens_{key}"] = {"shape": list(v.shape), "dtype": str(v.dtype)}
                    if key == "visible_gaussian_ids":
                        result_meta["dens_visible_count"] = int(v.numel())

        if not skip_backward:
            start = Event(enable_timing=True); end = Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            loss = frame.sum() + render_alpha.sum()
            loss.backward()
            end.record()
            torch.cuda.synchronize()
            bwd_ms = start.elapsed_time(end)
            bwd_status = "PASS"
    except SafeAbort as e:
        fwd_status = "SAFE_ABORT"
        err_text = json.dumps(e.info, default=str)[:2000]
    except torch.cuda.OutOfMemoryError as e:
        fwd_status = "OOM" if fwd_status == "NOT_RUN" else "OOM_DURING_BWD"
        bwd_status = "OOM" if fwd_status == "OOM_DURING_BWD" else bwd_status
        err_text = repr(e)[:2000]
    except Exception as e:  # noqa: BLE001
        fwd_status = "ERROR"
        err_text = repr(e)[:2000]
        traceback.print_exc()
    finally:
        uninstall(orig)

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    row.update({
        "fwd_status": fwd_status, "bwd_status": bwd_status,
        "fwd_ms": fwd_ms, "bwd_ms": bwd_ms, "peak_vram_gb": peak_gb,
    })

    # ---- recorder-derived metrics -----------------------------------------
    rec_summary = recorder.summary()
    cell_json["recorder"] = rec_summary
    first = rec_summary["first_call_records"] or {}
    row["first_call_full_n"] = first.get("means2d_shape", [None, None])[-2] if first.get("means2d_shape") else None
    row["n_qualifying_calls"] = rec_summary.get("n_qualifying_calls")
    row["sum_counts_first_call"] = first.get("sum_counts")
    row["sum_counts_all_calls"] = int(sum(c.get("sum_counts", 0) for c in rec_summary["qualifying_calls"]))
    cs = first.get("counts_stats") or {}
    row.update({
        "count_mean": cs.get("count_mean"), "count_median": cs.get("count_p50"),
        "count_p99": cs.get("count_p99"), "count_max": cs.get("count_max"),
        "frac_counts_gt_tile_bound": first.get("frac_counts_gt_tile_bound"),
    })
    cov = first.get("coverage") or {}
    row["coverage_tiles_hit"] = cov.get("tiles_hit_total")
    row["coverage_frac"] = cov.get("coverage_frac_total")
    row["n_images_first_call"] = first.get("n_images")

    # metadata from the actual result
    meta = result_meta.get("metadata", {}) if isinstance(result_meta.get("metadata"), dict) else {}
    row["n_visible"] = meta.get("n_visible") or result_meta.get("dens_visible_count")
    row["culling_ratio"] = meta.get("culling_ratio")
    row["n_isects_metadata"] = meta.get("n_isects")
    row["n_isects_full_metadata"] = meta.get("n_isects_full")

    # radius stats from the captured subset tensors (all sample rows)
    cap = recorder.first_capture
    if cap is not None:
        pkg = cap["pkg"]
        radii = pkg["radii"].float().numpy().reshape(-1, 2)
        rmax = radii.max(axis=1)
        rpos = rmax[rmax > 0]
        row["frac_radius_zero"] = float((rmax <= 0).mean()) if rmax.size else None
        if rpos.size:
            rs = np_stats(rpos, "")
            row.update({"radius_mean": rs["mean"], "radius_p50": rs["p50"],
                        "radius_p99": rs["p99"], "radius_max": rs["max"]})

    # ---- R15 gate (cell A) -------------------------------------------------
    if letter == "A":
        from common import R15_CAPTURE
        if not R15_CAPTURE.is_file():
            row["r15_gate"] = "NOT_EXECUTED_R15_CAPTURE_MISSING"
        elif cap is None:
            row["r15_gate"] = "NOT_EXECUTED_NO_CAPTURE"
        else:
            r15 = torch.load(str(R15_CAPTURE), map_location="cpu", weights_only=False)
            same_n = (r15["meta"].get("full_projected_gaussian_count_per_image")
                      == cap["full_n"])
            same_ids = bool(torch.equal(r15["sample_indices"], cap["pkg"]["sample_indices"]))
            same_counts = bool(torch.equal(r15["gpu_counts"], cap["pkg"]["gpu_counts"]))
            row["r15_gate"] = json.dumps({
                "r15_full_n": r15["meta"].get("full_projected_gaussian_count_per_image"),
                "r17_full_n": cap["full_n"], "same_call_size": same_n,
                "same_sample_ids": same_ids, "same_counts": same_counts,
                "r15_sha256": sha256_file(R15_CAPTURE)})[:1500]
            cell_json["r15_gate"] = json.loads(row["r15_gate"])

    # ---- oracle -------------------------------------------------------------
    if do_oracle and cap is not None and alpha and extend:
        try:
            cap_path = out_dir / "raw" / f"{letter}_r15_style_capture.pt"
            o = run_oracle(cap_path, out_dir / "oracle" / letter, alpha, extend)
            valid = int(o.get("valid_pairs", 0)) or 1
            row["oracle_mismatch_pairs"] = o.get("mismatch_pairs")
            row["oracle_mismatch_ratio"] = o.get("mismatch_pairs", 0) / valid
            row["oracle_verdict"] = o.get("verdict")
            cell_json["oracle"] = {k: o[k] for k in
                                   ("mismatch_pairs", "valid_pairs", "invalid_or_skipped_pairs",
                                    "count_above_physical_tiles",
                                    "count_above_padded_aabb_upper_bound",
                                    "reference_sum_valid", "gpu_sample_sum_all", "verdict")}
        except Exception as e:  # noqa: BLE001
            cell_json["oracle_error"] = repr(e)
            row["oracle_verdict"] = "ORACLE_ERROR"

    # ---- NaN checks ---------------------------------------------------------
    nan = nan_report(("frame", frame), ("alpha", render_alpha),
                     *[(f"grad_{k}", state[k].grad) for k in
                       ("means", "quats", "scales", "opacities", "shs")])
    any_grad_nan = any(isinstance(v, dict) and (v["has_nan"] or v["has_inf"])
                       for k, v in nan.items() if k.startswith("grad_"))
    row["nan_frame"] = nan["frame"]["has_nan"] if isinstance(nan["frame"], dict) else None
    row["nan_alpha"] = nan["alpha"]["has_nan"] if isinstance(nan["alpha"], dict) else None
    row["nan_grads"] = any_grad_nan
    cell_json["nan"] = nan
    cell_json["post_mem"] = gpu_mem_snapshot(f"{letter}_post")
    cell_json["status"] = fwd_status
    if err_text:
        cell_json["error"] = err_text
        row["error"] = err_text[:500]

    write_json(out_dir / "raw" / f"{letter}_cell.json", cell_json)

    # cleanup
    del state, frame, render_alpha, result
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return row, cell_pre


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", default=",".join(CELLS.keys()))
    ap.add_argument("--skip-backward", action="store_true")
    ap.add_argument("--no-oracle", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=MIN_FREE_GB)
    args = ap.parse_args()
    ensure_out_dirs()

    add_project_to_syspath()
    from gsplat.experimental.render.functional import rasterize_gaussian_higs_frozen

    const = read_common_constants()
    alpha, extend = const.get("alpha_threshold"), const.get("gaussian_extend")
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    print(f"constants: alpha={alpha} extend={extend}", flush=True)
    print(json.dumps(gpu_mem_snapshot("preflight"), indent=2), flush=True)

    ckpt, ck_info = load_checkpoint()
    n_full = int(ckpt["xyz"].shape[0])
    indices = build_580k_indices(n_full)
    try:
        cams540, cams_path, cams_sha = load_real_cameras(W540, H540, device="cpu")
    except Exception as e:
        cams540, cams_path, cams_sha = [], None, None
        print(f"CAMERAS_JSON_UNAVAILABLE: {e!r}; synthetic cells remain executable", flush=True)
    print(f"checkpoint N={n_full:,} sha={ck_info['sha256']}", flush=True)
    print(f"subset n={indices.size} cameras={cams_path}", flush=True)

    preamble = {
        "checkpoint": ck_info, "subset_n": int(indices.size),
        "subset_seed": EXPECTED["subset_seed"], "cameras_json": cams_path,
        "cameras_json_sha256": cams_sha, "constants": const,
        "cells": CELLS, "min_free_gb": args.min_free_gb,
    }
    write_json(OUT / "raw" / "R17_P3_PREAMBLE.json", preamble)

    from common import append_csv
    csv_path = OUT / "R17_580K_CONTROLLED_COMPARISON.csv"
    cov_path = OUT / "R17_INTERSECTION_COVERAGE.csv"
    # Re-running a step must not append a second generation of rows.
    for prior in (csv_path, cov_path):
        if prior.exists():
            backup = OUT / "logs" / f"{prior.name}.prior.{time.time_ns()}"
            prior.rename(backup)
            print(f"[archive] previous output -> {backup}", flush=True)
    for letter in args.cells.split(","):
        letter = letter.strip()
        if letter not in CELLS:
            print(f"unknown cell {letter}", flush=True)
            continue
        print(f"\n===== CELL {letter}: {CELLS[letter]['desc']} =====", flush=True)
        t0 = time.time()
        row, pre = run_cell(letter, CELLS[letter], ckpt, cams540, indices,
                            rasterize_gaussian_higs_frozen, alpha, extend, OUT,
                            skip_backward=args.skip_backward, do_oracle=not args.no_oracle,
                            min_free_gb=args.min_free_gb)
        print(json.dumps({k: row[k] for k in
                          ("cell", "fwd_status", "bwd_status", "sum_counts_first_call",
                           "n_isects_metadata", "coverage_frac", "peak_vram_gb", "oracle_mismatch_pairs",
                           "oracle_verdict", "r15_gate")}, indent=2, default=str), flush=True)
        print(f"cell {letter} done in {time.time()-t0:.1f}s", flush=True)
        append_csv(csv_path, row, CSV_FIELDS)
        append_csv(cov_path, {
            "cell": letter, "camera_mode": row["camera_mode"],
            "tiles_hit": row["coverage_tiles_hit"], "coverage_frac": row["coverage_frac"],
            "n_images": row["n_images_first_call"], "sum_counts": row["sum_counts_first_call"],
            "count_max": row["count_max"], "count_p99": row["count_p99"],
        }, ["cell", "camera_mode", "tiles_hit", "coverage_frac", "n_images",
            "sum_counts", "count_max", "count_p99"])
    print("P3 COMPLETE", flush=True)


if __name__ == "__main__":
    main()
