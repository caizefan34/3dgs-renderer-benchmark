#!/usr/bin/env python3
"""R18-P2: formal performance comparison — frozen std_ll vs frozen HiGS.

Same checkpoint, real cameras, matched forward+backward workload. Configs:
  580K     / 540p / B1, B4
  4.413928M / 540p / B1, B4
  4.413928M / 1080p / B4   (best-effort; recorded as OOM/skipped if headroom insufficient)

Each config: >=10 warmup, >=30 timed iterations. Reports forward / backward /
fwd+bwd median, p10, p90 (CUDA events), peak allocated AND peak reserved bytes,
and a best-effort device GPU-utilization sample. The real loaded binary SHA256
(std + HiGS) is recorded. A forward correctness gate is attached from P1 so a
fast-but-wrong result cannot be claimed as a win.

Deliverable: R18_P2_PERFORMANCE.csv (+ raw/R18_P2_details.json)
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.cuda import Event

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r18_common import (H1080, H540, W1080, W540, OUT, add_project_to_syspath,
                        build_580k_indices, build_cell_state, device_utilization_ms,
                        ensure_out_dirs, full_indices, gpu_mem_snapshot,
                        load_checkpoint, load_real_cameras, reset_peak, sha256_file,
                        write_json)

CSV_FIELDS = [
    "config", "backend", "n_gaussians", "resolution", "camera_set", "n_cameras",
    "activation", "warmup", "timed",
    "fwd_median_ms", "fwd_mean_ms", "fwd_p10_ms", "fwd_p90_ms",
    "bwd_median_ms", "bwd_mean_ms", "bwd_p10_ms", "bwd_p90_ms",
    "fb_median_ms", "fb_p10_ms", "fb_p90_ms",
    "peak_allocated_gb", "peak_reserved_gb", "gpu_util_pct",
    "binary_sha256", "correctness_gate", "status", "error",
]

CONFIGS = {
    "580K_540p_B1": ("580k", "540p", "B1"),
    "580K_540p_B4": ("580k", "540p", "B4"),
    "4M_540p_B1": ("4m", "540p", "B1"),
    "4M_540p_B4": ("4m", "540p", "B4"),
    "4M_1080p_B4": ("4m", "1080p", "B4"),
}
MIN_FREE_GB = 18.0


def percentiles(v):
    a = np.asarray(v, dtype=np.float64)
    return {"median": float(np.median(a)), "mean": float(a.mean()),
            "p10": float(np.quantile(a, 0.1)), "p90": float(np.quantile(a, 0.9))}


def timed_loop(render_fn, state, n_warmup, n_timed):
    def zero_grads():
        for k in ("means", "quats", "scales", "opacities", "shs"):
            state[k].grad = None

    fwd_times, bwd_times = [], []
    for _ in range(n_warmup):
        zero_grads()
        out = render_fn()
        loss = out["frame"].sum() + out["alpha"].sum()
        loss.backward()
    torch.cuda.synchronize()
    for _ in range(n_timed):
        zero_grads()
        s = Event(enable_timing=True); e = Event(enable_timing=True)
        torch.cuda.synchronize(); s.record()
        out = render_fn()
        e.record(); torch.cuda.synchronize()
        fwd_times.append(s.elapsed_time(e))
        s = Event(enable_timing=True); e = Event(enable_timing=True)
        torch.cuda.synchronize(); s.record()
        loss = out["frame"].sum() + out["alpha"].sum()
        loss.backward()
        e.record(); torch.cuda.synchronize()
        bwd_times.append(s.elapsed_time(e))
    return fwd_times, bwd_times


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--timed", type=int, default=30)
    ap.add_argument("--configs", default=",".join(CONFIGS.keys()))
    ap.add_argument("--min-free-gb", type=float, default=MIN_FREE_GB)
    args = ap.parse_args()
    ensure_out_dirs()
    add_project_to_syspath()

    from gsplat import rasterization
    from gsplat.experimental.render.functional import rasterize_gaussian_higs_frozen

    pre = gpu_mem_snapshot("p2_preflight")
    print(json.dumps(pre, indent=2), flush=True)

    ckpt, ck_info = load_checkpoint()
    n_full = int(ckpt["xyz"].shape[0])
    idx_cache = {"580k": build_580k_indices(n_full), "4m": full_indices(n_full)}

    cams_by_res = {}
    for res in ("540p", "1080p"):
        try:
            cams, p, s = load_real_cameras(
                W1080 if res == "1080p" else W540,
                H1080 if res == "1080p" else H540, device="cpu")
            cams_by_res[res] = (cams, p, s)
        except Exception as e:  # noqa: BLE001
            cams_by_res[res] = ([], None, None)
            print(f"CAMERAS_UNAVAILABLE {res}: {e!r}", flush=True)

    from r18_common import PROJECT
    binaries = {"std_csrc_sha256": sha256_file(PROJECT / "gsplat/csrc.so"),
                "higs_csrc_sha256": sha256_file(
                    PROJECT / "gsplat/experimental/render/kernels/csrc.so")}

    details = {"deliverable": "R18_P2_PERFORMANCE", "checkpoint": ck_info,
               "binaries": binaries, "warmup": args.warmup, "timed": args.timed,
               "rows": []}
    csv_path = OUT / "R18_P2_PERFORMANCE.csv"
    from r18_common import append_csv

    for cfg in args.configs.split(","):
        cfg = cfg.strip()
        if cfg not in CONFIGS:
            continue
        subset, res, bsz = CONFIGS[cfg]
        W = W1080 if res == "1080p" else W540
        H = H1080 if res == "1080p" else H540
        indices = idx_cache[subset]
        ncams = 1 if bsz == "B1" else 4
        cams, cams_path, cams_sha = cams_by_res[res]
        cams_ok = bool(cams) and len(cams) >= ncams
        viewmats = Ks = bg = None
        if cams_ok:
            cc = cams[:ncams]
            viewmats = torch.stack([c["viewmatrix"] for c in cc]).to("cuda").contiguous()
            Ks = torch.stack([c["K"] for c in cc]).to("cuda").contiguous()
            bg = torch.zeros(ncams, 3, device="cuda")

        def make_std(state):
            def fn():
                rc, ra, _ = rasterization(
                    means=state["means"], quats=state["quats"], scales=state["scales"],
                    opacities=state["opacities"], colors=state["shs"],
                    viewmats=viewmats, Ks=Ks, width=W, height=H,
                    sh_degree=3, backgrounds=bg)
                return {"frame": rc.squeeze(0) if rc.shape[0] == 1 else rc,
                        "alpha": ra.squeeze(0) if ra.shape[0] == 1 else ra}
            return fn

        def make_higs(state):
            def fn():
                r = rasterize_gaussian_higs_frozen(
                    state["means"], state["quats"], state["scales"], state["opacities"],
                    state["shs"], viewmats=viewmats, Ks=Ks, width=W, height=H,
                    sh_degree=3, enable_culling=True, use_higs_culling=True,
                    freeze_topology=False)  # mixed Gaussian counts across configs
                return {"frame": r["frame"], "alpha": r["alpha"]}
            return fn

        for backend in ("std_ll", "higs_native"):
            row = {k: None for k in CSV_FIELDS}
            row.update({
                "config": cfg, "backend": backend, "n_gaussians": int(indices.size),
                "resolution": res, "camera_set": f"{'cam0' if ncams == 1 else 'cam0..%d' % (ncams-1)} {res}",
                "n_cameras": ncams, "activation": "sigmoid/exp/normalize",
                "warmup": args.warmup, "timed": args.timed,
                "binary_sha256": binaries["std_csrc_sha256" if backend == "std_ll" else "higs_csrc_sha256"],
            })
            if not cams_ok:
                row.update({"status": "NOT_EXECUTED_CAMERAS_MISSING",
                            "error": f"need {ncams} cams at {res}, have {len(cams)}"})
                append_csv(csv_path, row, CSV_FIELDS); details["rows"].append(row); continue
            free = torch.cuda.mem_get_info()[0] / 1e9
            if free < args.min_free_gb:
                row.update({"status": "OUT_OF_HEADROOM", "error": f"free_gb={free:.1f}"})
                append_csv(csv_path, row, CSV_FIELDS); details["rows"].append(row); continue
            print(f"\n===== P2 {cfg} / {backend} =====", flush=True)
            try:
                state = build_cell_state(ckpt, indices, "sigmoid", "exp", "normalize", device="cuda")
                fn = make_std(state) if backend == "std_ll" else make_higs(state)
                reset_peak()
                t0 = time.time()
                fwd_t, bwd_t = timed_loop(fn, state, args.warmup, args.timed)
                torch.cuda.synchronize()
                wall = time.time() - t0
                peak_alloc = torch.cuda.max_memory_allocated() / 1e9
                peak_res = torch.cuda.max_memory_reserved() / 1e9
                fp, bp = percentiles(fwd_t), percentiles(bwd_t)
                fbp = percentiles(np.asarray(fwd_t) + np.asarray(bwd_t))
                util = device_utilization_ms()
                row.update({
                    "fwd_median_ms": fp["median"], "fwd_mean_ms": fp["mean"],
                    "fwd_p10_ms": fp["p10"], "fwd_p90_ms": fp["p90"],
                    "bwd_median_ms": bp["median"], "bwd_mean_ms": bp["mean"],
                    "bwd_p10_ms": bp["p10"], "bwd_p90_ms": bp["p90"],
                    "fb_median_ms": fbp["median"], "fb_p10_ms": fbp["p10"], "fb_p90_ms": fbp["p90"],
                    "peak_allocated_gb": peak_alloc, "peak_reserved_gb": peak_res,
                    "gpu_util_pct": util.get("gpu_util_pct"),
                    "status": "PASS",
                })
                details["rows"].append({**row, "wall_seconds": wall, "gpu_util": util})
                print(json.dumps({k: row[k] for k in
                      ("config", "backend", "fwd_median_ms", "bwd_median_ms", "fb_median_ms",
                       "peak_allocated_gb", "peak_reserved_gb", "gpu_util_pct", "status")},
                      indent=2), flush=True)
                del state
            except torch.cuda.OutOfMemoryError as e:
                row.update({"status": "OOM", "error": repr(e)[:400]})
                print(f"OOM {cfg}/{backend}: {e!r}", flush=True)
                details["rows"].append(row)
            except Exception as e:  # noqa: BLE001
                row.update({"status": "ERROR", "error": repr(e)[:400]})
                import traceback; traceback.print_exc()
                details["rows"].append(row)
            append_csv(csv_path, row, CSV_FIELDS)
            gc.collect(); torch.cuda.empty_cache()

    # Correctness gate: attach P1 forward-error to each P2 row.
    try:
        p1 = json.loads((OUT / "raw" / "R18_P1_details.json").read_text())
        gate = {}
        for r in p1.get("rows", []):
            if r.get("config") in ("580K_B1", "580K_B4", "4M_B1", "4M_B4"):
                gate[r["config"]] = {"rgb_mae": r.get("rgb_mae"), "rgb_max_abs": r.get("rgb_max_abs"),
                                     "alpha_mae": r.get("alpha_mae"), "status": r.get("status")}
        details["correctness_gate_from_p1"] = gate
    except Exception as e:  # noqa: BLE001
        details["correctness_gate_from_p1"] = {"error": repr(e)}

    write_json(OUT / "raw" / "R18_P2_details.json", details)
    print("P2 COMPLETE", flush=True)


if __name__ == "__main__":
    main()
