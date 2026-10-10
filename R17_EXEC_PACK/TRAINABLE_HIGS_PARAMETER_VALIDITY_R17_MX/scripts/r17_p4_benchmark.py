#!/usr/bin/env python3
"""R17-P4: matched benchmark — frozen std_ll vs frozen HiGS, identical activated state.

Protocol (both backends, same tensors, same cameras, same batch):
  state  : 580K R8/R15 subset, ACTIVATED per the training contract
           (opacity=sigmoid, scales=exp, quats=normalize)
  camera : real training cameras, 540p (960x540) — B1 = cam0; B4 = cams 0..3
  call   : B1 = one call [1,4,4]/[1,3,3]; B4 = ONE call [4,4,4]/[4,3,3] (R8 style)
  loop   : warmup, then timed fwd and timed bwd (loss=f.sum()+a.sum(), grads zeroed)

Deliverable: R17_VALIDATED_BENCHMARK.csv (+ raw/R17_P4_details.json)

No R3 speedup claim is reused; every number here is measured under this protocol.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
from torch.cuda import Event

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (H540, OUT, W540, add_project_to_syspath, append_csv,
                    build_580k_indices, build_cell_state, ensure_out_dirs,
                    gpu_mem_snapshot, load_checkpoint, load_real_cameras,
                    nan_report, sha256_file, write_json)

CSV_FIELDS = [
    "protocol", "backend", "camera_set", "camera_names", "n_gaussians", "activation",
    "fwd_median_ms", "fwd_mean_ms", "fwd_p10_ms", "fwd_p90_ms",
    "bwd_median_ms", "bwd_mean_ms", "bwd_p10_ms", "bwd_p90_ms",
    "fb_median_ms", "peak_vram_gb", "n_isects_instrumented", "n_qualifying_calls",
    "status", "nan_frame", "nan_alpha", "nan_grads", "binary_sha256", "error",
]

MIN_FREE_GB = 20.0


def percentiles(v):
    import numpy as np
    a = np.asarray(v, dtype=np.float64)
    return {
        "median": float(np.median(a)), "mean": float(a.mean()),
        "p10": float(np.quantile(a, 0.1)), "p90": float(np.quantile(a, 0.9)),
    }


def timed_loop(render_fn, state, n_warmup, n_timed, skip_backward=False):
    def zero_grads():
        for k in ("means", "quats", "scales", "opacities", "shs"):
            state[k].grad = None

    fwd_times, bwd_times = [], []
    for i in range(n_warmup):
        zero_grads()
        out = render_fn()
        if not skip_backward:
            loss = out["frame"].sum() + out["alpha"].sum()
            loss.backward()
    torch.cuda.synchronize()
    for i in range(n_timed):
        zero_grads()
        s = Event(enable_timing=True); e = Event(enable_timing=True)
        torch.cuda.synchronize()
        s.record()
        out = render_fn()
        e.record()
        torch.cuda.synchronize()
        fwd_times.append(s.elapsed_time(e))
        if not skip_backward:
            s = Event(enable_timing=True); e = Event(enable_timing=True)
            torch.cuda.synchronize()
            s.record()
            loss = out["frame"].sum() + out["alpha"].sum()
            loss.backward()
            e.record()
            torch.cuda.synchronize()
            bwd_times.append(s.elapsed_time(e))
        frame, alpha = out["frame"], out["alpha"]
    # final iteration gradients retained deliberately for the NaN check
    nan = nan_report(("frame", frame), ("alpha", alpha),
                     *[(f"grad_{k}", state[k].grad) for k in
                       ("means", "quats", "scales", "opacities", "shs")])
    return fwd_times, bwd_times, nan


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--timed", type=int, default=30)
    ap.add_argument("--protocols", default="B1,B4")
    ap.add_argument("--skip-backward", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=MIN_FREE_GB)
    args = ap.parse_args()
    ensure_out_dirs()

    add_project_to_syspath()
    from gsplat import rasterization
    from gsplat.experimental.render.functional import rasterize_gaussian_higs_frozen

    pre = gpu_mem_snapshot("p4_preflight")
    print(json.dumps(pre, indent=2), flush=True)

    ckpt, ck_info = load_checkpoint()
    n_full = int(ckpt["xyz"].shape[0])
    indices = build_580k_indices(n_full)
    cams540, cams_path, cams_sha = load_real_cameras(W540, H540, device="cpu")

    state = build_cell_state(ckpt, indices, "sigmoid", "exp", "normalize", device="cuda")

    proj = Path(__import__("common").PROJECT)
    binaries = {}
    for label, rel in (("higs", "gsplat/experimental/render/kernels/csrc.so"),
                       ("std", "gsplat/csrc.so")):
        binaries[f"{label}_csrc_sha256"] = sha256_file(proj / rel)

    details = {"checkpoint": ck_info, "cameras_json": cams_path,
               "cameras_json_sha256": cams_sha, "binaries": binaries,
               "warmup": args.warmup, "timed": args.timed,
               "state": "580K subset ACTIVATED (sigmoid/exp/normalize)",
               "rows": []}

    def make_higs_fn(viewmats, Ks):
        def fn():
            r = rasterize_gaussian_higs_frozen(
                state["means"], state["quats"], state["scales"], state["opacities"],
                state["shs"], viewmats=viewmats, Ks=Ks, width=W540, height=H540,
                sh_degree=3, enable_culling=True, use_higs_culling=True)
            return {"frame": r["frame"], "alpha": r["alpha"], "raw": r}
        return fn

    def make_std_fn(viewmats, Ks):
        bg = torch.zeros(viewmats.shape[0], 3, device="cuda")

        def fn():
            rc, ra, _ = rasterization(
                means=state["means"], quats=state["quats"], scales=state["scales"],
                opacities=state["opacities"], colors=state["shs"],
                viewmats=viewmats, Ks=Ks, width=W540, height=H540,
                sh_degree=3, backgrounds=bg)
            return {"frame": rc, "alpha": ra}
        return fn

    for protocol in args.protocols.split(","):
        protocol = protocol.strip()
        ncams = 1 if protocol == "B1" else 4
        cams = cams540[:ncams]
        viewmats = torch.stack([c["viewmatrix"] for c in cams]).to("cuda").contiguous()
        Ks = torch.stack([c["K"] for c in cams]).to("cuda").contiguous()
        cam_names = [c["img_name"] for c in cams]

        for backend in ("std_ll", "higs_native"):
            row = {k: None for k in CSV_FIELDS}
            row.update({
                "protocol": protocol, "backend": backend,
                "camera_set": f"{'cam0' if ncams == 1 else 'cam0..3'} real 540p",
                "camera_names": ";".join(cam_names), "n_gaussians": int(indices.size),
                "activation": "sigmoid/exp/normalize", "error": "",
                "binary_sha256": binaries.get("higs_csrc_sha256" if backend == "higs_native"
                                              else "std_csrc_sha256"),
            })
            free = torch.cuda.mem_get_info()[0] / 1e9
            if free < args.min_free_gb:
                row.update({"status": "OUT_OF_HEADROOM", "error": f"free_gb={free:.1f}"})
                append_csv(OUT / "R17_VALIDATED_BENCHMARK.csv", row, CSV_FIELDS)
                details["rows"].append(row)
                continue
            print(f"\n===== {protocol} / {backend} ({ncams} cam) =====", flush=True)
            try:
                # instrumented single call for n_isects (first call only)
                from capture_intersections import IsectRecorder, install, uninstall
                rec = IsectRecorder(None, mode="record_only", tag=f"p4_{protocol}_{backend}")
                orig = install(rec)
                fn = make_higs_fn(viewmats, Ks) if backend == "higs_native" else make_std_fn(viewmats, Ks)
                try:
                    out = fn()
                    torch.cuda.synchronize()
                finally:
                    uninstall(orig)
                row["n_qualifying_calls"] = rec.summary().get("n_qualifying_calls")
                fcr = rec.summary().get("first_call_records") or {}
                row["n_isects_instrumented"] = fcr.get("sum_counts")

                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                fwd_t, bwd_t, nan = timed_loop(fn, state, args.warmup, args.timed,
                                               skip_backward=args.skip_backward)
                wall = time.time() - t0
                fp, bp = percentiles(fwd_t), (percentiles(bwd_t) if bwd_t else {})
                row.update({
                    "fwd_median_ms": fp["median"], "fwd_mean_ms": fp["mean"],
                    "fwd_p10_ms": fp["p10"], "fwd_p90_ms": fp["p90"],
                    "bwd_median_ms": bp.get("median"), "bwd_mean_ms": bp.get("mean"),
                    "bwd_p10_ms": bp.get("p10"), "bwd_p90_ms": bp.get("p90"),
                    "fb_median_ms": (fp["median"] + bp.get("median", 0.0)) if bwd_t else None,
                    "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
                    "status": "PASS",
                    "nan_frame": nan["frame"].get("has_nan") if isinstance(nan["frame"], dict) else None,
                    "nan_alpha": nan["alpha"].get("has_nan") if isinstance(nan["alpha"], dict) else None,
                    "nan_grads": any(v.get("has_nan") or v.get("has_inf") for k, v in nan.items()
                                     if k.startswith("grad_") and isinstance(v, dict)),
                })
                details["rows"].append({**row, "wall_seconds": wall, "nan": nan})
                print(json.dumps({k: row[k] for k in
                                  ("protocol", "backend", "fwd_median_ms", "bwd_median_ms",
                                   "fb_median_ms", "peak_vram_gb", "n_isects_instrumented")},
                                 indent=2), flush=True)
            except torch.cuda.OutOfMemoryError as e:
                row.update({"status": "OOM", "error": repr(e)[:500]})
                print(f"OOM: {e!r}", flush=True)
            except Exception as e:  # noqa: BLE001
                row.update({"status": "ERROR", "error": repr(e)[:500]})
                import traceback
                traceback.print_exc()
            append_csv(OUT / "R17_VALIDATED_BENCHMARK.csv", row, CSV_FIELDS)
            gc.collect()
            torch.cuda.empty_cache()

    write_json(OUT / "raw" / "R17_P4_details.json", details)
    print("P4 COMPLETE", flush=True)


if __name__ == "__main__":
    main()
