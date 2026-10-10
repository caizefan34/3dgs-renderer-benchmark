#!/usr/bin/env python3
"""R17-P5: 4.41M safety-gated diagnosis (count-only first, full fwd/bwd only if memory-safe).

Full bicycle checkpoint (N = 4,413,928), 540p, four scenarios:

  raw_identity_b1    raw params + identity cam    -> ESTIMATE ONLY (never runs the full call)
  raw_identity_b4    raw params + 4 identity cams -> ESTIMATE ONLY (the R8 OOM scenario; R8 asked 111.28 GiB)
  activated_real_b1  contract params + real cam0  -> estimate, then full fwd+bwd iff the gate passes
  activated_real_b4  contract params + real cams  -> estimate, then full fwd+bwd iff the gate passes

Evidence recorded per scenario: subset count stats, projected isect count and
buffer-byte projection, live allocator/VRAM state at the gate, gate decision,
and (when executed) actual n_isects, peak VRAM, fwd/bwd ms, NaN checks.
If the full call OOMs anyway, the raw PyTorch OOM message and allocator summary
are captured verbatim. No estimated number is ever reported as measured.

Output: R17_4P4M_SAFETY_GATE.json (+ raw/R17_P5_<scenario>.json, raw/<scenario>_SAFETY_GATE.json)
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path

import torch
from torch.cuda import Event

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (H540, OUT, W540, add_project_to_syspath, build_cell_state,
                    ensure_out_dirs, gpu_mem_snapshot, load_checkpoint,
                    load_real_cameras, nan_report, read_common_constants,
                    write_json)

SCENARIOS = {
    "raw_identity_b1": dict(params="raw", camera="identity", batch=1, rec_mode="estimate_only",
                            desc="raw params + identity camera (R8 protocol core)"),
    "raw_identity_b4": dict(params="raw", camera="identity", batch=4, rec_mode="estimate_only",
                            desc="raw params + 4 identity cams in ONE call (R8 4.4M OOM scenario)"),
    "activated_real_b1": dict(params="activated", camera="real", batch=1, rec_mode="gated_full",
                              desc="contract-activated params + real cam0"),
    "activated_real_b4": dict(params="activated", camera="real", batch=4, rec_mode="gated_full",
                              desc="contract-activated params + real cam0..3 (ONE call)"),
}

R8_REFERENCE = {
    "source": "TRAINABLE_HIGS_RUNTIME_RECOVERY_R8_MX/R8_FINAL_REPORT.md",
    "scenario": "4.4M raw+identity B4 (960x540) -> CUDA OOM, allocation request 111.28 GiB",
    "reported_requested_bytes": 111.28 * (1024 ** 3),
    "gpu_total_bytes": 39.49 * (1024 ** 3),
    "gpu_free_bytes_at_failure": 13.50 * (1024 ** 3),
}

MIN_FREE_GB = 10.0
GATE_MARGIN_BYTES = 6 << 30
EXPECTED_N = 4_413_928


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenarios", default=",".join(SCENARIOS.keys()))
    ap.add_argument("--estimate-samples", type=int, default=20000)
    ap.add_argument("--min-free-gb", type=float, default=MIN_FREE_GB)
    args = ap.parse_args()
    ensure_out_dirs()

    add_project_to_syspath()
    from gsplat.experimental.render.functional import rasterize_gaussian_higs_frozen
    from capture_intersections import IsectRecorder, install, uninstall, SafeAbort

    const = read_common_constants()
    print(json.dumps(gpu_mem_snapshot("p5_preflight"), indent=2), flush=True)
    ckpt, ck_info = load_checkpoint()
    n_full = int(ckpt["xyz"].shape[0])
    if n_full != EXPECTED_N:
        print(f"WARNING: N={n_full} differs from expected {EXPECTED_N}", flush=True)
    cameras_error = None
    try:
        cams540, cams_path, cams_sha = load_real_cameras(W540, H540, device="cpu")
    except Exception as e:
        cams540, cams_path, cams_sha = [], None, None
        cameras_error = repr(e)
        print(f"CAMERAS_JSON_UNAVAILABLE: {cameras_error}; raw identity diagnosis remains available", flush=True)

    report = {
        "deliverable": "R17_4P4M_SAFETY_GATE.json",
        "checkpoint": ck_info, "n_gaussians": n_full,
        "cameras_json": cams_path, "cameras_json_sha256": cams_sha,
        "constants": const, "policy": {
            "raw_scenarios": "ESTIMATE ONLY — full call is never executed for raw params",
            "activated_scenarios": "estimate first; full fwd+bwd only if the byte projection fits free VRAM",
            "margin_bytes": GATE_MARGIN_BYTES,
            "estimate_samples": args.estimate_samples,
        },
        "r8_reference_claims": R8_REFERENCE,
        "scenarios": {},
        "status": "NOT_EXECUTED",
    }

    all_indices = torch.arange(n_full, dtype=torch.long)
    for name in args.scenarios.split(","):
        name = name.strip()
        if name not in SCENARIOS:
            continue
        cfg = SCENARIOS[name]
        scen = {"config": cfg, "status": "NOT_EXECUTED"}
        pre = gpu_mem_snapshot(f"{name}_pre")
        scen["pre_mem"] = pre
        if cfg["camera"] == "real" and len(cams540) < cfg["batch"]:
            scen["status"] = "NOT_EXECUTED_CAMERAS_JSON_MISSING"
            scen["error"] = cameras_error or f"Need {cfg['batch']} cameras, found {len(cams540)}"
            report["scenarios"][name] = scen
            write_json(OUT / "raw" / f"R17_P5_{name}.json", scen)
            continue
        if pre.get("free_gb", 0) < args.min_free_gb:
            scen["status"] = "OUT_OF_HEADROOM"
            scen["error"] = f"free_gb={pre.get('free_gb')} < {args.min_free_gb}"
            report["scenarios"][name] = scen
            write_json(OUT / "raw" / f"R17_P5_{name}.json", scen)
            continue
        print(f"\n===== P5 SCENARIO {name}: {cfg['desc']} =====", flush=True)
        t0 = time.time()
        state = None
        try:
            if cfg["params"] == "raw":
                state = build_cell_state(ckpt, all_indices, "raw", "raw", "raw", device="cuda")
            else:
                state = build_cell_state(ckpt, all_indices, "sigmoid", "exp", "normalize",
                                         device="cuda")
            scen["state_allocated_gb"] = torch.cuda.memory_allocated() / 1e9

            if cfg["camera"] == "identity":
                K = torch.tensor([[W540, 0, W540 / 2], [0, H540, H540 / 2], [0, 0, 1]],
                                 dtype=torch.float32, device="cuda").unsqueeze(0)
                viewmat = torch.eye(4, device="cuda", dtype=torch.float32).unsqueeze(0)
                if cfg["batch"] == 4:
                    K = K.repeat(4, 1, 1).contiguous()
                    viewmat = viewmat.repeat(4, 1, 1).contiguous()
            else:
                cams = cams540[:cfg["batch"]]
                K = torch.stack([c["K"] for c in cams]).to("cuda").contiguous()
                viewmat = torch.stack([c["viewmatrix"] for c in cams]).to("cuda").contiguous()

            rec = IsectRecorder(None, mode=cfg["rec_mode"], tag=name,
                                save_dir=OUT / "raw",
                                estimate_samples=args.estimate_samples,
                                memory_margin_bytes=GATE_MARGIN_BYTES)
            orig = install(rec)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            fwd_status, bwd_status, fwd_ms, bwd_ms = "NOT_RUN", "NOT_RUN", None, None
            nan = None
            try:
                s = Event(enable_timing=True); e = Event(enable_timing=True)
                torch.cuda.synchronize(); s.record()
                result = rasterize_gaussian_higs_frozen(
                    state["means"], state["quats"], state["scales"], state["opacities"],
                    state["shs"], viewmats=viewmat, Ks=K, width=W540, height=H540,
                    sh_degree=3, enable_culling=True, use_higs_culling=True)
                e.record(); torch.cuda.synchronize()
                fwd_ms = s.elapsed_time(e)
                fwd_status = "PASS"
                frame = result.get("frame"); alpha = result.get("alpha")
                meta = result.get("metadata") or {}
                if isinstance(meta, dict):
                    scen["metadata"] = {k: (v if isinstance(v, (int, float, str, bool, type(None)))
                                            else str(v)[:200]) for k, v in meta.items()}
                s = Event(enable_timing=True); e = Event(enable_timing=True)
                torch.cuda.synchronize(); s.record()
                loss = frame.sum() + alpha.sum()
                loss.backward()
                e.record(); torch.cuda.synchronize()
                bwd_ms = s.elapsed_time(e)
                bwd_status = "PASS"
                nan = nan_report(("frame", frame), ("alpha", alpha),
                                 *[(f"grad_{k}", state[k].grad) for k in
                                   ("means", "quats", "scales", "opacities", "shs")])
            except SafeAbort as ab:
                fwd_status = "SAFE_ABORT_" + str(ab.info.get("mode"))
                scen["abort_info"] = ab.info
            except torch.cuda.OutOfMemoryError as oom:
                fwd_status = "OOM" if fwd_status == "NOT_RUN" else "OOM_DURING_BWD"
                bwd_status = "OOM" if fwd_status == "OOM_DURING_BWD" else bwd_status
                scen["oom_message"] = str(oom)
                scen["allocator_summary"] = torch.cuda.memory_summary(abbreviated=True)
            finally:
                uninstall(orig)

            scen.update({
                "fwd_status": fwd_status, "bwd_status": bwd_status,
                "fwd_ms": fwd_ms, "bwd_ms": bwd_ms,
                "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
                "post_mem": gpu_mem_snapshot(f"{name}_post"),
                "recorder": rec.summary(),
                "estimate": rec.estimate,
                "nan": nan,
                "seconds": time.time() - t0,
            })
            if rec.estimate:
                est = rec.estimate["projected_bytes_documented_conservative"]
                scen["vs_r8_reported_request"] = {
                    "r8_requested_bytes": R8_REFERENCE["reported_requested_bytes"],
                    "r17_projected_bytes_conservative": est,
                    "ratio_r17_over_r8": est / R8_REFERENCE["reported_requested_bytes"],
                }
            if fwd_status.startswith("SAFE_ABORT") or fwd_status == "PASS":
                scen["status"] = ("FULL_EXECUTED" if fwd_status == "PASS"
                                  else "ESTIMATED_GATE_SKIPPED_FULL")
            elif fwd_status.startswith("OOM"):
                scen["status"] = "OOM_DURING_EXECUTED_FULL"
            else:
                scen["status"] = "ERROR"
        except Exception as e:  # noqa: BLE001
            scen["status"] = "ERROR"
            scen["error"] = repr(e)
            traceback.print_exc()
        finally:
            if state is not None:
                del state
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        report["scenarios"][name] = scen
        write_json(OUT / "raw" / f"R17_P5_{name}.json", scen)
        print(json.dumps({k: scen.get(k) for k in
                          ("status", "fwd_status", "bwd_status", "fwd_ms", "bwd_ms",
                           "peak_vram_gb")}, indent=2, default=str), flush=True)

    statuses = {k: v.get("status") for k, v in report["scenarios"].items()}
    report["status"] = ("EXECUTED" if statuses and all(s in
                        ("FULL_EXECUTED", "ESTIMATED_GATE_SKIPPED_FULL", "OUT_OF_HEADROOM",
                         "OOM_DURING_EXECUTED_FULL") for s in statuses.values())
                        else "PARTIAL")
    write_json(OUT / "R17_4P4M_SAFETY_GATE.json", report)
    print("P5 COMPLETE", flush=True)


if __name__ == "__main__":
    main()
