#!/usr/bin/env python3
"""P16-RB: normalized P12_C0 record (runs/P12_C0/<scene>/P16_RUN.json).

P12's protocol trains on ALL cameras and evaluates all of them; the external 2026
methods all use hold-8. Per the prereg split-resolution rule the common comparison is
hold-8, so this record's metrics_common come from the frozen-checkpoint render on the
hold-8 cameras (p16_p12_render_common.py), scored by the same common evaluator used for
every external method. The all-camera archived numbers stay in quality.json.
"""
import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

COMMON_W, COMMON_H = 1920, 1080
PIXEL_TOL = 0.02


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def parse_triad_log(p: Path, scene: str):
    """Return (rc, T0, T1, delta_s) for the scene's line in the triad driver log."""
    txt = p.read_text(errors="replace")
    m = re.search(rf"^{scene} rc=(\d+) T0=(\S+) T1=(\S+)\s*$", txt, re.M)
    if not m:
        return None, None, None, None
    rc, t0, t1 = int(m.group(1)), m.group(2), m.group(3)
    d = (datetime.fromisoformat(t1) - datetime.fromisoformat(t0)).total_seconds()
    return rc, t0, t1, d


def snap_gpu_uuid(p: Path):
    """gpu_before.csv rows are index,uuid,name,memory.used,memory.total (unfiltered, 5 cols)."""
    for line in p.read_text().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 2:
            return parts[1]
    return None


def snap_procs(p: Path, uuid):
    """procs csv rows are gpu_uuid,pid,used_memory for ALL GPUs; keep the target GPU only."""
    out = []
    for line in p.read_text().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 3 and parts[0] == uuid:
            out.append(tuple(parts))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--rundir", required=True)
    ap.add_argument("--triad-log", required=True)
    ap.add_argument("--metrics-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source-commit", required=True)
    ap.add_argument("--log-prefix", required=True, help="triad log prefix, e.g. logs/P12_C0_room")
    ap.add_argument("--trainer", default="/root/epic05_p12/publication_trainer_p12.py")
    args = ap.parse_args()

    rd = Path(args.rundir)
    timing = json.loads((rd / "timing.json").read_text())
    memory = json.loads((rd / "memory.json").read_text())
    quality = json.loads((rd / "quality.json").read_text())
    status = json.loads((rd / "final_status.json").read_text())
    met = json.loads(Path(args.metrics_json).read_text())

    n_iter = int(timing.get("_n_iterations", quality.get("step", 30000)))
    loop_only_s = timing["mean_iter_ms"] * n_iter / 1000.0

    rc, t0, t1, delta = parse_triad_log(Path(args.triad_log), args.scene)
    gpu_before_csv = Path(f"{args.log_prefix}.gpu_before.csv")
    gpu_after_csv = Path(f"{args.log_prefix}.gpu_after.csv")
    procs_before_csv = Path(f"{args.log_prefix}.procs_before.csv")
    procs_after_csv = Path(f"{args.log_prefix}.procs_after.csv")
    uuid_b = snap_gpu_uuid(gpu_before_csv) if gpu_before_csv.exists() else None
    uuid_a = snap_gpu_uuid(gpu_after_csv) if gpu_after_csv.exists() else None
    before = snap_procs(procs_before_csv, uuid_b) if procs_before_csv.exists() else None
    after = snap_procs(procs_after_csv, uuid_a) if procs_after_csv.exists() else None
    clean = bool(status.get("contaminated") is False and (rc == 0)
                 and uuid_b and uuid_b == uuid_a
                 and before == [] and after == [])

    res = met["resolution_seen"]
    w, h = int(res[0][0]), int(res[0][1])
    direct = abs(w * h - COMMON_W * COMMON_H) / (COMMON_W * COMMON_H) <= PIXEL_TOL

    rec = {
        "method": "P12_C0", "scene": args.scene,
        "source_commit": args.source_commit,
        "native_iterations": n_iter,
        "training_wall_s": loop_only_s,
        "training_wall_s_including_inloop_eval": timing["total_wall_s"],
        "command_wall_s": delta,
        "peak_vram_mb": round(memory["peak_vram_gb"] * 1024, 1),
        "num_gaussians_final": int(quality["n_gaussians"]),
        "resolution": {"w": w, "h": h},
        "metrics_common": met["mean"],
        "direct_comparable_resolution": direct,
        "clean": clean,
        "training_wall_semantics": (
            "sum of per-iteration training-loop times (mean_iter_ms*n) measured with "
            "perf_counter around each loop body; excludes the in-loop evaluations, the "
            "final all-camera render and checkpoint serialization"),
        "recipe_note": "frozen P12 C0 publication recipe, 30000 iters, seed 42, arm c0; "
                       "trains on ALL cameras (P12 protocol has no train/test split) -- "
                       "metrics_common come from the frozen-checkpoint render on the common "
                       "hold-8 test cameras",
        "command": "publication_trainer_p12.py --arm c0 --scene %s --iterations 30000 "
                   "--lpips --final-eval all --checkpoints --timing-grade PUBLICATION "
                   "--eps2d 0.3 --loss fused" % args.scene,
        "cwd": "/root/epic05_p12",
        "triad_rc": rc, "triad_t0": t0, "triad_t1": t1,
        "median_iter_ms": timing["median_iter_ms"], "mean_iter_ms": timing["mean_iter_ms"],
        "archived_all_camera_quality": {
            "psnr": quality["psnr"], "ssim": quality["ssim"], "lpips": quality["lpips"],
            "n_eval_cameras": quality["n_eval_cameras"],
            "wall_time_s_including_final_render": quality["wall_time_s"],
            "note": "archived P12 protocol numbers (all-camera eval, model trained on all "
                    "cameras); NOT used for cross-method comparison",
        },
        "contamination": {
            "gpu_before_csv": f"{args.log_prefix}.gpu_before.csv",
            "gpu_after_csv": f"{args.log_prefix}.gpu_after.csv",
            "procs_before": before, "procs_after": after,
        },
        "evidence": {
            "results_json": str(rd / "results.json"), "timing_json": str(rd / "timing.json"),
            "memory_json": str(rd / "memory.json"), "quality_json": str(rd / "quality.json"),
            "final_status_json": str(rd / "final_status.json"),
            "common_eval_json": str(Path(args.metrics_json)),
            "ckpt_sha256": sha256_file(rd / "ckpt_30000.pt") if (rd / "ckpt_30000.pt").exists() else None,
            "triad_log": args.triad_log,
            "trainer": args.trainer,
            "trainer_sha256": sha256_file(Path(args.trainer)),
        },
        "n_metrics_expected": met.get("n_expected"), "n_metrics_evaluated": met.get("n_evaluated"),
        "metrics_manifest_sha256": met.get("manifest", {}).get("renders"),
    }
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps({"method": "P12_C0", "scene": args.scene, "training_wall_s": loop_only_s,
                      "command_wall_s": delta, "peak_vram_mb": rec["peak_vram_mb"],
                      "num_gaussians_final": rec["num_gaussians_final"],
                      "metrics_common": rec["metrics_common"], "clean": clean}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
