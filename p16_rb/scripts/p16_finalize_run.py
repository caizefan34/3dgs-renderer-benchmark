#!/usr/bin/env python3
"""P16-RB Stage-1 finalizer: common metrics + normalized run record for one (method, scene).

Locates the run's native outputs, evaluates the native test render with the ONE common
metric pipeline, then assembles runs/<method>/<scene>/P16_RUN.json from the wrapper,
the VRAM sampler, the framework's own timing accounting and the metrics.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

D = Path("/root/epic05_p16rb")
PY = "/root/miniforge3/envs/p16_fastergs/bin/python"
GS = "/root/miniforge3/envs/gsplat/bin/python"
FINALIZE_VERSION = "2026-09-28.p16rb.stage1"


def metric_env(gpu: int):
    e = dict(os.environ)
    e["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return e


def newest_dir(parent: Path, pattern: str):
    cands = sorted(parent.glob(pattern), key=lambda p: p.stat().st_mtime) if parent.exists() else []
    return cands[-1] if cands else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--source-commit", required=True)
    ap.add_argument("--native-iterations", type=int, required=True)
    ap.add_argument("--recipe-note", default="")
    ap.add_argument("--vram-csv", required=True, help="external VRAM sampler CSV for this run")
    ap.add_argument("--stdout-log", default=None, help="graphdeco train stdout log to parse")
    ap.add_argument("--eval-gpu", type=int, default=3,
                    help="GPU for the metric evaluation itself (kept off the training card)")
    ap.add_argument("--target-gpu", type=int, default=2,
                    help="physical GPU the measured training ran on (cleanliness scope)")
    args = ap.parse_args()

    M, SC = args.method, args.scene
    OUT = D / "runs" / M / SC
    metrics_json = OUT / "metrics_common.json"

    if M in ("FasterGS", "FasterGSFused"):
        outdir = newest_dir(D / "src" / "nerficg" / "output" / M, f"p16_{M.lower()}_{SC}_*")
        if outdir is None:
            raise SystemExit(f"no nerficg output dir for {M} {SC}")
        test_dirs = sorted(outdir.glob("test_*"), key=lambda p: p.stat().st_mtime)
        if not test_dirs:
            raise SystemExit(f"no test_* render dir under {outdir}")
        rgb_dir = test_dirs[-1] / "rgb"
        gt_dir = test_dirs[-1] / "rgb_gt"
        namemap_json = OUT / "name_map.json"
        subprocess.run([GS, str(D / "scripts" / "p16_name_map.py"), "--method", M, "--scene", SC,
                        "--rgb-dir", str(rgb_dir), "--gt-dir", str(gt_dir),
                        "--out", str(namemap_json)], check=True, env=metric_env(args.eval_gpu))
        cmd = [GS, str(D / "scripts" / "p16_common_metrics.py"), "--method", M, "--scene", SC,
               "--rgb-dir", str(rgb_dir), "--name-map", str(namemap_json), "--out", str(metrics_json)]
        subprocess.run(cmd, check=True, env=metric_env(args.eval_gpu))
        build = [PY, str(D / "scripts" / "p16_build_run_json.py"),
                 "--method", M, "--scene", SC, "--source-commit", args.source_commit,
                 "--native-iterations", str(args.native_iterations),
                 "--wrapper-json", str(OUT / "wrapper_train.json"),
                 "--metrics-json", str(metrics_json),
                 "--vram-csv", args.vram_csv,
                 "--timings-txt", str(outdir / "timings.txt"),
                 "--n-gaussians-txt", str(outdir / "n_gaussians.txt"),
                 "--vram-stats-txt", str(outdir / "vram_stats.txt"),
                 "--target-gpu", str(args.target_gpu), "--recipe-note", args.recipe_note,
                 "--out", str(OUT / "P16_RUN.json")]
        subprocess.run(build, check=True)
        (OUT / "finalize_meta.json").write_text(json.dumps(
            {"finalize_version": FINALIZE_VERSION, "outdir": str(outdir),
             "test_dir": str(test_dirs[-1]), "rgb_dir": str(rgb_dir),
             "gt_dir": str(gt_dir), "name_map": str(namemap_json)}, indent=2), encoding="utf-8")
        print(f"[{M} {SC}] record written from {outdir}")
        return 0

    # graphdeco lineage
    if not args.stdout_log:
        raise SystemExit("graphdeco lineage requires --stdout-log")
    job = f"p16_{M}_{SC}"
    model_dir = D / "src" / M / "output" / job
    renders = sorted((model_dir / "test").glob("ours_*/renders"), key=lambda p: p.stat().st_mtime)
    if not renders:
        raise SystemExit(f"no rendered test images under {model_dir}")
    rgb_dir = renders[-1]
    gt_dir = rgb_dir.parent / "gt"
    namemap_json = OUT / "name_map.json"
    subprocess.run([GS, str(D / "scripts" / "p16_name_map.py"), "--method", M, "--scene", SC,
                    "--rgb-dir", str(rgb_dir), "--gt-dir", str(gt_dir),
                    "--out", str(namemap_json)], check=True, env=metric_env(args.eval_gpu))
    cmd = [GS, str(D / "scripts" / "p16_common_metrics.py"), "--method", M, "--scene", SC,
           "--rgb-dir", str(rgb_dir), "--name-map", str(namemap_json), "--out", str(metrics_json)]
    subprocess.run(cmd, check=True, env=metric_env(args.eval_gpu))
    build = [PY, str(D / "scripts" / "p16_build_run_json.py"),
             "--method", M, "--scene", SC, "--source-commit", args.source_commit,
             "--native-iterations", str(args.native_iterations),
             "--wrapper-json", str(OUT / "wrapper_train.json"),
             "--metrics-json", str(metrics_json),
             "--vram-csv", args.vram_csv,
             "--native-log", args.stdout_log,
             "--target-gpu", str(args.target_gpu), "--recipe-note", args.recipe_note,
             "--out", str(OUT / "P16_RUN.json")]
    subprocess.run(build, check=True)
    (OUT / "finalize_meta.json").write_text(json.dumps(
        {"finalize_version": FINALIZE_VERSION, "model_dir": str(model_dir),
         "rgb_dir": str(rgb_dir), "gt_dir": str(gt_dir), "name_map": str(namemap_json)},
        indent=2), encoding="utf-8")
    print(f"[{M} {SC}] record written from {model_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
