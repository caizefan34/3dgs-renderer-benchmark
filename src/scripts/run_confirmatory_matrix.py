#!/usr/bin/env python3
"""Run the frozen confirmatory HiGS training matrix.

Pre-registered launcher for paper/confirmatory-protocol.md. Schedules the
frozen ctrl/pd configuration pair for every scene x seed across the given
GPU slots, runs both arms of a pair back-to-back on the same GPU in a
randomized order, retries crashed runs, and writes a manifest plus per-run
JSON outputs and logs.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HARNESS = REPO / "benchmark" / "run_higs_train_benchmark.py"
PYTHON = os.environ.get("GSBENCH_PYTHON", sys.executable)

# Frozen canonical five: (scene, per-scene coarse schedule for the pd arm).
DEFAULT_SCENES = [
    ("mipnerf360/garden", "0.75:0,1.0:1500"),
    ("mipnerf360/bicycle", "0.5:0,1.0:1500"),
    ("mipnerf360/bonsai", "0.5:0,1.0:1500"),
    ("tanks_and_temples/truck", "0.5:0,1.0:1500"),
    ("tanks_and_temples/train", "0.5:0,1.0:1500"),
]
DEFAULT_SEEDS = [3, 4, 5]

# Flags that are identical for both arms (see paper/confirmatory-protocol.md).
FROZEN_FLAGS = {
    "backends": ["higs_dynamic_ts"],
    "n_train": 4,
    "n_eval": 3,
    "densify_every": 5,
    "densify_threshold": 0.005,
    "prune_threshold": 0.01,
    "anchor_densify": True,
    "anchor_densify_every": 2,
    "tile_sampling_ratio": 0.35,
    "sampling_mode": "error_guided",
    "error_alpha": 1.0,
    "error_refresh_every": 25,
    "error_lambda": 0.7,
    "eval_every": 300,
    "lr_decay": 0.1,
    "densify_window": 1500,
    "lpips_loss_weight": 0.1,
    "lpips_loss_every": 25,
    "lpips_full_res": True,
    "cull_interval": 1,
    "masked_adam": True,
    "fused_adam": True,
}

# Flags added only to the pd arm.
PD_FLAGS = {
    "masked_adam_union_decay": 0.99,
    "masked_adam_union_decay_eval_proj": True,
}


def build_command(
    scene: str,
    arm: str,
    seed: int,
    steps: int,
    width: int,
    height: int,
    base_dir: str,
    out_json: str,
    coarse: str | None,
) -> list[str]:
    """Build the harness argv for one confirmatory run."""
    cmd = [
        PYTHON, str(HARNESS),
        "--base-dir", base_dir,
        "--scene", scene,
        "--backends", "higs_dynamic_ts",
        "--n-train", "4", "--n-eval", "3",
        "--steps", str(steps),
        "--width", str(width), "--height", str(height),
        "--seed", str(seed),
        "--densify-every", "5",
        "--densify-threshold", "0.005",
        "--prune-threshold", "0.01",
        "--anchor-densify", "--anchor-densify-every", "2",
        "--tile-sampling-ratio", "0.35",
        "--sampling-mode", "error_guided",
        "--error-alpha", "1.0",
        "--error-refresh-every", "25",
        "--error-lambda", "0.7",
        "--eval-every", "300",
        "--lr-decay", "0.1",
        "--densify-window", "1500",
        "--lpips-loss-weight", "0.1",
        "--lpips-loss-every", "25",
        "--lpips-full-res",
        "--masked-adam",
        "--cull-interval", "1",
        "--out", out_json,
    ]
    if arm == "pd":
        cmd += [
            "--masked-adam-union-decay", "0.99",
            "--masked-adam-union-decay-eval-proj",
            "--res-schedule", coarse,
        ]
    return cmd


def parse_scene_specs(specs: list[str] | None):
    if not specs:
        return DEFAULT_SCENES
    parsed = []
    for spec in specs:
        if ":" in spec:
            scene, coarse = spec.split(":", 1)
        else:
            scene, coarse = spec, None
        parsed.append((scene, coarse))
    return parsed


def run_one(
    gpu: str,
    scene: str,
    arm: str,
    seed: int,
    coarse: str | None,
    run_id: str,
    out_dir: Path,
    steps: int,
    width: int,
    height: int,
    base_dir: str,
    max_retries: int,
    timeout_hours: float,
) -> dict:
    out_json = str(out_dir / f"{run_id}.json")
    log_path = out_dir / f"{run_id}.log"
    cmd = build_command(scene, arm, seed, steps, width, height, base_dir, out_json, coarse)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    record = {
        "run_id": run_id,
        "scene": scene,
        "arm": arm,
        "seed": int(seed),
        "gpu": gpu,
        "coarse_schedule": coarse,
        "out_json": out_json,
        "cmd": cmd,
    }
    for attempt in range(1, max_retries + 2):
        t0 = time.time()
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(f"=== {run_id} attempt {attempt} start {time.ctime()}\n")
            try:
                proc = subprocess.run(
                    cmd, env=env, stdout=lf, stderr=subprocess.STDOUT,
                    timeout=int(timeout_hours * 3600),
                )
                rc = proc.returncode
            except subprocess.TimeoutExpired:
                rc = 124
            lf.write(f"=== {run_id} attempt {attempt} end rc={rc} {time.ctime()}\n")
        record["attempts"] = attempt
        record["rc"] = rc
        record["process_wall_s"] = time.time() - t0
        if rc == 0 and Path(out_json).exists():
            try:
                with open(out_json, encoding="utf-8") as fh:
                    json.load(fh)
                record["status"] = "ok"
                return record
            except Exception:
                rc = 2
        record["status"] = "failed"
    return record


def worker(
    gpu: str,
    pairs: list[dict],
    manifest: dict,
    lock: threading.Lock,
    out_dir: Path,
    args,
):
    for pair in pairs:
        for arm in pair["order"]:
            run_id = f"{pair['scene'].replace('/', '_')}_{arm}_s{pair['seed']}"
            rec = run_one(
                gpu, pair["scene"], arm, pair["seed"], pair["coarse"],
                run_id, out_dir, args.steps, args.width, args.height,
                args.base_dir, args.max_retries, args.timeout_hours,
            )
            with lock:
                manifest["runs"].append(rec)
                _write_manifest(out_dir, manifest)


def _write_manifest(out_dir: Path, manifest: dict) -> None:
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default="datasets/processed")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument(
        "--scenes", nargs="*", default=None,
        help="scene specs 'family/scene[:coarse]' (default: canonical five)",
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--arm-order-seed", type=int, default=20260806)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--timeout-hours", type=float, default=3.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        raise SystemExit("no GPUs specified")

    scenes = parse_scene_specs(args.scenes)
    rng = random.Random(args.arm_order_seed)
    pairs: list[dict] = []
    for scene, coarse in scenes:
        for seed in args.seeds:
            order = ["ctrl", "pd"] if rng.random() < 0.5 else ["pd", "ctrl"]
            pairs.append({"scene": scene, "coarse": coarse, "seed": seed, "order": order})

    per_gpu: dict[str, list[dict]] = {g: [] for g in gpus}
    for index, pair in enumerate(pairs):
        per_gpu[gpus[index % len(gpus)]].append(pair)

    manifest = {
        "protocol": "confirmatory-protocol-2026-08-06",
        "base_dir": args.base_dir,
        "steps": args.steps,
        "width": args.width,
        "height": args.height,
        "seeds": args.seeds,
        "scenes": scenes,
        "arm_order_seed": args.arm_order_seed,
        "max_retries": args.max_retries,
        "gpus": gpus,
        "frozen_flags": FROZEN_FLAGS,
        "pd_flags": PD_FLAGS,
        "start_epoch": time.time(),
        "runs": [],
    }
    _write_manifest(out_dir, manifest)

    lock = threading.Lock()
    threads = [
        threading.Thread(
            target=worker, args=(gpu, per_gpu[gpu], manifest, lock, out_dir, args)
        )
        for gpu in gpus
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    manifest["end_epoch"] = time.time()
    runs = manifest["runs"]
    manifest["summary"] = {
        "n_runs": len(runs),
        "n_ok": sum(1 for r in runs if r.get("status") == "ok"),
        "n_failed": sum(1 for r in runs if r.get("status") != "ok"),
    }
    _write_manifest(out_dir, manifest)
    print(json.dumps(manifest["summary"]))
    return 0 if manifest["summary"]["n_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
