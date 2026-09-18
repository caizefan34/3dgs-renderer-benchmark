#!/usr/bin/env python3
"""
R6 baseline checkpoint generator.

Generates clean BASELINE (no candidate, no skip) checkpoints at 5K/15K/30K for
profiling the exact backward. Training is identical to the B2 reference baseline
(ReferenceV1Config); the ONLY change is checkpoint_iterations=(5000,15000,30000).

This does NOT modify baseline/reference_v1 source. It builds the same config the
r4_train_wrapper --mode baseline builds, then overrides checkpoint_iterations.

Usage (on mx):
  CUDA_VISIBLE_DEVICES=4 PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    ~/miniforge3/envs/anysplat/bin/python experiments/r6/gen_baseline_ckpts.py \
    --scene room --output /mnt/storage_pool/liaoyuanjun/r6_baseline_ckpts/room
"""
import os
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import sys, argparse, json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "baseline" / "reference_v1"))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "epic05" / "phase7"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--resolution", default="1080p")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpts", default="5000,15000,30000",
                        help="comma-separated checkpoint iterations")
    args = parser.parse_args()

    from config import ReferenceV1Config
    from trainer import run_training

    config = ReferenceV1Config()
    config.scene = args.scene
    config.iterations = args.iterations
    config.repo_root = str(REPO_ROOT)
    config.resolution = args.resolution
    config.seed = args.seed
    config.checkpoint_iterations = tuple(
        int(x) for x in args.ckpts.split(",") if x.strip()
    )

    print(f"=== R6 baseline ckpt gen: {args.scene} ===")
    print(f"  repo_root: {config.repo_root}")
    print(f"  checkpoint_iterations: {config.checkpoint_iterations}")
    print(f"  seed: {config.seed}, resolution: {config.resolution}")

    run_training(config, args.output, allow_dirty=True)

    with open(os.path.join(args.output, "r6_ckptgen_info.json"), "w") as f:
        json.dump({
            "scene": args.scene, "mode": "baseline",
            "iterations": args.iterations, "seed": args.seed,
            "checkpoint_iterations": list(config.checkpoint_iterations),
            "resolution": config.resolution,
        }, f, indent=2)
    print(f"=== Done: {args.scene} ===")


if __name__ == "__main__":
    main()
