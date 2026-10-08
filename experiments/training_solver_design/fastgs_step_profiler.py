"""Profile ordinary full-resolution FastGS training after a fixed checkpoint.

Diagnostic only. The author trainer and renderer are not edited. The first
post-checkpoint optimizer step warms up kernels and starts the profiler; the
next 44 steps are profiled. Final PLY saving is disabled for this probe.
"""

import runpy
import sys

import torch
from torch.profiler import ProfilerActivity, profile

from scene import GaussianModel, Scene


def main():
    original_step = GaussianModel.optimizer_step
    original_save = Scene.save
    state = {"steps": 0, "profiler": None}

    def profiled_step(self, iteration):
        original_step(self, iteration)
        state["steps"] += 1
        if state["steps"] == 1:
            state["profiler"] = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
            )
            state["profiler"].start()
        elif state["steps"] == 45:
            state["profiler"].stop()
            print("FASTGS_PROFILE_STEPS 44", flush=True)
            print(
                state["profiler"].key_averages().table(
                    sort_by="self_cuda_time_total", row_limit=35
                ),
                flush=True,
            )

    try:
        GaussianModel.optimizer_step = profiled_step
        Scene.save = lambda self, iteration: None
        sys.argv = [
            "train.py",
            "-s", "/root/3dgs-renderer-benchmark/data/datasets/mipnerf360/room",
            "-i", "images",
            "--eval",
            "--iterations", "4150",
            "--densification_interval", "100",
            "--optimizer_type", "default",
            "--highfeature_lr", "0.02",
            "--grad_abs_thresh", "0.0004",
            "--test_iterations", "999999",
            "--save_iterations", "999999",
            "--start_checkpoint", "checkpoints/room_big_4099.pth",
            "--model_path", "output/room_big_profile_4100_4150",
        ]
        runpy.run_path("train.py", run_name="__main__")
    finally:
        GaussianModel.optimizer_step = original_step
        Scene.save = original_save


if __name__ == "__main__":
    main()
