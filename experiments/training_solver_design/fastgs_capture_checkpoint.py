"""Save a FastGS model/optimizer snapshot at an exact pre-event iteration.

Set FASTGS_CAPTURE_ITER and FASTGS_CAPTURE_PATH, then pass train.py arguments.
The author train.py accepts checkpoint_iterations but does not write them.
"""

import os
import runpy
import sys
from pathlib import Path

import torch

from scene.gaussian_model import GaussianModel


capture_iteration = int(os.environ["FASTGS_CAPTURE_ITER"])
capture_path = Path(os.environ["FASTGS_CAPTURE_PATH"])
original_step = GaussianModel.optimizer_step


def saving_step(self, iteration):
    original_step(self, iteration)
    if iteration == capture_iteration:
        torch.cuda.synchronize()
        capture_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save((self.capture(self.optimizer_type), iteration), capture_path)
        print(f"FASTGS_CHECKPOINT_CAPTURE {iteration} {capture_path}", flush=True)


GaussianModel.optimizer_step = saving_step
sys.argv[0] = "train.py"
runpy.run_path("train.py", run_name="__main__")
