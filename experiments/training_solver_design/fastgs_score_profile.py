"""Time FastGS scoring without changing its training implementation.

Run from the pinned FastGS checkout and pass train.py arguments unchanged.
This wrapper is diagnostic only: CUDA synchronization adds to wall-clock time.
"""

import json
import runpy
import sys
import time

import torch

from utils import fast_utils


original_score = fast_utils.compute_gaussian_score_fastgs


def timed_score(camlist, gaussians, pipe, bg, args, DENSIFY=False):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = original_score(camlist, gaussians, pipe, bg, args, DENSIFY=DENSIFY)
    torch.cuda.synchronize()
    print(
        "FASTGS_SCORE_PROFILE "
        + json.dumps(
            {
                "seconds": time.perf_counter() - start,
                "cameras": len(camlist),
                "gaussians": int(gaussians.get_xyz.shape[0]),
                "densify": bool(DENSIFY),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return result


fast_utils.compute_gaussian_score_fastgs = timed_score
sys.argv[0] = "train.py"
runpy.run_path("train.py", run_name="__main__")
