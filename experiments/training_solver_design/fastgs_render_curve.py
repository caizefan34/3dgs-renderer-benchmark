"""Render several FastGS saved iterations with one camera-loading pass.

Uses the author's Scene, GaussianModel, and render_set unchanged. Run from a
pinned FastGS checkout, passing -m MODEL and --iterations 10000 20000 30000.
"""

import os

import torch

from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from render import render_set
from scene import GaussianModel, Scene
from utils.general_utils import safe_state


parser = ArgumentParser()
model = ModelParams(parser, sentinel=True)
pipeline = PipelineParams(parser)
parser.add_argument("--iterations", nargs="+", type=int, required=True)
parser.add_argument("--mult", type=float, default=0.5)
parser.add_argument("--quiet", action="store_true")
args = get_combined_args(parser)
safe_state(args.quiet)
dataset = model.extract(args)
pipe = pipeline.extract(args)

with torch.no_grad():
    gaussians = GaussianModel(dataset.sh_degree, optimizer_type="default")
    scene = Scene(dataset, gaussians, load_iteration=args.iterations[0], shuffle=False)
    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32,
        device="cuda",
    )
    for iteration in args.iterations:
        if iteration != args.iterations[0]:
            gaussians.load_ply(
                os.path.join(dataset.model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
            )
        render_set(dataset.model_path, "test", iteration, scene.getTestCameras(), gaussians, pipe, background, args)
