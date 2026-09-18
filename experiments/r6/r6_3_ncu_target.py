#!/usr/bin/env python3
"""R6-A ncu target: single forward+backward for ncu profiling of rasterize_to_pixels_3dgs_bwd_kernel."""
import os
os.environ.setdefault("PYTHONNOUSERSITE", "1")
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "epic05" / "phase7"))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "baseline" / "reference_v1"))

from gaussian_model import GaussianModel
from config import ReferenceV1Config
from dataset import GTDataset
from trainer import SepSSIM, render_with_meta

scene = sys.argv[1] if len(sys.argv) > 1 else "room"
ckpt = sys.argv[2] if len(sys.argv) > 2 else "/mnt/storage_pool/liaoyuanjun/r6_baseline_ckpts/room/checkpoints/iter_5000.pt"

config = ReferenceV1Config()
config.scene = scene
config.repo_root = str(REPO_ROOT)
dataset = GTDataset(scene=scene, repo_root=config.repo_root, resolution="1080p", device="cuda", background="black")
ck = torch.load(ckpt, map_location="cuda", weights_only=False)
model = GaussianModel(max_sh_degree=config.sh_degree)
model.restore(ck, {
    "position_lr_init": config.position_lr_init, "position_lr_final": config.position_lr_final,
    "position_lr_delay_mult": config.position_lr_delay_mult, "position_lr_max_steps": config.position_lr_max_steps,
    "feature_lr": config.feature_lr, "opacity_lr": config.opacity_lr,
    "scaling_lr": config.scaling_lr, "rotation_lr": config.rotation_lr,
    "percent_dense": config.percent_dense,
})
model.active_sh_degree = ck["active_sh_degree"]
ssim_fn = SepSSIM(device="cuda")
cam, gt = dataset.get_item(0)

# Warmup
for _ in range(3):
    model.optimizer.zero_grad(set_to_none=True)
    img, meta, m2d = render_with_meta(model, cam, model.active_sh_degree)
    loss = 0.8 * F.l1_loss(img, gt) + 0.2 * ssim_fn(img, gt)
    loss.backward()
    torch.cuda.synchronize()

# Profiled iteration
model.optimizer.zero_grad(set_to_none=True)
img, meta, m2d = render_with_meta(model, cam, model.active_sh_degree)
loss = 0.8 * F.l1_loss(img, gt) + 0.2 * ssim_fn(img, gt)
loss.backward()
torch.cuda.synchronize()
print(f"Done: N={model._xyz.shape[0]}")
