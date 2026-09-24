"""Shared real-Room experiment implementation for the C0 runners."""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "epic05" / "phase7"))

from dataset import GTDataset, load_initial_checkpoint
from gaussian_model import GaussianModel

# The historical A100 environment has a valid cu118 JIT extension beside an
# incompatible packaged cu124 csrc.so.  Inject the measured historical binary
# process-locally; do not mutate the environment installation.
_historical_extension = Path.home() / ".cache/torch_extensions/py310_cu118/gsplat_cuda/gsplat_cuda.so"
if _historical_extension.exists():
    import importlib.util
    import gsplat
    _spec = importlib.util.spec_from_file_location("gsplat_cuda", _historical_extension)
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    gsplat.csrc = _module
    sys.modules["gsplat.csrc"] = _module

from gsplat import rasterization
from variants.reference_semantics import apply_prune_transaction, apply_topology_transaction

SEED = 42
GRAD_THRESHOLD = 0.001
PRUNE_THRESHOLD = 0.01
PARAMETER_NAMES = ("xyz", "rotations", "scales", "opacity", "shs")
EVAL_CAMERAS = list(range(0, 311, 24))[:13]


class SepSSIM:
    def __init__(self, device: str):
        coords = torch.arange(11, device=device, dtype=torch.float32) - 5
        kernel = torch.exp(-(coords.square()) / (2 * 1.5**2))
        kernel /= kernel.sum()
        self.kh = kernel[None, None, None, :].repeat(15, 1, 1, 1).contiguous()
        self.kv = kernel[None, None, :, None].repeat(15, 1, 1, 1).contiguous()

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.permute(2, 0, 1).unsqueeze(0)
        target = target.permute(2, 0, 1).unsqueeze(0)
        stack = torch.cat((pred, target, pred.square(), target.square(), pred * target), 1)
        blurred = F.conv2d(stack, self.kh, padding=(0, 5), groups=15)
        blurred = F.conv2d(blurred, self.kv, padding=(5, 0), groups=15)
        mp, mt = blurred[:, :3], blurred[:, 3:6]
        vp = blurred[:, 6:9] - mp.square()
        vt = blurred[:, 9:12] - mt.square()
        cov = blurred[:, 12:15] - mp * mt
        value = ((2 * mp * mt + 0.01**2) * (2 * cov + 0.03**2)) / (
            (mp.square() + mt.square() + 0.01**2) * (vp + vt + 0.03**2)
        )
        return value.mean()


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target).item()
    return 100.0 if mse <= 1e-10 else -10 * math.log10(mse)


def render(model: GaussianModel, camera):
    data = model.forward()
    return rasterization(
        means=data["xyz"], quats=data["rotations"], scales=data["scales"],
        opacities=data["opacity"], colors=data["shs"],
        viewmats=camera.viewmatrix.unsqueeze(0), Ks=camera.K.unsqueeze(0),
        width=camera.image_width, height=camera.image_height,
        tile_size=16, packed=False, sh_degree=model.sh_degree,
        radius_clip=0.0, eps2d=0.1, render_mode="RGB",
    )


def make_model(sfm: dict, device: str) -> GaussianModel:
    n = sfm["xyz"].shape[0]
    model = GaussianModel(n, sh_degree=0, max_sh_degree=3, device=device)
    model.init_from_sfm(
        xyz=sfm["xyz"].clone().float().to(device),
        opacity_logit=torch.logit(torch.full((n, 1), 0.1, device=device)),
        scales_log=sfm["scales"].clone().float().to(device),
        rotations_raw=sfm["rotations"].clone().float().to(device),
        shs=sfm["shs"].clone().float().to(device),
    )
    model.set_sh_degree(0)
    return model


def make_optimizer(model: GaussianModel) -> torch.optim.Adam:
    rates = (1.6e-4, 1e-3, 5e-3, 5e-2, 2.5e-3)
    return torch.optim.Adam([
        {"params": [getattr(model, name)], "lr": rate, "name": name}
        for name, rate in zip(PARAMETER_NAMES, rates)
    ], eps=1e-15)


@torch.no_grad()
def resize_sh(model, optimizer, degree: int, optimizer_semantics: str):
    old_parameter = model.shs
    old_state = optimizer.state.get(old_parameter)
    old_coefficients = old_parameter.shape[1]
    model.set_sh_degree(degree)
    if optimizer_semantics == "reset":
        return make_optimizer(model)
    group = optimizer.param_groups[PARAMETER_NAMES.index("shs")]
    optimizer.state.pop(old_parameter, None)
    group["params"][0] = model.shs
    if old_state is not None:
        migrated = dict(old_state)
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            value = old_state.get(key)
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                padded = torch.zeros_like(model.shs)
                padded[:, :old_coefficients] = value
                migrated[key] = padded
        optimizer.state[model.shs] = migrated
    return optimizer


def masks(model: GaussianModel) -> tuple[torch.Tensor, torch.Tensor]:
    avg = model._xyz_grad_accum / model._denf_steps
    high = avg >= GRAD_THRESHOLD
    scale = model.scales.detach().exp()
    small = (scale <= scale.median(dim=0).values).all(dim=-1)
    return high & small, high & ~small


def evaluate(model, dataset, ssim, cameras=EVAL_CAMERAS) -> tuple[float, float]:
    values = []
    with torch.no_grad():
        for index in cameras:
            if index >= len(dataset):
                continue
            image, _, _ = render(model, dataset.get_camera(index))
            pred, gt = image[0].clamp(0, 1), dataset.get_gt_image(index)
            values.append((psnr(pred, gt), float(ssim(pred, gt))))
    return float(np.mean([x[0] for x in values])), float(np.mean([x[1] for x in values]))


def load_room(device: str):
    dataset = GTDataset("room", ROOT, resolution="1080p", device=device)
    sfm = load_initial_checkpoint("room", ROOT, device="cpu")
    return dataset, sfm


def topology_event(model, optimizer, split_semantics, optimizer_semantics, seed):
    model.accumulate_positional_gradient()
    clone_mask, split_mask = masks(model)
    result = apply_topology_transaction(
        model, optimizer, clone_mask, split_mask,
        split_semantics=split_semantics,
        optimizer_semantics=optimizer_semantics,
        generator=torch.Generator(device=model.xyz.device).manual_seed(seed),
    )
    optimizer = result.optimizer
    prune_mask = torch.sigmoid(model.opacity.detach()).squeeze(-1) < PRUNE_THRESHOLD
    pruned = int(prune_mask.sum())
    if pruned:
        optimizer = apply_prune_transaction(
            model, optimizer, prune_mask, optimizer_semantics=optimizer_semantics
        )
    return optimizer, result, pruned


def train_condition(condition: str, iterations: int, device: str) -> dict:
    settings = {
        "A": ("current", "reset"), "B": ("reference", "reset"),
        "C": ("current", "preserve"), "D": ("reference", "preserve"),
    }
    split_semantics, optimizer_semantics = settings[condition]
    torch.manual_seed(SEED); np.random.seed(SEED)
    dataset, sfm = load_room(device)
    model, optimizer = make_model(sfm, device), None
    optimizer = make_optimizer(model)
    ssim = SepSSIM(device)
    camera_order = np.arange(len(dataset)); np.random.shuffle(camera_order)
    per_iteration, events = [], []
    totals = {"clone": 0, "split": 0, "prune": 0}
    opacity_resets = []
    initial_psnr, initial_ssim = evaluate(model, dataset, ssim)
    for iteration in range(iterations):
        start = time.perf_counter()
        camera_index = int(camera_order[iteration % len(camera_order)])
        camera, gt = dataset.get_camera(camera_index), dataset.get_gt_image(camera_index)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        image, _, _ = render(model, camera)
        pred = image[0].clamp(0, 1)
        torch.cuda.synchronize(); t_render = time.perf_counter()
        train_ssim_tensor = ssim(pred, gt)
        train_psnr = psnr(pred, gt)
        loss = 0.8 * F.l1_loss(pred, gt) + 0.2 * (1 - train_ssim_tensor)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.cuda.synchronize(); t2 = time.perf_counter()
        event = iteration >= 500 and iteration < iterations // 2 and iteration % 100 == 0
        iteration_clone = iteration_split = iteration_prune = 0
        if event:
            pre_n = model.xyz.shape[0]
            pre_loss, pre_psnr = float(loss), psnr(pred, gt)
            optimizer, transaction, pruned = topology_event(
                model, optimizer, split_semantics, optimizer_semantics, SEED + iteration
            )
            totals["clone"] += transaction.clone_count
            totals["split"] += transaction.split_count
            totals["prune"] += pruned
            iteration_clone = transaction.clone_count
            iteration_split = transaction.split_count
            iteration_prune += pruned
            with torch.no_grad():
                post, _, _ = render(model, camera); post = post[0].clamp(0, 1)
                post_loss = 0.8 * F.l1_loss(post, gt) + 0.2 * (1 - ssim(post, gt))
            events.append({
                "iteration": iteration, "clone": transaction.clone_count,
                "split": transaction.split_count, "removed_parent": transaction.removed_parent_count,
                "prune": pruned, "population_before": pre_n,
                "population_after": model.xyz.shape[0],
                "population_jump": model.xyz.shape[0] - pre_n,
                "delta_loss": float(post_loss) - pre_loss,
                "delta_psnr": psnr(post, gt) - pre_psnr,
            })
        if iteration > 0 and iteration % 3000 == 0:
            opacity_prune = torch.sigmoid(model.opacity.detach()).squeeze(-1) < PRUNE_THRESHOLD
            reset_pruned = int(opacity_prune.sum())
            if reset_pruned:
                optimizer = apply_prune_transaction(
                    model, optimizer, opacity_prune,
                    optimizer_semantics=optimizer_semantics,
                )
                totals["prune"] += reset_pruned
                iteration_prune += reset_pruned
            elif optimizer_semantics == "reset":
                optimizer = make_optimizer(model)
            activated = torch.sigmoid(model.opacity.detach()).squeeze(-1)
            near = (activated < PRUNE_THRESHOLD * 10) & (activated >= PRUNE_THRESHOLD)
            reset_count = int(near.sum())
            if reset_count:
                model.opacity.data[near] = torch.logit(torch.full(
                    (reset_count, 1), PRUNE_THRESHOLD * 2,
                    dtype=model.opacity.dtype, device=device,
                ))
            opacity_resets.append({"iteration": iteration, "pruned": reset_pruned, "reset": reset_count})
        if iteration and iteration % 1000 == 0 and model.sh_degree < 3:
            optimizer = resize_sh(
                model, optimizer, model.sh_degree + 1, "preserve"
            )
        torch.cuda.synchronize(); t_topology = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize(); t3 = time.perf_counter()
        eval_psnr = eval_ssim = None
        if iteration % 500 == 0 or iteration == iterations - 1:
            eval_psnr, eval_ssim = evaluate(model, dataset, ssim)
        per_iteration.append({
            "iteration": iteration, "loss": float(loss),
            "train_psnr": train_psnr, "train_ssim": float(train_ssim_tensor),
            "eval_psnr": eval_psnr, "eval_ssim": eval_ssim,
            "n_gaussians": model.xyz.shape[0], "clone": iteration_clone,
            "split": iteration_split, "prune": iteration_prune,
            "render_ms": (t_render-t0)*1000, "loss_ms": (t1-t_render)*1000,
            "backward_ms": (t2-t1)*1000, "topology_ms": (t_topology-t2)*1000,
            "optimizer_ms": (t3-t_topology)*1000,
            "total_ms": (t3-start)*1000,
        })
    final_psnr, final_ssim = evaluate(model, dataset, ssim)
    stable = per_iteration[min(100, len(per_iteration)-1):]
    return {
        "status": "COMPLETE", "condition": condition,
        "split_semantics": split_semantics, "optimizer_semantics": optimizer_semantics,
        "iterations": iterations, "seed": SEED,
        "initial_psnr": initial_psnr, "initial_ssim": initial_ssim,
        "final_psnr": final_psnr, "final_ssim": final_ssim,
        "final_n": model.xyz.shape[0], "totals": totals,
        "mean_iter_ms": float(np.mean([row["total_ms"] for row in stable])),
        "per_iteration": per_iteration, "topology_events": events,
        "opacity_reset_events": opacity_resets,
    }
