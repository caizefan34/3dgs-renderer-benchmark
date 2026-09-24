"""Isolated topology semantics used by the C0 parity audit.

This module deliberately does not alter the historical GaussianModel.  It
applies either the historical project's child construction or the pinned
Graphdeco child construction to a model exposing ``xyz``, ``rotations``,
``scales``, ``opacity`` and ``shs`` Parameters.  Optimizer handling is an
independent switch so the 2x2 factorial can isolate topology from Adam state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

PARAMETER_NAMES = ("xyz", "rotations", "scales", "opacity", "shs")


@dataclass
class TopologyResult:
    optimizer: torch.optim.Optimizer
    clone_count: int
    split_count: int
    removed_parent_count: int
    row_lineage: list[dict]


def _rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Return Graphdeco-style rotation matrices for w,x,y,z quaternions."""
    q = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    matrix = torch.empty((*q.shape[:-1], 3, 3), dtype=q.dtype, device=q.device)
    matrix[..., 0, 0] = 1 - 2 * (y * y + z * z)
    matrix[..., 0, 1] = 2 * (x * y - w * z)
    matrix[..., 0, 2] = 2 * (x * z + w * y)
    matrix[..., 1, 0] = 2 * (x * y + w * z)
    matrix[..., 1, 1] = 1 - 2 * (x * x + z * z)
    matrix[..., 1, 2] = 2 * (y * z - w * x)
    matrix[..., 2, 0] = 2 * (x * z - w * y)
    matrix[..., 2, 1] = 2 * (y * z + w * x)
    matrix[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return matrix


def _new_optimizer_like(
    optimizer: torch.optim.Optimizer, model: nn.Module
) -> torch.optim.Optimizer:
    groups = []
    for old_group, name in zip(optimizer.param_groups, PARAMETER_NAMES):
        group = {key: value for key, value in old_group.items() if key != "params"}
        group["params"] = [getattr(model, name)]
        groups.append(group)
    return type(optimizer)(groups)


def _install_parameters_and_preserve_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    tensors: dict[str, torch.Tensor],
    survivor_rows: torch.Tensor,
    appended_count: int,
) -> torch.optim.Optimizer:
    """Migrate survivor moments and zero-pad appended rows like Graphdeco."""
    for group, name in zip(optimizer.param_groups, PARAMETER_NAMES):
        old_parameter = group["params"][0]
        old_state = optimizer.state.pop(old_parameter, None)
        new_parameter = nn.Parameter(tensors[name].contiguous().requires_grad_(True))
        setattr(model, name, new_parameter)
        group["params"][0] = new_parameter
        if old_state is not None:
            migrated = dict(old_state)
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                value = old_state.get(key)
                if isinstance(value, torch.Tensor) and value.ndim > 0:
                    padding = torch.zeros(
                        (appended_count, *value.shape[1:]),
                        dtype=value.dtype,
                        device=value.device,
                    )
                    migrated[key] = torch.cat((value[survivor_rows], padding), dim=0)
            optimizer.state[new_parameter] = migrated
    return optimizer


@torch.no_grad()
def apply_topology_transaction(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    clone_mask: torch.Tensor,
    split_mask: torch.Tensor,
    *,
    split_semantics: Literal["current", "reference"],
    optimizer_semantics: Literal["reset", "preserve"],
    generator: torch.Generator | None = None,
    logical_ids: list[str] | None = None,
    split_children: int = 2,
) -> TopologyResult:
    """Apply one clone+split event with independent topology/Adam switches."""
    if clone_mask.dtype != torch.bool or split_mask.dtype != torch.bool:
        raise TypeError("clone_mask and split_mask must be bool tensors")
    n_initial = model.xyz.shape[0]
    if clone_mask.shape != (n_initial,) or split_mask.shape != (n_initial,):
        raise ValueError("masks must match the pre-event population")
    if torch.any(clone_mask & split_mask):
        raise ValueError("clone and split masks must be disjoint")
    if split_children != 2:
        raise ValueError("the audited canonical event uses N=2")

    clone_indices = torch.where(clone_mask)[0]
    split_indices = torch.where(split_mask)[0]
    ids = logical_ids or [f"g{i}" for i in range(n_initial)]
    if len(ids) != n_initial:
        raise ValueError("logical_ids must match the pre-event population")

    original = {name: getattr(model, name).detach() for name in PARAMETER_NAMES}
    activated_scales = original["scales"].exp()

    if split_semantics == "current":
        clone_noise = torch.randn(
            (len(clone_indices), 3), device=model.xyz.device,
            dtype=model.xyz.dtype, generator=generator,
        ) * 0.01 * activated_scales[clone_indices]
        clone_xyz = original["xyz"][clone_indices] + clone_noise
        split_xyz = original["xyz"][split_indices].repeat(split_children, 1)
        split_noise = torch.randn(
            split_xyz.shape, device=split_xyz.device, dtype=split_xyz.dtype,
            generator=generator,
        ) * 0.0025
        split_xyz = split_xyz + split_noise
        split_scales = (
            original["scales"][split_indices] - torch.log(
                torch.tensor(float(split_children), device=model.xyz.device)
            )
        ).repeat(split_children, 1)
        survivor_rows = torch.arange(n_initial, device=model.xyz.device)
    else:
        clone_xyz = original["xyz"][clone_indices]
        stds = activated_scales[split_indices].repeat(split_children, 1)
        samples = torch.normal(
            mean=torch.zeros_like(stds), std=stds, generator=generator
        )
        rotations = _rotation_matrix(original["rotations"][split_indices]).repeat(
            split_children, 1, 1
        )
        split_xyz = (
            torch.bmm(rotations, samples.unsqueeze(-1)).squeeze(-1)
            + original["xyz"][split_indices].repeat(split_children, 1)
        )
        split_scales = torch.log(stds / (0.8 * split_children))
        survivor_rows = torch.where(~split_mask)[0]

    appended: dict[str, torch.Tensor] = {}
    appended["xyz"] = torch.cat((clone_xyz, split_xyz), dim=0)
    for name in ("rotations", "opacity", "shs"):
        appended[name] = torch.cat(
            (
                original[name][clone_indices],
                original[name][split_indices].repeat(
                    split_children, *([1] * (original[name].ndim - 1))
                ),
            ),
            dim=0,
        )
    appended["scales"] = torch.cat(
        (original["scales"][clone_indices], split_scales), dim=0
    )
    tensors = {
        name: torch.cat((original[name][survivor_rows], appended[name]), dim=0)
        for name in PARAMETER_NAMES
    }

    lineage = []
    for row in survivor_rows.tolist():
        lineage.append({"id": ids[row], "kind": "survivor", "parent_id": None})
    for row in clone_indices.tolist():
        lineage.append({"id": f"{ids[row]}.clone", "kind": "clone", "parent_id": ids[row]})
    for generation in range(split_children):
        for row in split_indices.tolist():
            lineage.append({
                "id": f"{ids[row]}.split{generation}",
                "kind": "split_child",
                "parent_id": ids[row],
            })

    appended_count = len(clone_indices) + split_children * len(split_indices)
    if optimizer_semantics == "preserve":
        optimizer = _install_parameters_and_preserve_state(
            model, optimizer, tensors, survivor_rows, appended_count
        )
    else:
        for name in PARAMETER_NAMES:
            setattr(model, name, nn.Parameter(tensors[name].contiguous()))
        optimizer = _new_optimizer_like(optimizer, model)
    model.num_points = model.xyz.shape[0]
    if hasattr(model, "_xyz_grad_accum"):
        model._xyz_grad_accum = None
        model._denf_steps = 0
    return TopologyResult(
        optimizer=optimizer,
        clone_count=len(clone_indices),
        split_count=len(split_indices),
        removed_parent_count=len(split_indices) if split_semantics == "reference" else 0,
        row_lineage=lineage,
    )


@torch.no_grad()
def apply_prune_transaction(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    prune_mask: torch.Tensor,
    *,
    optimizer_semantics: Literal["reset", "preserve"],
) -> torch.optim.Optimizer:
    """Prune rows with either project reset or Graphdeco state migration."""
    survivor_rows = torch.where(~prune_mask)[0]
    tensors = {
        name: getattr(model, name).detach()[survivor_rows]
        for name in PARAMETER_NAMES
    }
    if optimizer_semantics == "preserve":
        return _install_parameters_and_preserve_state(
            model, optimizer, tensors, survivor_rows, appended_count=0
        )
    for name in PARAMETER_NAMES:
        setattr(model, name, nn.Parameter(tensors[name].contiguous()))
    model.num_points = model.xyz.shape[0]
    return _new_optimizer_like(optimizer, model)


def reference_prune_mask(
    opacity_logit: torch.Tensor,
    scales_log: torch.Tensor,
    *,
    min_opacity: float,
    scene_extent: float,
    max_radii2d: torch.Tensor | None = None,
    max_screen_size: float | None = None,
) -> torch.Tensor:
    """Pinned Graphdeco opacity plus optional screen/world-size prune mask."""
    mask = torch.sigmoid(opacity_logit).squeeze(-1) < min_opacity
    if max_screen_size is not None:
        if max_radii2d is None:
            raise ValueError("max_radii2d is required with max_screen_size")
        mask |= max_radii2d > max_screen_size
        mask |= scales_log.exp().max(dim=1).values > 0.1 * scene_extent
    return mask


@torch.no_grad()
def reference_opacity_reset(
    model: nn.Module, optimizer: torch.optim.Optimizer
) -> torch.optim.Optimizer:
    """Clamp all opacities to 0.01 and zero their moments as Graphdeco does."""
    activated = torch.sigmoid(model.opacity)
    reset = torch.logit(torch.minimum(activated, torch.full_like(activated, 0.01)))
    opacity_index = PARAMETER_NAMES.index("opacity")
    group = optimizer.param_groups[opacity_index]
    old = group["params"][0]
    state = optimizer.state.pop(old, None)
    new = nn.Parameter(reset.contiguous().requires_grad_(True))
    model.opacity = new
    group["params"][0] = new
    if state is not None:
        migrated = dict(state)
        migrated["exp_avg"] = torch.zeros_like(new)
        migrated["exp_avg_sq"] = torch.zeros_like(new)
        optimizer.state[new] = migrated
    return optimizer
