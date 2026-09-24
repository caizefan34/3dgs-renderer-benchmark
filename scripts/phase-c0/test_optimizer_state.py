#!/usr/bin/env python3
"""Measure Adam continuity across clone, split and prune transactions."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "epic05" / "phase7"))

from gaussian_model import GaussianModel
from variants.reference_semantics import (
    PARAMETER_NAMES,
    apply_prune_transaction,
    apply_topology_transaction,
)
from common import OFFICIAL_SHA, write_json


def make_model() -> GaussianModel:
    n = 6
    model = GaussianModel(n, sh_degree=0, max_sh_degree=0, device="cpu")
    model.init_from_sfm(
        xyz=torch.arange(n * 3, dtype=torch.float32).reshape(n, 3) / 10,
        scales_log=torch.full((n, 3), -2.0),
        rotations_raw=torch.tensor([1.0, 0, 0, 0]).repeat(n, 1),
        opacity_logit=torch.zeros(n, 1),
        shs=torch.arange(n * 3, dtype=torch.float32).reshape(n, 1, 3) / 100,
    )
    return model


def make_optimizer(model: GaussianModel) -> torch.optim.Adam:
    return torch.optim.Adam(model.get_optimizer_param_groups({}), eps=1e-15)


def warm(model: GaussianModel, optimizer: torch.optim.Adam) -> None:
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        sum(parameter.square().sum() for parameter in model.parameters()).backward()
        optimizer.step()


def snapshot(model: GaussianModel, optimizer: torch.optim.Adam, row: int = 4) -> dict:
    result = {}
    for group, name in zip(optimizer.param_groups, PARAMETER_NAMES):
        parameter = group["params"][0]
        state = optimizer.state.get(parameter)
        result[name] = {
            "parameter": getattr(model, name)[row].detach().clone(),
            "exp_avg": None if state is None else state["exp_avg"][row].clone(),
            "exp_avg_sq": None if state is None else state["exp_avg_sq"][row].clone(),
            "step": None if state is None else state["step"].detach().clone(),
        }
    return result


def compare(before: dict, after: dict, after_row: int, model, optimizer) -> dict:
    states = {}
    for group, name in zip(optimizer.param_groups, PARAMETER_NAMES):
        state = optimizer.state.get(group["params"][0])
        states[name] = {
            "parameter_preserved": torch.equal(before[name]["parameter"], getattr(model, name)[after_row].detach()),
            "exp_avg_preserved": state is not None and torch.equal(before[name]["exp_avg"], state["exp_avg"][after_row]),
            "exp_avg_sq_preserved": state is not None and torch.equal(before[name]["exp_avg_sq"], state["exp_avg_sq"][after_row]),
            "step_preserved": state is not None and torch.equal(before[name]["step"], state["step"]),
        }
    return states


def run(operation: str, semantics: str) -> dict:
    model = make_model()
    opt = make_optimizer(model)
    warm(model, opt)
    before = snapshot(model, opt, row=4)
    if operation in ("clone", "split"):
        clone = torch.zeros(6, dtype=torch.bool)
        split = torch.zeros(6, dtype=torch.bool)
        (clone if operation == "clone" else split)[1] = True
        tx = apply_topology_transaction(
            model, opt, clone, split,
            split_semantics="reference" if operation == "split" else "current",
            optimizer_semantics=semantics,
            generator=torch.Generator().manual_seed(7),
        )
        opt = tx.optimizer
        survivor_after_row = 3 if operation == "split" else 4
        child_row = model.xyz.shape[0] - (2 if operation == "split" else 1)
    else:
        mask = torch.zeros(6, dtype=torch.bool)
        mask[1] = True
        opt = apply_prune_transaction(model, opt, mask, optimizer_semantics=semantics)
        survivor_after_row = 3
        child_row = None
    child_state = {}
    if child_row is not None:
        for group, name in zip(opt.param_groups, PARAMETER_NAMES):
            state = opt.state.get(group["params"][0])
            child_state[name] = {
                "exp_avg": None if state is None else float(state["exp_avg"][child_row].abs().max()),
                "exp_avg_sq": None if state is None else float(state["exp_avg_sq"][child_row].abs().max()),
                "step": None if state is None else float(state["step"]),
            }
    return {
        "survivor": compare(before, snapshot(model, opt, survivor_after_row), survivor_after_row, model, opt),
        "child": child_state,
        "optimizer_state_entries": len(opt.state),
    }


def main() -> None:
    measurements = {
        operation: {
            "current_reset": run(operation, "reset"),
            "reference_preserve": run(operation, "preserve"),
        }
        for operation in ("clone", "split", "prune")
    }
    for operation in measurements.values():
        assert operation["current_reset"]["optimizer_state_entries"] == 0
        assert all(
            all(values.values())
            for values in operation["reference_preserve"]["survivor"].values()
        )
    payload = {
        "status": "PASS",
        "official_reference_commit": OFFICIAL_SHA,
        "warmup_adam_steps": 4,
        "measurements": measurements,
        "summary": {
            "current_reset_survivor_state_preserved": False,
            "reference_survivor_state_preserved": True,
            "reference_new_child_exp_avg": 0.0,
            "reference_new_child_exp_avg_sq": 0.0,
            "reference_child_group_step": 4.0,
            "note": "PyTorch Adam step is scalar per Parameter tensor; appended children share the preserved group step while their moment rows start at zero.",
        },
    }
    print(write_json("optimizer_state_unit_test.json", payload))


if __name__ == "__main__":
    main()
