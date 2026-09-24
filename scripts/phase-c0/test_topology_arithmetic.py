#!/usr/bin/env python3
"""Exact deterministic population and logical-lineage test for C0."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "epic05" / "phase7"))

from gaussian_model import GaussianModel
from variants.reference_semantics import apply_topology_transaction
from common import OFFICIAL_SHA, write_json


def make_model() -> GaussianModel:
    n = 1000
    model = GaussianModel(n, sh_degree=0, max_sh_degree=0, device="cpu")
    scales = torch.empty(n, 3)
    scales[:600] = torch.log(torch.tensor(0.01))
    scales[600:] = torch.log(torch.tensor(0.1))
    model.init_from_sfm(
        xyz=torch.arange(n * 3, dtype=torch.float32).reshape(n, 3) / 1000,
        scales_log=scales,
        rotations_raw=torch.tensor([1.0, 0, 0, 0]).repeat(n, 1),
        opacity_logit=torch.zeros(n, 1),
        shs=torch.zeros(n, 1, 3),
    )
    return model


def optimizer(model: GaussianModel) -> torch.optim.Adam:
    return torch.optim.Adam(model.get_optimizer_param_groups({}), eps=1e-15)


def main() -> None:
    clone_mask = torch.zeros(1000, dtype=torch.bool)
    split_mask = torch.zeros(1000, dtype=torch.bool)
    clone_mask[:100] = True
    split_mask[600:700] = True

    current = make_model()
    current._xyz_grad_accum = torch.zeros(1000)
    current._xyz_grad_accum[clone_mask | split_mask] = 1.0
    current._denf_steps = 1
    torch.manual_seed(20260913)
    counts = current.densification(grad_threshold=0.5)

    current_lineage_model = make_model()
    current_tx = apply_topology_transaction(
        current_lineage_model,
        optimizer(current_lineage_model),
        clone_mask,
        split_mask,
        split_semantics="current",
        optimizer_semantics="reset",
        generator=torch.Generator().manual_seed(20260913),
        logical_ids=[f"g{i}" for i in range(1000)],
    )

    reference = make_model()
    generator = torch.Generator().manual_seed(20260913)
    transaction = apply_topology_transaction(
        reference,
        optimizer(reference),
        clone_mask,
        split_mask,
        split_semantics="reference",
        optimizer_semantics="preserve",
        generator=generator,
        logical_ids=[f"g{i}" for i in range(1000)],
    )
    payload = {
        "status": "PASS",
        "official_reference_commit": OFFICIAL_SHA,
        "input": {"n_initial": 1000, "clone_selected": 100, "split_selected": 100, "split_children": 2},
        "current": {
            "counts_returned": counts,
            "n_after": current.xyz.shape[0],
            "delta_n": current.xyz.shape[0] - 1000,
            "formula": "N + N_clone + 2*N_split",
        },
        "reference": {
            "n_after": reference.xyz.shape[0],
            "delta_n": reference.xyz.shape[0] - 1000,
            "removed_split_parents": transaction.removed_parent_count,
            "formula": "N + N_clone + (2-1)*N_split",
        },
        "difference": current.xyz.shape[0] - reference.xyz.shape[0],
        "lineage": {
            "current_parent_g600_present": any(row["id"] == "g600" for row in current_tx.row_lineage),
            "current_children_of_g600": [row["id"] for row in current_tx.row_lineage if row["parent_id"] == "g600"],
            "reference_parent_g600_present": any(row["id"] == "g600" for row in transaction.row_lineage),
            "reference_children_of_g600": [row["id"] for row in transaction.row_lineage if row["parent_id"] == "g600"],
            "reference_survivor_g700_present": any(row["id"] == "g700" for row in transaction.row_lineage),
        },
        "assertions": {
            "current_n_after_is_1300": current.xyz.shape[0] == 1300,
            "reference_n_after_is_1200": reference.xyz.shape[0] == 1200,
            "current_split_parent_retained": any(row["id"] == "g600" for row in current_tx.row_lineage),
            "reference_split_parent_removed": not any(row["id"] == "g600" for row in transaction.row_lineage),
            "same_masks": True,
        },
    }
    assert all(payload["assertions"].values())
    print(write_json("topology_unit_test.json", payload))


if __name__ == "__main__":
    main()
