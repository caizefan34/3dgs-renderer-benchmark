"""PyTorch reference for a numerically stable survival-conserving child kernel.

This checks a single-pixel fixed-order compositor, including gradients near
saturated opacity. It is not a CUDA renderer or a complete training result.
"""

import json
from pathlib import Path

import torch


def child_alpha(logit, footprint):
    """Compute 1-sqrt(1-sigmoid(logit)*footprint) without 1-alpha cancellation."""
    parent_survival = torch.sigmoid(-logit)
    survival = (1.0 - footprint) + footprint * parent_survival
    root = torch.sqrt(survival)
    # Direct subtraction is stable when survival is small; rationalization is
    # stable when child opacity is small. Both branches represent the same h.
    return torch.where(
        survival < 0.5,
        1.0 - root,
        (footprint * (1.0 - parent_survival)) / (1.0 + root),
    )


def compare(dtype, n=10000):
    generator = torch.Generator().manual_seed(270927)
    logit = (torch.rand(n, generator=generator, dtype=dtype) * 24.0 - 12.0)
    logit[:4] = torch.tensor([16.0, 20.0, 30.0, 40.0], dtype=dtype)
    logit.requires_grad_()
    footprint = torch.rand(n, generator=generator, dtype=dtype)
    footprint[:4] = 1.0
    prefix_survival = torch.rand(n, generator=generator, dtype=dtype)
    color = torch.rand(n, 3, generator=generator, dtype=dtype)
    suffix = torch.rand(n, 3, generator=generator, dtype=dtype)
    adjoint = torch.rand(n, 3, generator=generator, dtype=dtype) - 0.5

    alpha = torch.sigmoid(logit) * footprint
    parent = prefix_survival[:, None] * (
        alpha[:, None] * color + (1.0 - alpha[:, None]) * suffix
    )
    h = child_alpha(logit, footprint)
    children = prefix_survival[:, None] * (
        h[:, None] * color
        + (1.0 - h[:, None]) * (h[:, None] * color + (1.0 - h[:, None]) * suffix)
    )
    parent_grad = torch.autograd.grad((parent * adjoint).sum(), logit, retain_graph=True)[0]
    child_grad = torch.autograd.grad((children * adjoint).sum(), logit)[0]
    assert torch.isfinite(h).all() and torch.isfinite(child_grad).all()
    return {
        "dtype": str(dtype),
        "n": n,
        "max_image_abs_diff": float((parent - children).detach().abs().max()),
        "max_grad_abs_diff": float((parent_grad - child_grad).abs().max()),
        "saturated_logits": [16.0, 20.0, 30.0, 40.0],
        "saturated_child_alpha": h[:4].detach().tolist(),
        "saturated_parent_grad": parent_grad[:4].detach().tolist(),
        "saturated_child_grad": child_grad[:4].detach().tolist(),
    }


def naive_saturation():
    z = torch.tensor(20.0, requires_grad=True)
    h = 1.0 - torch.sqrt(1.0 - torch.sigmoid(z))
    grad = torch.autograd.grad(h, z)[0]
    return {
        "value": float(h.detach()),
        "grad_is_finite": bool(torch.isfinite(grad)),
        "grad": float(grad.detach()) if torch.isfinite(grad) else None,
    }


if __name__ == "__main__":
    result = {
        "scope": "single-pixel differentiable reference, fixed child adjacency",
        "fp64": compare(torch.float64),
        "fp32": compare(torch.float32),
        "naive_fp32_logit20": naive_saturation(),
    }
    out = Path(__file__).with_name("transmittance_split_reference.json")
    out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))
