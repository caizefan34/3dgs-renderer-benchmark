"""GPU-agnostic PyTorch mask oracle for G16, G4, and hierarchical HWCS.

This module generates masks only. It does *not* render sparsely or measure GPU work.
It is intended to be imported by a controlled copy of an existing WV0 replay.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CommonUniforms:
    parent: torch.Tensor
    child: torch.Tensor
    seed: int


def generate_common_uniforms(h: int, w: int, seed: int) -> CommonUniforms:
    """Always return CPU uniforms shared across methods for one camera/seed."""
    if h <= 0 or w <= 0:
        raise ValueError("positive h and w required")
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed))
    p_h, p_w = math.ceil(h / 16), math.ceil(w / 16)
    c_h, c_w = math.ceil(h / 4), math.ceil(w / 4)
    return CommonUniforms(
        parent=torch.rand((p_h, p_w), generator=rng),
        child=torch.rand((c_h, c_w), generator=rng),
        seed=int(seed),
    )


def make_mask(
    h: int,
    w: int,
    uniforms: CommonUniforms,
    method: str,
    *,
    p: float = 0.25,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, dict]:
    """Return weighted HxW pixel mask and reconstructable diagnostics.

    method in {G16, G4, HWCS1, HWCS2}.  The same `uniforms` instance must
    be reused for all four methods at the same camera and seed.
    """
    if not 0 < p <= 1:
        raise ValueError("p must belong to (0,1]")
    if uniforms.parent.shape != (math.ceil(h / 16), math.ceil(w / 16)):
        raise ValueError("wrong parent uniform grid")
    if uniforms.child.shape != (math.ceil(h / 4), math.ceil(w / 4)):
        raise ValueError("wrong child uniform grid")
    if uniforms.parent.device.type != "cpu" or uniforms.child.device.type != "cpu":
        raise ValueError("uniforms must remain on CPU")

    c_h, c_w = uniforms.child.shape
    parent_rows = torch.arange(c_h) // 4
    parent_cols = torch.arange(c_w) // 4
    parent_of_child = uniforms.parent[parent_rows[:, None], parent_cols[None, :]]

    if method == "G16":
        selected_parents = uniforms.parent < p
        selected_children = selected_parents[parent_rows[:, None], parent_cols[None, :]]
        # This produces exactly the same HxW mask as the 16x16 parent expansion.
    elif method == "G4":
        selected_parents = torch.ones_like(uniforms.parent, dtype=torch.bool)
        selected_children = uniforms.child < p
    elif method in ("HWCS1", "HWCS2"):
        q, r = ((0.5, 0.5) if method == "HWCS1" else (0.75, 1.0 / 3.0))
        if not math.isclose(q * r, p, abs_tol=1e-12, rel_tol=1e-12):
            raise ValueError(f"{method} requires p={q*r}")
        selected_parents = uniforms.parent < q
        selected_children = selected_parents[parent_rows[:, None], parent_cols[None, :]] & (uniforms.child < r)
    else:
        raise ValueError(f"unknown method {method!r}")

    selected_pixels = selected_children.repeat_interleave(4, dim=0).repeat_interleave(4, dim=1)[:h, :w]
    weight = selected_pixels.to(device=device, dtype=torch.float32) * (1.0 / p)
    packed = selected_children.to(torch.uint8).contiguous().numpy().tobytes()
    metadata = {
        "method": method,
        "seed": uniforms.seed,
        "p": float(p),
        "active_parent_count": int(selected_parents.sum().item()),
        "selected_child_count": int(selected_children.sum().item()),
        "selected_pixel_count": int(selected_pixels.sum().item()),
        "parent_grid": tuple(uniforms.parent.shape),
        "child_grid": tuple(uniforms.child.shape),
        "child_mask_sha256": hashlib.sha256(packed).hexdigest(),
        "evidence_class": "sampling_mask_only_NOT_GPU_WORK",
    }
    return weight, metadata


def custom_hwcs_mask(
    h: int,
    w: int,
    uniforms: CommonUniforms,
    q: float,
    r: float,
    *,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """General q,r implementation (supports exact endpoint sanity tests)."""
    if not (0 < q <= 1 and 0 < r <= 1):
        raise ValueError("q,r in (0,1]")
    p = q * r
    c_h, c_w = uniforms.child.shape
    pr = torch.arange(c_h) // 4
    pc = torch.arange(c_w) // 4
    parent_selected = (uniforms.parent < q)[pr[:, None], pc[None, :]]
    child_selected = parent_selected & (uniforms.child < r)
    pixels = child_selected.repeat_interleave(4, dim=0).repeat_interleave(4, dim=1)[:h, :w]
    return pixels.to(device=device, dtype=torch.float32) * (1 / p)
