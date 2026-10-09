"""R21 diagnostic state contract. Status: IMPLEMENTED; MX NOT_EXECUTED.

All files contain detached CPU tensors. No renderer/trainer production edits.
Stable IDs identify lineage, whereas native Gaussian IDs identify local rows.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

SCHEMA = "r21.gaussian-state.v1"
RAW = ("xyz", "shs", "opacity", "scaling", "rotation")
STATS = ("xyz_gradient_accum", "denom", "max_radii2D")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_sha256(path):
    # Normalize ONLY Windows newlines for cross-platform source identity.
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu(v) for v in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Unsupported checkpoint value: {type(value)}")


def digest(value):
    h = hashlib.sha256()
    def visit(v):
        if isinstance(v, torch.Tensor):
            t = v.detach().cpu().contiguous()
            h.update(str(t.dtype).encode())
            h.update(str(tuple(t.shape)).encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v, dict):
            for k in sorted(v, key=str):
                h.update(str(k).encode()); visit(v[k])
        elif isinstance(v, (tuple, list)):
            for item in v:
                visit(item)
        else:
            h.update(json.dumps(v, sort_keys=True, allow_nan=False).encode())
    visit(value)
    return h.hexdigest()


def row_digest(ids, fields):
    """Canonicalize Gaussian row order before comparing trajectory hashes."""
    order = ids.detach().cpu().argsort()
    def canonical(value):
        if isinstance(value, torch.Tensor):
            return cpu(value)[order]
        if isinstance(value, dict):
            return {key:canonical(item) for key,item in value.items()}
        if value is None:
            return None
        raise TypeError("row_digest accepts per-Gaussian tensors/dicts/None")
    return digest({"ids":cpu(ids)[order], "fields":canonical(fields)})


def describe(value):
    if isinstance(value, torch.Tensor):
        return {"dtype": str(value.dtype), "shape": list(value.shape)}
    if isinstance(value, dict):
        return {str(k): describe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [describe(v) for v in value]
    return type(value).__name__


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def save_state(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing checkpoint overwrite: {path}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(cpu(value), tmp)
    os.replace(tmp, path)
    write_json(path.with_suffix(".json"), {
        "schema": SCHEMA, "sha256": sha256(path), "tensors": describe(value),
        "iteration": value.get("iteration"), "phase": value.get("phase"),
        "instrumentation_overhead": True,
    })


def load_state(path):
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if meta["schema"] != SCHEMA or sha256(path) != meta["sha256"]:
        raise ValueError(f"Checkpoint schema/hash mismatch: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["schema"] != SCHEMA:
        raise ValueError("Unknown checkpoint schema")
    validate_ids(state["stable_ids"])
    return state


def validate_ids(ids):
    if ids.dtype != torch.int64 or ids.ndim != 1 or ids.unique().numel() != ids.numel():
        raise ValueError("Stable IDs must be unique int64 [N]")


def child_ids(parents, iteration, kind, copies=1):
    # split uses .repeat(N, ...) in the actual model: child ordinal OUTER loop.
    values = []
    for ordinal in range(copies):
        for parent in parents.detach().cpu().tolist():
            key = f"R21:{int(parent)}:{iteration}:{kind}:{ordinal}".encode()
            # Negative namespace prevents collision with initial nonnegative IDs.
            values.append(-1 - (int.from_bytes(hashlib.sha256(key).digest()[:8], "big") & ((1 << 62) - 1)))
    result = torch.tensor(values, dtype=torch.int64, device=parents.device)
    validate_ids(result)
    return result


def topology_digest(events):
    """Hash lineage operations, independent of compact row order within a phase."""
    canonical=[]
    for event in events:
        if event["kind"] == "birth":
            parents=event["parents"].tolist(); children=event["children"].tolist()
            if len(children) != len(parents)*event["copies"]:
                raise ValueError("Invalid birth lineage contract")
            edges=sorted((parent,children[ordinal*len(parents)+index],ordinal)
                for ordinal in range(event["copies"]) for index,parent in enumerate(parents))
            if edges:
                canonical.append({"kind":"birth","operation":event["operation"],"edges":edges})
        elif event["kind"] == "remove":
            ids=sorted(event["stable_ids"].tolist())
            if ids:
                canonical.append({"kind":"remove","stable_ids":ids})
        else:
            raise ValueError(f"Unknown topology event: {event['kind']}")
    return digest(canonical)


def align_ids(left, right, require_same=False):
    validate_ids(left); validate_ids(right)
    a, ai = left.cpu().sort(); b, bi = right.cpu().sort()
    pos = torch.searchsorted(b, a)
    good = pos < b.numel()
    if b.numel():
        good &= b[pos.clamp(max=b.numel() - 1)] == a
    if require_same and (good.sum().item() != len(a) or len(a) != len(b)):
        raise ValueError("Intervention requires identical Gaussian lineage sets")
    return ai[good], bi[pos[good]]


def remap(value, source_ids, target_ids):
    target_rows, source_rows = align_ids(target_ids, source_ids, require_same=True)
    result = torch.empty_like(value)
    result[target_rows.to(value.device)] = value[source_rows.to(value.device)]
    return result


def rng_state():
    n = np.random.get_state()
    return {"python": random.getstate(), "numpy": [n[0], n[1].tolist(), int(n[2]), int(n[3]), float(n[4])],
            "torch": torch.get_rng_state(),
            "cuda": [cpu(x) for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG device count mismatch")
        torch.cuda.set_rng_state_all(state["cuda"])


def optimizer_named(model):
    groups = []
    state = {}
    for group in model.optimizer.param_groups:
        if len(group["params"]) != 1:
            raise ValueError("Expected one raw tensor per optimizer group")
        name = group["name"]
        groups.append(cpu({k: v for k, v in group.items() if k != "params"}))
        state[name] = cpu(model.optimizer.state.get(group["params"][0], {}))
    return {"groups": groups, "state": state}


def model_state(model, stable_ids):
    validate_ids(stable_ids)
    return {"stable_ids": cpu(stable_ids), "native_ids": cpu(model._gaussian_ids),
            "next_native_id": model._next_gaussian_id,
            "raw": {k: cpu(getattr(model, "_" + k)) for k in RAW},
            "raw_grad": {k: cpu(getattr(model, "_" + k).grad) for k in RAW},
            "stats": {k: cpu(getattr(model, k)) for k in STATS},
            "optimizer": optimizer_named(model), "rng": rng_state(),
            "active_sh_degree": model.active_sh_degree,
            "spatial_lr_scale": model.spatial_lr_scale,
            "percent_dense": model.percent_dense, "count": len(stable_ids)}


def restore_model(model, state):
    """Do not call GaussianModel.restore: it resets saved statistics and IDs."""
    device = model._xyz.device
    for name in RAW:
        p = torch.nn.Parameter(state["raw"][name].to(device).clone())
        grad = state["raw_grad"][name]
        p.grad = None if grad is None else grad.to(device).clone()
        setattr(model, "_" + name, p)
    model.optimizer.state.clear()
    saved_groups = {g["name"]: g for g in state["optimizer"]["groups"]}
    for group in model.optimizer.param_groups:
        name = group["name"]; p = getattr(model, "_" + name)
        group.update(saved_groups[name]); group["params"] = [p]
        # Standard Adam keeps step on CPU unless capturable/fused is enabled.
        s = {}
        for key, value in state["optimizer"]["state"][name].items():
            target = device if key != "step" or group.get("capturable") or group.get("fused") else "cpu"
            s[key] = value.to(target).clone() if isinstance(value, torch.Tensor) else value
        if s:
            model.optimizer.state[p] = s
    for name in STATS:
        setattr(model, name, state["stats"][name].to(device).clone())
    model._gaussian_ids = state["native_ids"].to(device).clone()
    model._next_gaussian_id = state["next_native_id"]
    for name in ("active_sh_degree", "spatial_lr_scale", "percent_dense"):
        setattr(model, name, state[name])
    model.tmp_radii = None
    return state["stable_ids"].to(device).clone()


def candidates(model, threshold, extent, min_opacity, max_screen_size):
    """Pre-event parent preview; actual post-topology prune is recorded separately."""
    score = model.xyz_gradient_accum / model.denom
    score = score.clone(); score[score.isnan()] = 0.0
    scale = model.get_scaling.max(dim=1).values
    grow = torch.norm(score, dim=-1) >= threshold
    return {"grow_score": score.squeeze(-1), "scale_max": scale,
            "clone": grow & (scale <= model.percent_dense * extent),
            "split": (score.squeeze(-1) >= threshold) & (scale > model.percent_dense * extent),
            "prune_parent_preview": (model.get_opacity < min_opacity) |
                ((model.max_radii2D > max_screen_size) | (scale > .1 * extent)
                 if max_screen_size else torch.zeros_like(grow)),
            "screen_prune_enabled": bool(max_screen_size), "threshold": threshold,
            "scale_threshold": model.percent_dense * extent}


def tensor_metrics(a, b, atol=0.0, rtol=0.0):
    if a.shape != b.shape or a.dtype != b.dtype:
        return {"compatible": False, "left": describe(a), "right": describe(b), "equal": False}
    if not a.numel():
        return {"compatible": True, "equal": True, "close": True, "count": 0}
    x, y = a.double(), b.double()
    finite = torch.isfinite(x) & torch.isfinite(y)
    if not bool(finite.all()):
        return {"compatible": True, "equal": False, "close": False,
                "nonfinite_count": int((~finite).sum())}
    d = x - y
    denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    return {"compatible": True, "equal": torch.equal(a, b),
            "close": bool(torch.allclose(x, y, atol=atol, rtol=rtol)),
            "count": a.numel(), "mae": d.abs().mean().item(),
            "rmse": d.square().mean().sqrt().item(), "max_abs": d.abs().max().item(),
            "relative_l2": (torch.linalg.vector_norm(d) / torch.linalg.vector_norm(x).clamp_min(1e-30)).item(),
            "cosine": (torch.sum(x*y) / denom).item() if denom > 0 else (1.0 if torch.equal(a,b) else 0.0)}
