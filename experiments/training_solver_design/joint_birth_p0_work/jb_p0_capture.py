"""Small opt-in hook for the frozen SAD joint-birth stage-1 capture."""
import copy
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def _cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu(v) for v in value)
    if isinstance(value, list):
        return [_cpu(v) for v in value]
    return copy.deepcopy(value)


def _snapshot(model):
    names = (
        "_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity",
        "max_radii2D", "xyz_gradient_accum", "xyz_gradient_accum_abs", "denom",
        "filter_3D", "tmp_radii", "accum_eta", "accum_view_count", "max_eta_3ch",
        "accum_weights_valid", "densify_count", "eta_high_count", "eta_high_sum_3ch",
        "eta_mid_count", "eta_mid_sum_3ch", "eta_low_count",
    )
    tensors = {}
    for name in names:
        value = getattr(model, name, None)
        if value is not None:
            tensors[name] = _cpu(value)
    grads = {}
    for name in ("_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity"):
        param = getattr(model, name, None)
        grads[name] = _cpu(param.grad) if param is not None and param.grad is not None else None
    result = {
        "active_sh_degree": int(model.active_sh_degree),
        "spatial_lr_scale": float(model.spatial_lr_scale),
        "percent_dense": float(model.percent_dense),
        "tensors": tensors,
        "grads": grads,
        "optimizer": _cpu(model.optimizer.state_dict()) if model.optimizer is not None else None,
        "shoptimizer": _cpu(model.shoptimizer.state_dict()) if model.shoptimizer is not None else None,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().cpu(),
            "torch_cuda_all": [x.cpu() for x in torch.cuda.get_rng_state_all()],
        },
    }
    return result


def _restore(model, state):
    from torch import nn
    names = ("_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity")
    device = model.get_xyz.device
    model.active_sh_degree = state["active_sh_degree"]
    model.spatial_lr_scale = state["spatial_lr_scale"]
    for name in names:
        setattr(model, name, nn.Parameter(state["tensors"][name].to(device).requires_grad_(True)))
    model.training_setup(SimpleNamespace(**state["training_options"]))
    for name, value in state["tensors"].items():
        if name not in names:
            setattr(model, name, value.to(device))
    model.active_sh_degree = state["active_sh_degree"]
    model.percent_dense = state["percent_dense"]
    model.optimizer.load_state_dict(state["optimizer"])
    model.shoptimizer.load_state_dict(state["shoptimizer"])
    for name, grad in state["grads"].items():
        getattr(model, name).grad = grad.to(device) if grad is not None else None


def install(GaussianModel):
    original_prune = GaussianModel.densify_and_prune_structgs
    original_split = GaussianModel.densify_and_split_structgs

    def wrapped_prune(self, *args, **kwargs):
        self._jb_event_count = getattr(self, "_jb_event_count", 0) + 1
        self._jb_capture_active = self._jb_event_count == int(os.environ.get("SADGS_JB_TARGET_EVENT", "3"))
        if self._jb_capture_active:
            self._jb_structural_mask = _cpu(kwargs.get("custom_split_mask"))
            self._jb_importance = _cpu(kwargs.get("importance_score"))
            self._jb_importance_threshold = float(getattr(kwargs.get("args"), "importance_score_threshold", 0.5))
        try:
            return original_prune(self, *args, **kwargs)
        finally:
            self._jb_capture_active = False

    def wrapped_split(self, metric_mask, max_eta_3ch=None, scale_power=1.0):
        if not getattr(self, "_jb_capture_active", False):
            return original_split(self, metric_mask, max_eta_3ch=max_eta_3ch, scale_power=scale_power)
        outdir = Path(os.environ["SADGS_JB_CAPTURE_DIR"])
        outdir.mkdir(parents=True, exist_ok=True)
        n_entry = int(self.get_xyz.shape[0])
        entry_mask = metric_mask.detach().clone()
        if entry_mask.numel() < n_entry:
            entry_mask = torch.cat([entry_mask, torch.zeros(n_entry-entry_mask.numel(), dtype=torch.bool, device=entry_mask.device)])
        parents = torch.nonzero(entry_mask, as_tuple=False).flatten()
        eta = max_eta_3ch
        if eta is None:
            ks = torch.ones((parents.numel(), 3), dtype=torch.int32, device=parents.device)
            scales = self.get_scaling[parents]
            axes = scales.argmax(dim=1)
            ks.scatter_(1, axes[:, None], 2)
        else:
            eta = eta.to(device=parents.device)
            if eta.shape[0] < n_entry:
                eta = torch.cat([eta, torch.zeros((n_entry-eta.shape[0], 3), device=eta.device)], dim=0)
            ks = torch.sqrt(torch.clamp(eta[parents], min=1.0)).ceil().int().clamp(min=1)
        counts = ks.prod(dim=1).to(torch.int64)
        structural = self._jb_structural_mask
        if structural is None:
            structural = torch.zeros(n_entry, dtype=torch.bool)
        structural = structural.flatten().bool()
        if structural.numel() < n_entry:
            structural = torch.cat([structural, torch.zeros(n_entry-structural.numel(), dtype=torch.bool)])
        importance = self._jb_importance
        if importance is None:
            importance = torch.ones(n_entry)
        importance = importance.flatten()
        if importance.numel() < n_entry:
            importance = torch.cat([importance, torch.zeros(n_entry-importance.numel())])
        candidate = structural[:n_entry].cpu()[parents.cpu()] & (importance[:n_entry].cpu()[parents.cpu()] > self._jb_importance_threshold) & (counts.cpu() >= 2)
        entry = _snapshot(self)
        entry["training_options"] = dict(self._jb_training_options)
        entry["context"] = dict(getattr(self, "_jb_context", {}))
        pipe, background, mult = self._jb_render_context
        camera = self._jb_validation_camera
        from gaussian_renderer import render_structgs
        with torch.no_grad():
            before = render_structgs(camera, self, pipe, background, mult)["render"].detach()
        before_cpu = before.cpu()
        del before
        torch.save({"model": entry, "entry_mask": entry_mask.cpu(), "eta": _cpu(max_eta_3ch),
                    "parents": parents.cpu(), "child_counts": counts.cpu(), "structural_candidates": parents.cpu()[candidate],
                    "event_index": self._jb_event_count}, outdir / "split_entry.pt")
        _restore(self, entry)
        with torch.no_grad():
            after = render_structgs(camera, self, pipe, background, mult)["render"].detach().cpu()
        restore_max_abs = float((before_cpu - after).abs().max())
        restore_mean_abs = float((before_cpu - after).abs().mean())
        if restore_max_abs > 1e-5:
            raise RuntimeError(f"split-entry restore render mismatch: max={restore_max_abs}")
        (outdir / "restore_parity.json").write_text(json.dumps({
            "camera": camera.image_name, "max_abs": restore_max_abs, "mean_abs": restore_mean_abs,
            "tolerance": 1e-5, "passed": True,
        }, indent=2) + "\n")
        del before_cpu, after
        original_split(self, metric_mask, max_eta_3ch=max_eta_3ch, scale_power=scale_power)
        n_post = int(self.get_xyz.shape[0])
        split_set = set(parents.cpu().tolist())
        row_map = [{"kind": "entry", "entry_row": i} for i in range(n_entry) if i not in split_set]
        for parent, count in zip(parents.cpu().tolist(), counts.cpu().tolist()):
            row_map.extend({"kind": "child", "parent_entry_row": parent, "child_index": j} for j in range(count))
        if len(row_map) != n_post:
            raise RuntimeError(f"stable row-map mismatch: {len(row_map)} != {n_post}")
        torch.save({"model": _snapshot(self), "row_map": row_map}, outdir / "all_split.pt")
        meta = {
            "event_index": int(self._jb_event_count), "entry_rows": n_entry, "post_split_rows": n_post,
            "split_parent_count": int(parents.numel()), "candidate_parent_count": int(candidate.sum()),
            "split_child_count": int(counts.sum()), "candidate_child_count": int(counts[candidate.to(counts.device)].sum()),
            "child_count_min": int(counts.min()) if counts.numel() else 0,
            "child_count_max": int(counts.max()) if counts.numel() else 0,
            "row_map_file": "all_split.pt", "entry_file": "split_entry.pt",
            "restore_parity_file": "restore_parity.json",
            "A_camera_names": list(getattr(self, "_jb_a_camera_names", [])),
            "context": dict(getattr(self, "_jb_context", {})),
        }
        (outdir / "capture_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        return None

    GaussianModel.densify_and_prune_structgs = wrapped_prune
    GaussianModel.densify_and_split_structgs = wrapped_split
