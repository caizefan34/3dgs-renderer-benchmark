#!/usr/bin/env python3
"""
ReproTopo0 event runners (packet Sec.2-Sec.5).

Provides:
  - restore_original_model(snap, device): rebuild the C0 GaussianModel +
    optimizer from a captured snapshot, ready to run the ORIGINAL event.
    NOTE: GaussianModel.restore() calls training_setup() which ZEROES
    xyz_gradient_accum/denom, so they MUST be re-assigned from the snapshot
    afterwards (otherwise the original event sees zero gradients).
  - run_original_event(model, snap, device, seed): run the original
    densify_and_prune verbatim (reset RNG first), return post-state + actions.
  - run_reprotopo_event(snap, device): run the deterministic state machine.
  - full_state_hash(post): byte hash over logical-state tensors sorted by id.
  - permute_snapshot(snap, perm): row-permuted input copy (Sec.5C).

The original event uses torch.normal for split child positions (draw order
depends on row order + CUDA RNG); ReproTopo uses counter-based RNG keyed by
(parent_id, child_index, component). This is exactly the determinism gap.
"""
import sys
import hashlib
import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import rt0_common as C
import rt0_statemachine as SM

# logical-state tensor names hashed for full-state determinism (packet Sec.2/5C)
LOGICAL_TENSORS = ["logical_id", "xyz", "scaling", "rotation", "opacity", "shs",
                   "max_radii2D", "xyz_gradient_accum", "denom"]


_BOOTSTRAPPED = {"done": False}


def restore_original_model(snap, device="cuda"):
    """Rebuild a C0 GaussianModel + optimizer from a captured snapshot."""
    import rt0_capture as CAP
    if not _BOOTSTRAPPED["done"]:
        CAP.bootstrap_c0()
        _BOOTSTRAPPED["done"] = True
    # GaussianModel.restore() uses .cuda() (default device); pin it to the
    # requested device so cross-GPU runs do not hit a cuda:0 vs cuda:N mismatch.
    torch.cuda.set_device(device)
    from gaussian_model import GaussianModel
    cfg = snap["config"]
    ts_cfg = {
        "position_lr_init": 0.00016, "position_lr_final": 0.0000016,
        "position_lr_delay_mult": 0.01, "position_lr_max_steps": 30000,
        "feature_lr": 0.0025, "opacity_lr": 0.025, "scaling_lr": 0.005,
        "rotation_lr": 0.001, "percent_dense": cfg["percent_dense"],
    }
    model = GaussianModel(max_sh_degree=3)
    state = {
        "active_sh_degree": snap["active_sh_degree"],
        "spatial_lr_scale": snap["scene_extent"],
        "xyz": snap["xyz"], "shs": snap["shs"], "scaling": snap["scaling"],
        "rotation": snap["rotation"], "opacity": snap["opacity"],
        "max_radii2D": snap["max_radii2D"],
        "xyz_gradient_accum": snap["xyz_gradient_accum"],
        "denom": snap["denom"],
    }
    model.restore(state, ts_cfg)
    # restore() -> training_setup() ZEROED accum/denom and made a fresh
    # optimizer; re-assign the captured gradient stats and optimizer moments.
    model.xyz_gradient_accum = snap["xyz_gradient_accum"].to(device)
    model.denom = snap["denom"].to(device)
    model.max_radii2D = snap["max_radii2D"].to(device)
    for pg in model.optimizer.param_groups:
        name = pg["name"]
        p = pg["params"][0]
        d = snap["optimizer"].get(name, None)
        if d is not None and d.get("exp_avg") is not None:
            model.optimizer.state[p] = {
                "step": torch.zeros((), device=device),
                "exp_avg": d["exp_avg"].to(device),
                "exp_avg_sq": d["exp_avg_sq"].to(device),
            }
    # restore explicit logical ids
    model._gaussian_ids = snap["gaussian_ids"].to(device).long()
    model._next_gaussian_id = int(snap["next_gaussian_id"])
    model.tmp_radii = None
    return model


def run_original_event(model, snap, device="cuda", seed=42):
    """Run the ORIGINAL C0 densify_and_prune verbatim. Returns (post, actions)."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.synchronize()
    cfg = snap["config"]
    # radii proxy for the (dead) screen-size prune term; clone/split use scaling.
    model.tmp_radii = snap["max_radii2D"].to(device)
    n_before = int(model._xyz.shape[0])
    ret = model.densify_and_prune(
        max_grad=cfg["densify_grad_threshold"],
        min_opacity=cfg["min_opacity"],
        extent=snap["scene_extent"],
        max_screen_size=cfg["max_screen_size"],
        radii=snap["max_radii2D"].to(device),
    )
    if SM._DYNAMIC_SCENE is not None:
        SM._DYNAMIC_SCENE.mark_dirty()
    post = {
        "logical_id": model._gaussian_ids.detach().cpu(),
        "xyz": model._xyz.detach().cpu(),
        "scaling": model._scaling.detach().cpu(),
        "rotation": model._rotation.detach().cpu(),
        "opacity": model._opacity.detach().cpu(),
        "shs": model._shs.detach().cpu(),
        "max_radii2D": model.max_radii2D.detach().cpu(),
        "xyz_gradient_accum": model.xyz_gradient_accum.detach().cpu(),
        "denom": model.denom.detach().cpu(),
        "optimizer": {
            pg["name"]: {
                "exp_avg": (model.optimizer.state[pg["params"][0]]["exp_avg"].detach().cpu()
                            if model.optimizer.state.get(pg["params"][0]) else None),
                "exp_avg_sq": (model.optimizer.state[pg["params"][0]]["exp_avg_sq"].detach().cpu()
                               if model.optimizer.state.get(pg["params"][0]) else None),
            }
            for pg in model.optimizer.param_groups
        },
    }
    actions = {
        "n_cloned": int(ret["cloned"]),
        "n_split": int(ret["split"]),
        "n_pruned_total": int(ret["pruned_total"]),
        "n_pruned_opacity": int(ret["pruned_opacity"]),
        "n_pruned_screen": int(ret["pruned_screen"]),
        "n_pruned_world": int(ret["pruned_world"]),
        "n_before": n_before,
        "n_after": int(post["logical_id"].numel()),
    }
    return post, actions


def run_reprotopo_event(snap, device="cuda"):
    """Run the deterministic ReproTopo state machine on a snapshot."""
    state, meta, actions = SM.reprotopo_event(snap, device=device)
    post = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in state.items()}
    return post, meta, actions


def full_state_hash(post):
    """Byte hash over logical-state tensors sorted by logical id."""
    return C.canonical_state_hash({k: post[k] for k in LOGICAL_TENSORS if k in post})


def aux_state_hash(post):
    """Separate hash over optimizer moments (auxiliary)."""
    h = hashlib.sha256()
    for gname in sorted(post.get("optimizer", {}).keys()):
        for mname in ("exp_avg", "exp_avg_sq"):
            t = post["optimizer"][gname].get(mname)
            if t is not None and t.numel() > 0:
                h.update(C.sha256_bytes(t.detach().cpu().contiguous().numpy().tobytes()).encode())
    return h.hexdigest()


def permute_snapshot(snap, perm, device="cuda"):
    """Return a row-permuted copy of a snapshot (all per-Gaussian arrays)."""
    # index on CPU (snapshot tensors are CPU), then move to device
    perm = perm.long().cpu() if torch.is_tensor(perm) else torch.as_tensor(perm, dtype=torch.long).cpu()
    out = dict(snap)
    for k in ("xyz", "shs", "scaling", "rotation", "opacity", "max_radii2D",
              "xyz_gradient_accum", "denom", "gaussian_ids"):
        if k in snap and torch.is_tensor(snap[k]):
            out[k] = snap[k][perm].to(device)
    return out


def predicate_action_ids(snap, device="cuda"):
    """Clone/split parent-id sets from the EXACT original predicate (deterministic
    on the pre-event state). Used for policy-parity comparison (Sec.4)."""
    accum = snap["xyz_gradient_accum"].to(device)
    denom = snap["denom"].to(device)
    scaling = snap["scaling"].to(device)
    opacity = snap["opacity"].to(device)
    max_radii2D = snap["max_radii2D"].to(device)
    cfg = snap["config"]
    clone_mask, split_mask, prune_mask, _ = SM._compute_actions(
        accum, denom, scaling, opacity, max_radii2D, snap["scene_extent"], cfg, device)
    parent_ids = snap["gaussian_ids"].to(device).long()
    return {
        "clone_parent_ids": sorted(int(parent_ids[i]) for i in torch.where(clone_mask)[0].tolist()),
        "split_parent_ids": sorted(int(parent_ids[i]) for i in torch.where(split_mask)[0].tolist()),
        "prune_parent_ids_pre": sorted(int(parent_ids[i]) for i in torch.where(prune_mask)[0].tolist()),
        "n_clone": int(clone_mask.sum()),
        "n_split": int(split_mask.sum()),
        "n_prune_pre": int(prune_mask.sum()),
    }
