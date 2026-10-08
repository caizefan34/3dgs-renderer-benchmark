#!/usr/bin/env python
"""
ReproTopo0 deterministic topology state machine (packet Sec.3-Sec.4).

Given a captured pre-event input (from rt0_capture.py), execute the SAME
topology policy as the original C0 densify_and_prune, but as a bitwise-
reproducible state transition:

  A. Candidate identity  : stable 64-bit logical ID inherited from input row.
  B. Action ordering     : parent actions by logical ID ascending; clone
                           children before split children only if original
                           dictates both; prune applied AFTER all policy stats
                           are evaluated, on the EXPANDED array (mirrors the
                           original, where densification_postfix appends clone/
                           split children before prune_points runs).
  C. Split randomness    : counter-based, keyed by
                           (global_seed, event_index, parent_logical_id,
                            child_index, random_component). Device-independent.
  D. Child IDs           : child_logical_id = Hash64(parent_id, event_index,
                            child_index, action_type); collision check.
  E. Compaction/allocation: survivors keep parent logical-ID order; children
                           appended in deterministic (parentID, child_index)
                           order. Output is emitted in this stable build order;
                           all consumers hash via canonical_state_hash which
                           sorts by logical_id internally, so the hash is
                           invariant under input-row permutation.
  F. Opacity reset / aux : opacity-reset mask + optimizer state follow the
                           identical stable remap.

This module does NOT change densification/pruning thresholds or the parent
action predicates (clone/split/prune selection) -- those use the exact
original predicates so policy action-set/count parity holds (Sec.4).
"""
import sys
import math
import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import rt0_common as C

# handle to the C0 HIGS dynamic-scene tracker (set by bootstrap_c0)
_DYNAMIC_SCENE = None


def _logical_ids_from_snapshot(snap):
    """Return int64 logical ids [N] (fallback to row index if absent)."""
    ids = snap.get("gaussian_ids", None)
    if ids is None:
        return torch.arange(snap["N"], dtype=torch.long)
    return ids.to(torch.long)


def _compute_actions(xyz_grad_accum, denom, scaling, opacity, max_radii2D,
                     scene_extent, cfg, device):
    """Compute clone/split/prune masks with the EXACT original predicates.

    Returns (clone_mask, split_mask, prune_mask, grads) all bool/real [N].
    Predicates are identical to reference_v1 GaussianModel:
      clone:  grad_norm >= thresh AND max_scale <= percent_dense*extent
      split:  grad      >= thresh AND max_scale >  percent_dense*extent
      prune:  sigmoid(opacity) < min_opacity OR max_scale > 0.1*extent
              (the screen-size term max_radii2D > max_screen_size is DEAD:
               densification_postfix resets max_radii2D to zero before prune)
    (split uses the raw padded grad on the [N] mean-norm, matching original.)
    """
    grads = (xyz_grad_accum / denom).squeeze(-1)
    grads = torch.nan_to_num(grads, nan=0.0)
    max_scale = torch.exp(scaling).max(dim=1).values
    thresh = cfg["densify_grad_threshold"]
    pd_extent = cfg["percent_dense"] * scene_extent

    grad_norm = torch.norm(grads.reshape(-1, 1), dim=-1)  # [N], grads>=0
    clone_mask = (grad_norm >= thresh) & (max_scale <= pd_extent)
    split_mask = (grads >= thresh) & (max_scale > pd_extent)

    # get_opacity = sigmoid(_opacity).squeeze(-1); compare ACTIVATED opacity.
    opacity_1d = torch.sigmoid(opacity).squeeze(-1)
    prune_mask = opacity_1d < cfg["min_opacity"]
    prune_mask = prune_mask | (max_scale > 0.1 * scene_extent)
    return clone_mask, split_mask, prune_mask, grads


def reprotopo_event(snap, device="cuda"):
    """Execute one deterministic topology event on a captured snapshot.

    Returns a dict with the post-event logical state (tensors keyed by name,
    each row-aligned to 'logical_id') plus action records and metadata.
    Prune is computed on the EXPANDED array (survivors + clones + split
    children), mirroring the original densify_and_prune transaction.

    Output is emitted in a stable build order (survivors by logical id, then
    clones by parent id, then split children by (parent id, child index)).
    No final argsort is performed: canonical_state_hash sorts by logical id
    internally, so the id-sorted byte hash (used by every parity/determinism
    test) is invariant under both input-row permutation and this build order.
    """
    cfg = snap["config"]
    scene_extent = float(snap["scene_extent"])
    event_index = int(snap["event_index"])
    global_seed = int(snap["global_seed"])
    N = int(snap["N"])
    iteration = int(snap.get("iteration", 0))

    xyz = snap["xyz"].to(device)
    shs = snap["shs"].to(device)
    scaling = snap["scaling"].to(device)
    rotation = snap["rotation"].to(device)
    opacity = snap["opacity"].to(device)
    max_radii2D = snap["max_radii2D"].to(device)
    accum = snap["xyz_gradient_accum"].to(device)
    denom = snap["denom"].to(device)
    parent_ids = _logical_ids_from_snapshot(snap).to(device).long()

    clone_mask, split_mask, prune_mask, grads = _compute_actions(
        accum, denom, scaling, opacity, max_radii2D, scene_extent, cfg, device)

    min_op = cfg["min_opacity"]
    log16 = math.log(1.6)
    max_scale_all = torch.exp(scaling).max(dim=1).values  # [N]

    # ---- B. action ordering: sort parent actions by logical ID ascending ----
    clone_idx = torch.where(clone_mask)[0]
    split_idx = torch.where(split_mask)[0]
    clone_order = clone_idx[torch.argsort(parent_ids[clone_idx])]
    split_order = split_idx[torch.argsort(parent_ids[split_idx])]

    # ---- E. survivors: non-split originals, in parent logical-ID order ----
    surv_mask = ~split_mask
    surv_src = torch.where(surv_mask)[0]
    surv_order = surv_src[torch.argsort(parent_ids[surv_src])]
    # prune on the EXPANDED array: a survivor is pruned iff its own row is
    # pruned (opacity / max_scale), identical to original prune_mask on survivors.
    surv_keep = surv_order[~prune_mask[surv_order]]
    # clones inherit the parent prune decision (exact copy -> same opacity/scale).
    clone_keep = clone_order[~prune_mask[clone_order]]

    optim_src = _snapshot_optim(snap, device)  # {group: {"exp_avg": T|None, ...}}
    moments = None
    for _g, _m in optim_src.items():
        moments = _m
        break

    E_xyz, E_shs, E_scaling, E_rotation, E_opacity, E_maxr = [], [], [], [], [], []
    E_accum, E_denom, E_ids = [], [], []
    E_optim = {g: {m: [] for m in moments} for g, moments in optim_src.items()}

    def _push_optim(rows_idx, zeros, count=1):
        for g, moments in optim_src.items():
            for m, t in moments.items():
                if t is None:
                    E_optim[g][m].append(None)
                elif zeros:
                    E_optim[g][m].append(
                        torch.zeros((count,) + tuple(t.shape[1:]), device=t.device, dtype=t.dtype))
                else:
                    E_optim[g][m].append(t[rows_idx])

    # 1) survivors (keep own state; accum/radii reset to 0 like postfix)
    n_surv = int(surv_keep.numel())
    if n_surv > 0:
        E_xyz.append(xyz[surv_keep]); E_shs.append(shs[surv_keep])
        E_scaling.append(scaling[surv_keep]); E_rotation.append(rotation[surv_keep])
        E_opacity.append(opacity[surv_keep]); E_maxr.append(torch.zeros(n_surv, device=device))
        E_accum.append(torch.zeros((n_surv, 1), device=device))
        E_denom.append(torch.zeros((n_surv, 1), device=device))
        E_ids.append(parent_ids[surv_keep])
        _push_optim(surv_keep, zeros=False)

    # 2) clones (exact copy of parent, no noise)
    n_cl = int(clone_keep.numel())
    if n_cl > 0:
        cid = torch.from_numpy(
            C.child_logical_id_np(parent_ids[clone_keep].cpu().numpy(),
                                  event_index, 0, "clone")).to(device)
        E_xyz.append(xyz[clone_keep]); E_shs.append(shs[clone_keep])
        E_scaling.append(scaling[clone_keep]); E_rotation.append(rotation[clone_keep])
        E_opacity.append(opacity[clone_keep]); E_maxr.append(torch.zeros(n_cl, device=device))
        E_accum.append(torch.zeros((n_cl, 1), device=device))
        E_denom.append(torch.zeros((n_cl, 1), device=device))
        E_ids.append(cid)
        _push_optim(clone_keep, zeros=True, count=n_cl)

    # 3) split children (2 per parent; counter-based offset; parent removed)
    n_sp = int(split_order.numel())
    n_spk = 0
    if n_sp > 0:
        sp_pid = parent_ids[split_order]
        parent_scale = torch.exp(scaling[split_order])          # [S,3] activated
        R = _build_rotation(rotation[split_order])             # [S,3,3]
        sp_xyz = xyz[split_order]
        child_maxscale = max_scale_all[split_order] / 1.6
        child_prune = (torch.sigmoid(opacity[split_order]).squeeze(-1) < min_op) | \
                      (child_maxscale > 0.1 * scene_extent)      # [S]
        sp_keep = split_order[~child_prune]
        n_spk = int(sp_keep.numel())
        if n_spk > 0:
            spk_pid = sp_pid[~child_prune]
            pscale = parent_scale[~child_prune]
            Rk = R[~child_prune]
            xyzk = sp_xyz[~child_prune]
            shsk = shs[split_order][~child_prune]
            sclk = scaling[split_order][~child_prune]
            rotk = rotation[split_order][~child_prune]
            opk = opacity[split_order][~child_prune]
            for ci in range(2):
                Z = torch.from_numpy(C.counter_normal_np(
                    spk_pid.cpu().numpy(), global_seed, event_index, ci, n_comp=3)
                ).to(device=device, dtype=rotation.dtype)          # [K,3] std-normal
                sample = pscale * Z                               # N(0, parent_scale)
                child_xyz = torch.einsum('sij,sj->si', Rk, sample) + xyzk
                cid = torch.from_numpy(
                    C.child_logical_id_np(spk_pid.cpu().numpy(),
                                          event_index, ci, "split")).to(device)
                E_xyz.append(child_xyz); E_shs.append(shsk)
                E_scaling.append(sclk - log16); E_rotation.append(rotk); E_opacity.append(opk)
                E_maxr.append(torch.zeros(n_spk, device=device))
                E_accum.append(torch.zeros((n_spk, 1), device=device))
                E_denom.append(torch.zeros((n_spk, 1), device=device))
                E_ids.append(cid)
                _push_optim(sp_keep, zeros=True, count=n_spk)

    # ---- concat expanded arrays (already pruned: filter-during-build) ----
    def _cat(lst, empty):
        return torch.cat(lst) if lst else empty
    state = {
        "logical_id": _cat(E_ids, torch.zeros((0,), dtype=torch.long, device=device)),
        "xyz": _cat(E_xyz, torch.zeros((0, 3), device=device)),
        "shs": _cat(E_shs, torch.zeros((0, 16, 3), device=device)),
        "scaling": _cat(E_scaling, torch.zeros((0, 3), device=device)),
        "rotation": _cat(E_rotation, torch.zeros((0, 4), device=device)),
        "opacity": _cat(E_opacity, torch.zeros((0,), device=device)),
        "max_radii2D": _cat(E_maxr, torch.zeros((0,), device=device)),
        "xyz_gradient_accum": _cat(E_accum, torch.zeros((0, 1), device=device)),
        "denom": _cat(E_denom, torch.zeros((0, 1), device=device)),
        "optimizer": {g: {m: (torch.cat([t for t in ts if t is not None])
                              if any(t is not None for t in ts) else None)
                          for m, ts in moments.items()}
                      for g, moments in E_optim.items()},
    }

    # ---- opacity-reset mask (packet Sec.4): reset only at reset_interval ----
    reset_interval = int(cfg.get("opacity_reset_interval", 0))
    opacity_reset = bool(reset_interval > 0 and iteration % reset_interval == 0)

    n_surv_pruned = int(surv_order.numel() - surv_keep.numel())
    n_cl_pruned = int(clone_order.numel() - clone_keep.numel())
    n_child_pruned = 2 * (n_sp - n_spk)
    meta = {
        "N_in": N,
        "N_out": int(state["logical_id"].numel()),
        "n_clone": int(clone_order.numel()),
        "n_split": n_sp,
        "n_children_per_split_parent": 2,
        "n_prune_expanded": n_surv_pruned + n_cl_pruned + n_child_pruned,
        "event_index": event_index,
        "global_seed": global_seed,
        "iteration": iteration,
        "opacity_reset": opacity_reset,
        "opacity_reset_rows": int(state["logical_id"].numel()) if opacity_reset else 0,
    }
    cc, _ = C.detect_collisions(state["logical_id"])
    meta["collision_count"] = int(cc)
    nan_inf = 0
    for k in ("xyz", "scaling", "rotation", "opacity", "shs"):
        t = state[k]
        if t.numel() > 0:
            nan_inf += int((~torch.isfinite(t)).sum())
    meta["n_nan_inf"] = int(nan_inf)

    # Batched device->host transfer: one DtoH per action list instead of one
    # per element (per-element int(tensor[i]) forces a serialized sync that
    # dominates event wall time).
    clone_pids = parent_ids[clone_order].tolist()
    split_pids = parent_ids[split_order].tolist()
    action_records = {
        "clone": [{"parent_id": int(pid), "child_index": 0} for pid in clone_pids],
        "split": [{"parent_id": int(pid), "child_index": ci}
                  for pid in split_pids for ci in range(2)],
    }
    return state, meta, action_records


def _snapshot_optim(snap, device):
    """Move captured optimizer moments to device.

    Returns {group: {"exp_avg": T|None, "exp_avg_sq": T|None}}.
    """
    out = {}
    for name, d in snap["optimizer"].items():
        out[name] = {
            "exp_avg": d["exp_avg"].to(device) if d.get("exp_avg") is not None else None,
            "exp_avg_sq": d["exp_avg_sq"].to(device) if d.get("exp_avg_sq") is not None else None,
        }
    return out


def _build_rotation(quats):
    """Quaternion -> rotation matrix (matches reference_v1 _build_rotation)."""
    q = torch.nn.functional.normalize(quats, dim=-1)
    r, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.zeros(q.shape[0], 3, 3, device=q.device, dtype=q.dtype)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R
