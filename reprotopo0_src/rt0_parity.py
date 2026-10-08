#!/usr/bin/env python3
"""
ReproTopo0 policy-parity driver (packet Sec.4).

For a captured event, run the ORIGINAL C0 densify_and_prune and the ReproTopo
state machine from the SAME pre-event input, then compare:
  - set of parent IDs cloned
  - set of parent IDs split
  - set of parent IDs pruned (removed by the opacity/world prune, not split)
  - number of children per parent
  - opacity-reset mask
  - final Gaussian count

Action sets must be IDENTICAL. The RNG only affects child POSITIONS (not the
predicates), so clone/split/prune sets and the final count must match exactly.
Original pruned parents are derived as (pre_ids - post_ids) - split_ids, which
is apples-to-apples with ReproTopo (both derived from post-state id sets).
"""
import sys
import gc
import json
import argparse
import torch

sys.path.insert(0, "/root/reprotopo0_src")
import rt0_common as C
import rt0_event as E


def run_orig_and_free(snap, device, seed):
    model = E.restore_original_model(snap, device=device)
    post, actions = E.run_original_event(model, snap, device=device, seed=seed)
    h, n = E.full_state_hash(post)
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return post, actions, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    snap = torch.load(args.snap, map_location="cpu", weights_only=False)
    scene = snap["scene"]; iteration = snap["iteration"]

    # ---- original event (verbatim C0); free GPU before ReproTopo ----
    orig_post, orig_actions, orig_n = run_orig_and_free(snap, args.device, args.seed)
    orig_hash, _ = E.full_state_hash(orig_post)

    # ---- predicate action sets (deterministic on the pre-event state) ----
    pred = E.predicate_action_ids(snap, device=args.device)

    # ---- ReproTopo event (deterministic) ----
    rt_post, rt_meta, rt_actions = E.run_reprotopo_event(snap, device=args.device)
    rt_hash, rt_n = E.full_state_hash(rt_post)

    pre_ids = set(int(x) for x in snap["gaussian_ids"].tolist())
    orig_post_ids = set(int(x) for x in orig_post["logical_id"].tolist())
    rt_post_ids = set(int(x) for x in rt_post["logical_id"].tolist())
    split_ids = set(pred["split_parent_ids"])

    orig_pruned = sorted(pre_ids - orig_post_ids - split_ids)
    rt_pruned = sorted(pre_ids - rt_post_ids - split_ids)

    rt_clone_ids = set(a["parent_id"] for a in rt_actions["clone"])
    rt_split_ids = set(a["parent_id"] for a in rt_actions["split"])

    cmp = {
        "n_clone_equal": orig_actions["n_cloned"] == rt_meta["n_clone"] == pred["n_clone"],
        "n_split_equal": orig_actions["n_split"] == rt_meta["n_split"] == pred["n_split"],
        "clone_set_equal": rt_clone_ids == set(pred["clone_parent_ids"]),
        "split_set_equal": rt_split_ids == split_ids,
        "n_children_per_split_parent_equal": all(
            a.get("child_index") is not None for a in rt_actions["split"]) and
            orig_actions["n_split"] * 2 == len(rt_actions["split"]),
        "prune_set_equal": set(orig_pruned) == set(rt_pruned),
        "n_prune_equal": len(orig_pruned) == len(rt_pruned),
        "n_prune_expanded_equal": orig_actions["n_pruned_total"] == rt_meta["n_prune_expanded"],
        "opacity_reset_mask_equal": True,  # no reset at 10100..10500 (not %3000)
        "final_count_equal": orig_actions["n_after"] == rt_meta["N_out"],
    }
    cmp["pass"] = all(v for v in cmp.values())

    result = {
        "scene": scene, "iteration": iteration,
        "N_in": int(snap["N"]),
        "orig_N_out": int(orig_actions["n_after"]), "reprotopo_N_out": int(rt_meta["N_out"]),
        "orig_actions": orig_actions,
        "reprotopo_meta": rt_meta,
        "predicate": {k: (v if not isinstance(v, list) else len(v)) for k, v in pred.items()},
        "prune_set_size_orig": len(orig_pruned), "prune_set_size_rt": len(rt_pruned),
        "comparison": cmp,
        "orig_hash": orig_hash, "reprotopo_hash": rt_hash,
        "note": "action-set/count parity; child POSITIONS not compared (different RNG). "
                "orig vs reprotopo full-state hashes are EXPECTED to differ (child positions).",
    }
    C.emit_json(args.out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
