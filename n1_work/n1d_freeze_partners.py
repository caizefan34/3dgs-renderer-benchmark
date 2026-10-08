#!/usr/bin/env python3
"""N1D Stage A (before ANY new action): identity re-verification + partner freeze.

1. Re-verify the exact P12/C0 source identities and the N1 snapshot
   checkpoints against N1's recorded identities (sha256, N, iteration,
   Adam steps). Any mismatch is a hard stop.
2. Reuse the original N1 anchors and held-out sets EXACTLY (read
   N1_PROBE_LIST.json; verify its sha256 against N1's MANIFEST_N1.json).
3. Freeze THREE deterministic random partners per anchor from the original
   7-view pool using a NEW globally fixed seed (N1D_SEED=20260929):
   per scene, rng = random.Random(N1D_SEED) iterated over that scene's
   anchors in frozen order (checkpoint asc, anchor_id asc);
   partners = rng.sample(pool_ids_in_frozen_order, 3)  # without replacement
   A2-rk uses partners[k-1]. Selection is outcome- and gradient-blind.

Writes (before any N1D action):
  /root/epic05_n1d/artifacts/N1D_SOURCE_IDENTITY.json
  /root/epic05_n1d/artifacts/N1D_PARTNER_LIST.json
"""
import json
import os
import random
import sys

sys.path.insert(0, "/root/epic05_n1")
import n1_core as C

N1D_ROOT = "/root/epic05_n1d"
N1D_ARTIFACTS = os.path.join(N1D_ROOT, "artifacts")
N1D_SEED = 20260929
SCENES = ("room", "bicycle")


def load(p):
    with open(p) as f:
        return json.load(f)


def main():
    os.makedirs(N1D_ARTIFACTS, exist_ok=True)

    # ---------------- 1. source identity re-verification ----------------
    n1_src = load(os.path.join(C.ARTIFACTS, "N1_SOURCE_IDENTITY.json"))
    n1_ck = load(os.path.join(C.ARTIFACTS, "N1_CHECKPOINT_IDENTITY.json"))
    n1_man = load(os.path.join(C.ARTIFACTS, "MANIFEST_N1.json"))

    def manifest_sha(name):
        for grp in ("required_artifacts", "supporting_evidence"):
            for e in n1_man[grp]:
                if e["file"] == name:
                    return e.get("sha256")
        return None

    probe_sha = C.sha256_file(os.path.join(C.ARTIFACTS, "N1_PROBE_LIST.json"))
    probe_ok = probe_sha == manifest_sha("N1_PROBE_LIST.json")
    print(f"[n1d] N1_PROBE_LIST.json sha256 match vs N1 manifest: {probe_ok}")

    boot = C.bootstrap_c0()
    ident = {
        "campaign": "N1D_ReBudgetGS_Mechanism_Disambiguation",
        "task": "DSH_TASK_N1D.md (execution only; raw rows; no verdict)",
        "frozen_prereg": "N1D_PREREG.md (2026-09-29)",
        "reuses": "exact N1 anchors, held-out sets, checkpoints, and frozen "
                  "P12/C0 semantics (N1 campaign artifacts)",
        "sources": {
            "p12_trainer_sha256": C.sha256_file(C.P12_TRAINER),
            "n1_snapshot_trainer_sha256": C.sha256_file(C.N1_SNAP_TRAINER),
            "core_so_sha256": boot["core_sha256"],
            "c0_so_sha256": boot["so_sha256"],
            "fused_ssim_so_sha256": C.P11_FUSED_SO_SHA256,
            "c0_env": boot["env"],
        },
        "n1_probe_list_sha256": probe_sha,
        "n1_probe_list_match": probe_ok,
        "optimizer_note": None,  # filled after model build below
        "lr_scaling_semantics": (
            "One-step diagnostic damping: before each scaled action the five "
            "parameter-group lrs are SET to (action-time master lr) * alpha "
            "(absolute set, never cumulative); after the action they are set "
            "back to the action-time master lrs. update_learning_rate(C+1) "
            "re-establishes the frozen schedule; Adam moments, beta, eps, "
            "state['step'] and the future LR schedule are never modified."),
        "n1d_seed": N1D_SEED,
        "checkpoints": {},
    }

    # checkpoint identity vs N1 records
    all_match = probe_ok
    for s in SCENES:
        rec_n1 = n1_ck["scenes"][s]["checkpoints"]
        ident["checkpoints"][s] = {}
        for it in C.CKPTS:
            p = os.path.join(C.SNAP_RUNS[s], f"ckpt_{it}.pt")
            sha = C.sha256_file(p)
            r1 = rec_n1[str(it)]
            match = (sha == r1["sha256"])
            all_match = all_match and match
            ident["checkpoints"][s][str(it)] = {
                "path": p, "sha256": sha, "bytes": os.path.getsize(p),
                "matches_n1_record": match,
            }
            print(f"[n1d] {s}@{it}: sha256 match vs N1 = {match}")
    ident["all_identities_match_n1"] = all_match
    if not all_match:
        C.json_dump(ident, os.path.join(N1D_ARTIFACTS, "N1D_SOURCE_IDENTITY.json"))
        raise SystemExit("N1D IDENTITY MISMATCH — STOP before any action")

    # optimizer class + lr-group structure (needs a model build)
    ck = C.load_checkpoint("room", 15000)
    model = C.build_model(ckpt=ck)
    opt_cls = type(model.optimizer).__name__
    opt_mod = type(model.optimizer).__module__
    groups = [{"lr": g["lr"], "beta1": g["betas"][0], "beta2": g["betas"][1],
               "eps": g["eps"], "n_params": len(g["params"])}
              for g in model.optimizer.param_groups]
    model.update_learning_rate(15001)
    ident["optimizer_note"] = {
        "class": opt_cls, "module": opt_mod,
        "param_groups_at_restore_then_updateLR_15001": groups,
        "lr_group_count": len(groups),
    }
    print(f"[n1d] optimizer: {opt_mod}.{opt_cls}, {len(groups)} groups, "
          f"eps={groups[0]['eps']}")
    del model, ck

    C.json_dump(ident, os.path.join(N1D_ARTIFACTS, "N1D_SOURCE_IDENTITY.json"))

    # ---------------- 2+3. anchors/held-out reuse + partner freeze --------
    probes = load(os.path.join(C.ARTIFACTS, "N1_PROBE_LIST.json"))
    out = {
        "campaign": "N1D_ReBudgetGS_Mechanism_Disambiguation",
        "n1d_seed": N1D_SEED,
        "freeze_rule": (
            "Per scene: rng = random.Random(N1D_SEED), iterated over that "
            "scene's N1 anchors in frozen order (checkpoint asc, anchor_id "
            "asc); partners = rng.sample(pool partner ids in frozen pool "
            "order, 3) — without replacement, outcome- and gradient-blind, "
            "frozen BEFORE any N1D action. A2-rk uses partners[k-1]."),
        "reused_from_n1": {
            "anchors": "exact N1 anchors (64 per checkpoint per scene)",
            "heldout": "exact N1 held-out 8-camera sets",
            "probe_list_sha256": probe_sha,
        },
        "scenes": {},
    }
    for s in SCENES:
        P = probes["scenes"][s]
        rng = random.Random(N1D_SEED)
        anchors = []
        for a in P["anchors"]:  # frozen order: ckpt asc, anchor_id asc
            pool_ids = [p["partner"] for p in a["pool"]]
            draws = rng.sample(pool_ids, 3)
            anchors.append({
                "checkpoint": a["checkpoint"], "anchor_id": a["anchor_index"],
                "iteration": a["iteration"], "v1": a["v1"],
                "n1_a2_random_partner": a["a2_random_partner"],
                "pool": pool_ids,
                "n1d_partners": draws,
            })
        assert len(anchors) == 192
        for x in anchors:
            assert len(set(x["n1d_partners"])) == 3
            assert all(p in x["pool"] for p in x["n1d_partners"])
        out["scenes"][s] = {
            "heldout_cameras": P["heldout_cameras"],
            "n_anchors": len(anchors),
            "anchors": anchors,
        }
        print(f"[n1d] {s}: froze 3 partners for {len(anchors)} anchors "
              f"(seed {N1D_SEED}); heldout={P['heldout_cameras']}")
    C.json_dump(out, os.path.join(N1D_ARTIFACTS, "N1D_PARTNER_LIST.json"))
    print("[n1d] FREEZE COMPLETE — actions may now run")


if __name__ == "__main__":
    main()
