#!/usr/bin/env python3
"""P12 preregistered campaign manifests.

Preregistration (before ANY 30K P12 run):
- scene order = alphabetical (13 scenes);
- arm order within each pair ALTERNATES across scenes:
    even scene index -> [arm X, arm Y]; odd scene index -> [arm Y, arm X]
  so first-position warm-up bias is counterbalanced;
- P12-A: X=GMAIN_FUSED (gmain), Y=C0_FUSED (c0);
- P12-B: X=A0_FUSED (a0), Y=A1_FUSED (a1);
- GPU assignment (physical, EPIC-05; GPU 4 EXCLUDED - foreign co-tenant):
    GPU0: bicycle, truck        GPU3: treehill, bonsai
    GPU1: garden, drjohnson     GPU5: flowers, room
    GPU2: stump, playroom       GPU6: kitchen, train
    GPU7: counter
  (balance follows the P8 measured per-scene wall times; pairs stay whole
   on one GPU; nothing else runs on that GPU during a pair).
- P12-A and P12-B publication windows MUST NOT overlap on this host; the
  P12-B scheduler is launched only after the P12-A window is closed.
"""
import json
import os
import subprocess
import time

SCENES = ["bicycle", "bonsai", "counter", "drjohnson", "flowers", "garden",
          "kitchen", "playroom", "room", "stump", "train", "treehill", "truck"]
# GPU map v2 (2026-09-26 21:20 UTC): foreign co-tenants occupied GPUs 0, 1
# (32.7 GB each) and 4 (65.7 GB) at preregistration time and persist; the
# campaign is reassigned to the 5 verified-clean GPUs. GPU assignment is NOT
# part of the preregistered arm-order alternation (which is unchanged); each
# pair still runs back-to-back on ONE physical GPU behind the clean-gate.
GPU_OF = {"bicycle": 2, "counter": 2, "playroom": 2,
          "garden": 3, "room": 3, "bonsai": 3,
          "stump": 5, "flowers": 5, "drjohnson": 5,
          "treehill": 6, "kitchen": 6,
          "truck": 7, "train": 7}

CAMPAIGNS = {
    "p12a": {
        "campaign_id": "P12A_SYSTEM_ENDPOINT_FUSED",
        "question": "Under the P11-validated fused-loss baseline, is frozen C0 "
                    "still faster than current gsplat main under the strict "
                    "matched 13-scene 30K protocol?",
        "arms": {"X": {"arm": "gmain", "label": "GMAIN_FUSED"},
                 "Y": {"arm": "c0", "label": "C0_FUSED"}},
        "primary_ratio": "R_sys(scene) = wall_GMAIN_FUSED / wall_C0_FUSED",
    },
    "p12b": {
        "campaign_id": "P12B_F9_REVALIDATION_FUSED",
        "question": "Does the original F9 materialization-free producer retain "
                    "a reproducible full-training benefit after the common "
                    "SepSSIM bottleneck is removed?",
        "arms": {"X": {"arm": "a0", "label": "A0_FUSED"},
                 "Y": {"arm": "a1", "label": "A1_FUSED"}},
        "primary_ratio": "R_F9(scene) = wall_A0_FUSED / wall_A1_FUSED",
    },
}


def make(camp):
    spec = CAMPAIGNS[camp]
    groups = []
    for i, scene in enumerate(SCENES):
        x_first = (i % 2 == 0)
        order = [("X", "Y") if x_first else ("Y", "X")]
        jobs = []
        for k in order[0]:
            arm_spec = spec["arms"][k]
            jobs.append({
                "run_id": f"{camp}_{arm_spec['arm']}f_s42_{scene}",
                "stage": spec["campaign_id"],
                "arm": arm_spec["arm"],
                "arm_label": arm_spec["label"],
                "scene": scene,
                "seed": 42,
                "eps2d": 0.3,
                "iterations": 30000,
                "loss": "fused",
            })
        groups.append({
            "group_id": f"{camp}_pair_{scene}",
            "gpu": GPU_OF[scene],
            "pair_arm_order": [spec["arms"][k]["label"] for k in order[0]],
            "jobs": jobs,
        })
    man = {
        "campaign_id": spec["campaign_id"],
        "preregistered_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "question": spec["question"],
        "primary_ratio": spec["primary_ratio"],
        "alternation_rule": "scene order alphabetical; even index -> [X, Y]; "
                            "odd index -> [Y, X]",
        "pairing_rule": "fresh pair per scene; back-to-back on ONE physical "
                        "GPU; no other publication pair interleaved on that "
                        "GPU; clean-GPU gate before each pair",
        "gpu_assignment": {s: GPU_OF[s] for s in SCENES},
        "excluded_gpus": {"0": "foreign co-tenant host PID 2858677 (32746 MiB) "
                               "since ~2026-09-26 20:50 UTC",
                          "1": "foreign co-tenant host PID 2858678 (32746 MiB) "
                               "since ~2026-09-26 20:50 UTC",
                          "4": "foreign co-tenant host PID 121642 (65672 MiB) "
                               "present before P12 start"},
        "loss": "fused (P11-validated fused-ssim padding=same; "
                "P12_FUSED_LOSS_LOCK.json)",
        "recipe": "eps2d=0.3, seed=42, 30000 iters, reference_v1 frozen "
                  "recipe, eval SSIM metric = SepSSIM (frozen procedure)",
        "host": subprocess.run(["hostname"], capture_output=True,
                               text=True).stdout.strip(),
        "groups": groups,
    }
    out = f"/root/epic05_p12/manifests/{camp}_manifest.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(man, f, indent=2)
    print(f"WROTE {out}  ({len(groups)} pairs, {sum(len(g['jobs']) for g in groups)} jobs)")
    for g in groups:
        print(f"  GPU{g['gpu']} {g['group_id']:24s} "
              f"{[j['run_id'] for j in g['jobs']]}")


if __name__ == "__main__":
    for camp in ("p12a", "p12b"):
        make(camp)
