#!/usr/bin/env python3
"""Stage B finalizer: verify the 13-scene ladder is complete, regenerate aggregates
+ figtables, print the final ladder, and stage the Stage B pull. Idempotent.
"""
import json
import os
import subprocess
import sys

PUB = "/mnt/storage_pool/liaoyuanjun/pub_runs"
F30K = "/mnt/storage_pool/liaoyuanjun/final30k_runs"
PHASE = "/mnt/storage_pool/liaoyuanjun/pubphase"

ALL13 = ["bicycle", "bonsai", "counter", "drjohnson", "flowers", "garden",
         "kitchen", "playroom", "room", "stump", "train", "treehill", "truck"]

missing = []
for s in ALL13:
    for a in ("a0", "a1", "a2"):
        if not os.path.exists(f"{PUB}/{a}_{s}/results.json"):
            missing.append(f"{a}_{s}")
    if not os.path.exists(f"{F30K}/c0_{s}/results.json"):
        missing.append(f"c0_{s}")

if missing:
    print(f"STAGE B INCOMPLETE ({len(missing)} missing): {missing}")
    sys.exit(1)

print("LADDER COMPLETE: 39 a-runs + 13 c0 FINAL-30K runs present")

# regenerate
for cmd in (["python3", f"{PHASE}/aggregate_publication.py"],
            ["/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python",
             f"{PHASE}/make_figtables.py"]):
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(f"$ {' '.join(cmd)} -> rc={r.returncode}")
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        sys.exit(1)

# final ladder numbers
A = f"{PHASE}/aggregates"
chain = [("p2_a1_vs_a0", "A0->A1 (F9)"),
         ("p2_a2_vs_a1", "A1->A2 (SCALAR_ADJOINT)"),
         ("p2_c0_vs_a2", "A2->A3 (H8-MR)"),
         ("p2_c0_vs_a0", "A0->C0 (full stack)")]
print(f"\nFINAL 13-scene ladder (seed 42, clean pairs):")
for name, lbl in chain:
    t = json.load(open(f"{A}/{name}.json"))
    agg = t["aggregate"]
    print(f"  {lbl:28s} pairs={agg['n_pairs']} speedup_geomean={agg['speedup_geomean']:.4f} "
          f"faster={agg['n_faster']} slower={agg['n_slower']} "
          f"mean_dPSNR={agg['mean_d_psnr']:+.3f} "
          f"mean_dSSIM={agg['mean_d_ssim']:+.5f} "
          f"mean_dLPIPS={agg['mean_d_lpips']:+.5f} N_geo={agg['n_geo_ratio']:.4f}")

# headline self-check (must be unchanged)
h = json.load(open(f"{A}/headline_c0_vs_b1a_f30k.json"))["aggregate"]
ok = abs(h["speedup_geomean"] - 1.0685) < 0.0005 and abs(h["mean_d_psnr"] - 0.070) < 0.005
print(f"\nheadline self-check: geomean={h['speedup_geomean']:.4f} "
      f"mean_dPSNR={h['mean_d_psnr']:+.4f} -> {'UNCHANGED-OK' if ok else 'DRIFT!!'}")

# stage the pull list
pull = []
for s in ALL13:
    for a in ("a0", "a1", "a2"):
        if not os.path.exists(f"artifacts/.stage_a_marker_{a}_{s}"):
            pull.append(f"{a}_{s}")
with open("/tmp/stage_b_pull_list.txt", "w") as f:
    f.write("\n".join(pull))
print(f"\nstaged pull list: {len(pull)} runs -> /tmp/stage_b_pull_list.txt")
