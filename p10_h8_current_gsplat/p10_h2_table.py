#!/usr/bin/env python3
"""Compact H2 gate table (stdout) from the per-scene equivalence JSONs."""
import argparse
import json

AP = argparse.ArgumentParser()
AP.add_argument("--dir", default="dumps")
AP.add_argument("--scenes", default="room,bicycle,garden")
AP.add_argument("--out", default=None)
A = AP.parse_args()

KEYS = ("bias_l2_ratio", "bias_max_ratio", "tail_ratio", "pass")
table = {}
for s in A.scenes.split(","):
    d = json.load(open(f"{A.dir}/EPIC05_P10_EQUIVALENCE_{s}.json"))
    v = d["verdict"]
    row = {
        "classification": v["classification"],
        "switch_engaged": v["switch_engagement_control"],
        "boundary_reconstruction_equivalent": v["boundary_reconstruction_equivalent"],
        "absgrad_path_equivalent": v["absgrad_path_equivalent"],
        "patched_binary_inert_when_switch_off": v["patched_binary_inert_when_switch_off"],
        "boundary_representations": v["boundary_representations"],
        "pipeline": {k: {kk: v["pipeline_gate"][k][kk] for kk in KEYS} for k in v["pipeline_gate"]},
        "inert": {k: {kk: v["cross_build_inert_gate"][k][kk] for kk in KEYS}
                  for k in v["cross_build_inert_gate"]},
        "inert_null": v["cross_build_inert_null"]["gate_null"],
        "elevated_h8_nondeterminism": v["elevated_h8_nondeterminism"],
    }
    table[s] = row
    pg = row["pipeline"]
    print(f"{s:8s} {row['classification']:32s} engaged={row['switch_engaged']} "
          f"recon={row['boundary_reconstruction_equivalent']} "
          f"absgrad={row['absgrad_path_equivalent']} "
          f"inert(all)={row['patched_binary_inert_when_switch_off']} "
          f"null={row['inert_null']}")
    for k, g in pg.items():
        print(f"    pipeline {k:12s} bias_l2={g['bias_l2_ratio']:.2f} "
              f"bias_max={g['bias_max_ratio']:.2f} tail={g['tail_ratio']:.2f} pass={g['pass']}")
    for k, g in row["inert"].items():
        print(f"    inert    {k:12s} bias_l2={g['bias_l2_ratio']:.2f} "
              f"bias_max={g['bias_max_ratio']:.2f} tail={g['tail_ratio']:.2f} pass={g['pass']}")

if A.out:
    json.dump(table, open(A.out, "w"), indent=2)
    print(f"-> {A.out}")
