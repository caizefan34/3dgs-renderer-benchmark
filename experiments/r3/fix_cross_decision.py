#!/usr/bin/env python3
"""Aggregate 3 window JSONs into root-level files for cross-window r3_decision.py."""
import json
import os

MERGED = "/mnt/storage_pool/liaoyuanjun/r3_1_full"
WINDOWS = ["5000", "15000", "30000"]

def merge_json(filename, key_field="iterations"):
    merged = {"provenance": {"merged_from": WINDOWS}, key_field: {}}
    total_violations = 0
    for w in WINDOWS:
        path = os.path.join(MERGED, w, filename)
        if not os.path.exists(path):
            print(f"  WARN: {path} not found")
            continue
        data = json.load(open(path))
        # Merge correctness: has 'correctness' dict with iteration keys
        for top_key in data:
            if top_key == "provenance":
                continue
            if isinstance(data[top_key], dict):
                if top_key not in merged:
                    merged[top_key] = {}
                for iter_key, val in data[top_key].items():
                    merged[top_key][f"{w}_{iter_key}"] = val
                    if isinstance(val, dict):
                        for fam, info in val.items():
                            if isinstance(info, dict) and "violation_count" in info:
                                total_violations += info["violation_count"]
    merged["total_violations"] = total_violations
    out_path = os.path.join(MERGED, filename)
    json.dump(merged, open(out_path, "w"), indent=2)
    print(f"  Wrote {out_path}: total_violations={total_violations}")
    return merged

print("=== Merging window JSONs for cross-window decision ===")
merge_json("certificate_correctness.json", "correctness")
merge_json("certificate_tightness.json", "tightness")
merge_json("complexity_accounting.json", "complexity")
merge_json("exact_zero_statistics.json", "exact_zero")
merge_json("certificate_disabled.json", "disabled")
merge_json("tile_gaussian_certificate.json", "tile_gaussian")

print("=== Running cross-window decision ===")
os.system(f"cd /home/liaoyuanjun/3dgs-renderer-benchmark && PYTHONNOUSERSITE=1 ~/miniforge3/envs/anysplat/bin/python experiments/r3/r3_decision.py --input {MERGED}")

print("=== Cross-window decision result ===")
dec = json.load(open(os.path.join(MERGED, "final_decision.json")))
print(json.dumps(dec, indent=2))
