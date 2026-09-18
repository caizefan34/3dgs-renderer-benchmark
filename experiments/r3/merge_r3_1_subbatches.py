#!/usr/bin/env python3
"""
R3.1 Sub-Batch Merger

Merges outputs from parallel sub-batches into per-window directories.
Each sub-batch produces certificate_correctness.json, certificate_tightness.json,
etc. This script combines them into a single set per window.
"""
import json
import os
import sys
import shutil
import numpy as np
from pathlib import Path
from collections import defaultdict

PARALLEL_DIR = Path("/mnt/storage_pool/liaoyuanjun/r3_1_parallel")
MERGED_DIR = Path("/mnt/storage_pool/liaoyuanjun/r3_1_full")

WINDOWS = {
    "5000": ["5000_A", "5000_B", "5000_C"],
    "15000": ["15000_A", "15000_B", "15000_C"],
    "30000": ["30000_A", "30000_B", "30000_C"],
}

JSON_FILES = [
    "certificate_correctness.json",
    "certificate_tightness.json",
    "exact_zero_statistics.json",
    "complexity_accounting.json",
    "certificate_disabled.json",
    "tile_gaussian_certificate.json",
]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def merge_json(sub_dirs, key, out_path):
    """Merge a JSON file across sub-batches."""
    merged = {}
    provenance_parts = []
    total_iters = 0

    for sd in sub_dirs:
        path = sd / key
        if not path.exists():
            print(f"  [WARN] {path} not found, skipping")
            continue
        data = load_json(path)

        # The main data is under a top-level key (e.g., "correctness", "tightness")
        main_key = key.replace("certificate_", "").replace(".json", "")
        if main_key == "exact_zero_statistics":
            main_key = "exact_zero"
        elif main_key == "complexity_accounting":
            main_key = "complexity"
        elif main_key == "disabled":
            pass  # already "disabled"
        elif main_key == "tile_gaussian_certificate":
            main_key = "joint_skip"

        section = data.get(main_key, {})
        if isinstance(section, dict):
            for iter_key, iter_data in section.items():
                merged[iter_key] = iter_data
                total_iters += 1

        # Collect provenance
        prov = data.get("provenance", {})
        if isinstance(prov, dict):
            prov["_sub_batch"] = sd.name
            provenance_parts.append(prov)

    result = {
        main_key: merged,
        "provenance": {
            "merged_from": [sd.name for sd in sub_dirs],
            "total_iterations": total_iters,
            "sub_batch_provenances": provenance_parts,
        },
    }
    save_json(out_path, result)
    print(f"  Merged {key}: {total_iters} iterations from {len(sub_dirs)} sub-batches")


def merge_npz(sub_dirs, out_path):
    """Merge pair_records.npz across sub-batches."""
    all_arrays = defaultdict(list)
    found = False

    for sd in sub_dirs:
        path = sd / "pair_records.npz"
        if not path.exists():
            print(f"  [WARN] {path} not found, skipping NPZ")
            continue
        found = True
        data = np.load(path, allow_pickle=True)
        for k in data.files:
            all_arrays[k].append(data[k])

    if not found:
        print(f"  [WARN] No NPZ files found, skipping")
        return

    # Concatenate arrays
    merged = {}
    for k, arrays in all_arrays.items():
        if all(a.shape == arrays[0].shape for a in arrays):
            try:
                merged[k] = np.concatenate(arrays, axis=0)
            except Exception:
                merged[k] = arrays[0]  # keep first if can't concatenate
        else:
            merged[k] = arrays[0]

    np.savez(out_path, **merged)
    print(f"  Merged pair_records.npz: {len(merged)} arrays")


def main():
    print("=== R3.1 Sub-Batch Merger ===")
    for window, sub_names in WINDOWS.items():
        print(f"\n--- Merging window {window} ---")
        sub_dirs = [PARALLEL_DIR / sn for sn in sub_names]
        out_dir = MERGED_DIR / window
        out_dir.mkdir(parents=True, exist_ok=True)

        # Check all sub-batches completed
        all_ready = True
        for sd in sub_dirs:
            corr_file = sd / "certificate_correctness.json"
            if not corr_file.exists():
                print(f"  [ERROR] {sd.name} not complete (no certificate_correctness.json)")
                all_ready = False
        if not all_ready:
            print(f"  Window {window} NOT ready, skipping")
            continue

        # Merge all JSON files
        for jf in JSON_FILES:
            merge_json(sub_dirs, jf, out_dir / jf)

        # Merge NPZ
        merge_npz(sub_dirs, out_dir / "pair_records.npz")

        # Copy runner logs
        log_combined = out_dir / "runner.log"
        with open(log_combined, "w") as f:
            for sd in sub_dirs:
                log_path = sd / "runner.log"
                if log_path.exists():
                    f.write(f"=== {sd.name} ===\n")
                    f.write(log_path.read_text())
                    f.write("\n")

        print(f"  Window {window} merged to {out_dir}")

    print("\n=== Merge complete ===")
    print(f"Output: {MERGED_DIR}")


if __name__ == "__main__":
    main()
