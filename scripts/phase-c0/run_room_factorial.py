#!/usr/bin/env python3
"""Run or aggregate the predeclared Room 5K split x Adam factorial."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OFFICIAL_SHA, environment, write_json
from experiment import train_condition


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["A", "B", "C", "D", "aggregate"], required=True)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.condition == "aggregate":
        rows = []
        full = {}
        output = Path(__file__).resolve().parents[2] / "results" / "a100" / "phase-c0"
        for condition in "ABCD":
            path = output / f"room_5k_{condition}.json"
            if not path.exists():
                rows.append({"condition": condition, "status": "MISSING"})
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            full[condition] = data
            rows.append({key: data[key] for key in (
                "condition", "status", "final_psnr", "final_ssim", "final_n",
                "mean_iter_ms", "totals",
            )})
        status = "COMPLETE" if all(row["status"] == "COMPLETE" for row in rows) else "BLOCKED_INCOMPLETE"
        effects = None
        if len(full) == 4:
            def contrast(key, nested=None):
                values = {c: (full[c][nested][key] if nested else full[c][key]) for c in "ABCD"}
                split_reset = values["B"] - values["A"]
                split_preserve = values["D"] - values["C"]
                adam_current = values["C"] - values["A"]
                adam_reference = values["D"] - values["B"]
                return {
                    "values": values,
                    "split_effect_at_reset": split_reset,
                    "split_effect_at_preserve": split_preserve,
                    "mean_split_effect": (split_reset + split_preserve) / 2,
                    "adam_effect_at_current_split": adam_current,
                    "adam_effect_at_reference_split": adam_reference,
                    "mean_adam_effect": (adam_current + adam_reference) / 2,
                    "split_x_adam_interaction": split_preserve - split_reset,
                }
            effects = {
                "final_psnr": contrast("final_psnr"), "final_ssim": contrast("final_ssim"),
                "final_n": contrast("final_n"), "mean_iter_ms": contrast("mean_iter_ms"),
                "clone": contrast("clone", "totals"), "split": contrast("split", "totals"),
                "prune": contrast("prune", "totals"),
            }
        print(write_json("room_5k_factorial.json", {
            "status": status, "official_reference_commit": OFFICIAL_SHA,
            "environment": environment(), "iterations": args.iterations,
            "design": {"A": "current split + Adam reset", "B": "reference split + Adam reset", "C": "current split + Adam preserve", "D": "reference split + Adam preserve"},
            "controlled_common_mode": "project masks/prune/opacity-reset/SH schedule; SH replacement preserves moments in every condition so the Adam factor applies only to topology transactions",
            "condition_result_files": {condition: f"room_5k_{condition}.json" for condition in "ABCD"},
            "rows": rows, "factorial_effects": effects,
        }))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    result = train_condition(args.condition, args.iterations, args.device)
    result["environment"] = environment()
    result["official_reference_commit"] = OFFICIAL_SHA
    print(write_json(f"room_5k_{args.condition}.json", result))


if __name__ == "__main__":
    main()
