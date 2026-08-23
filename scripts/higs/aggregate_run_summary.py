"""Aggregate per-run HiGS training-benchmark JSONs into a grouped summary.

Each per-run file (written by benchmark/run_higs_train_benchmark.py --out) holds
exactly one scene; this script groups runs by explicit name/glob pairs, reports
mean +/- sd for the standard metric set, and (optionally) computes deltas and
speedups vs a reference group.

Usage:
  python scripts/higs/aggregate_run_summary.py     --out m5-summary.json     --group garden_full "results/higs-round42/m5_garden_full_*.json"     --group garden_eg_fr "results/higs-round42/m5_garden_eg035_l07_fr_*.json"     --reference garden_eg_fr garden_full
"""

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path

FIELDS = [
    "psnr", "ssim", "lpips",
    "total_ms", "train_ms", "fwd_ms", "bwd_ms",
    "peak_vram_gb", "sampled_tile_ratio", "final_n",
]


def _load(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    scenes = d.get("scenes", {})
    if len(scenes) != 1:
        raise SystemExit(f"expected exactly one scene in {path}, got {list(scenes)}")
    return d, list(scenes.values())[0][0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="output summary JSON path")
    ap.add_argument("--group", action="append", nargs=2, metavar=("NAME", "GLOB"),
                    required=True, help="group name + per-run glob (repeatable)")
    ap.add_argument("--reference", action="append", nargs=2, metavar=("NAME", "REF"),
                    default=[], help="compute delta/speedup of NAME vs REF (repeatable)")
    args = ap.parse_args()

    meta = {}
    runs = {}
    groups = {}
    for name, pat in args.group:
        files = sorted(glob.glob(pat))
        if not files:
            print(f"warning: no files matched {pat}", file=sys.stderr)
        rows = []
        for f in files:
            d, entry = _load(f)
            meta.setdefault("device", d.get("device"))
            meta.setdefault("torch", d.get("torch"))
            runs[Path(f).stem] = entry
            rows.append(entry)
        g = {"n": len(rows)}
        for k in FIELDS:
            vals = [r.get(k) for r in rows if r.get(k) is not None]
            if vals:
                g[f"{k}_mean"] = statistics.mean(vals)
                g[f"{k}_sd"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        groups[name] = g
    for name, ref in args.reference:
        if name in groups and ref in groups:
            rg, g = groups[ref], groups[name]
            for k in ("psnr", "ssim", "lpips"):
                if rg.get(f"{k}_mean") is not None and g.get(f"{k}_mean") is not None:
                    g[f"delta_{k}"] = g[f"{k}_mean"] - rg[f"{k}_mean"]
            if rg.get("total_ms_mean"):
                g["speedup"] = rg["total_ms_mean"] / g["total_ms_mean"]
            if rg.get("train_ms_mean"):
                g["speedup_train"] = rg["train_ms_mean"] / g["train_ms_mean"]

    out = {"device": meta.get("device"), "torch": meta.get("torch"),
           "runs": runs, "groups": groups}
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {args.out}: {len(groups)} groups, {len(runs)} runs")


if __name__ == "__main__":
    main()
