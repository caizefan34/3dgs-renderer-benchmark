"""SecantGS S0 finalizer.

Aggregates per-cell JSONs into S0_SUMMARY.json (S0_SCHEMA v1 layout), runs the
frozen packet analyzer (s0_analyze.py) to produce S0_VERDICT.json, and writes
S0_REPORT.md. The analyzer owns the verdict; this script only assembles inputs.
"""
import json
import os
import subprocess
import sys
import hashlib
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PACKET_ANALYZER = "/root/epic05_sg0/s0_analyze.py"   # frozen packet analyzer (copied)
CELLS = ["room@10000", "room@15000", "bicycle@10000", "bicycle@15000"]
MS = ("2", "4")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    cells_dir = os.path.join(HERE, "cells")
    out_dir = os.path.join(HERE, "out")
    os.makedirs(out_dir, exist_ok=True)

    summary = {"schema_version": "S0-v1", "cells": {}}
    for cell in CELLS:
        cp = os.path.join(cells_dir, f"S0_CELL_{cell}.json")
        if not os.path.exists(cp):
            raise SystemExit(f"missing per-cell file: {cp}")
        c = load(cp)
        summary["cells"][cell] = {"memory": {}}
        for m in MS:
            src = c[m]
            summary["cells"][cell]["memory"][m] = {
                "valid_source_descent_count": int(src["valid_source_descent_count"]),
                "E_span_median": src["E_span_median"],
                "E_span_p10": src["E_span_p10"],
                "P_src_median": src["P_src_median"],
                "P_src_p10": src["P_src_p10"],
                "peer_loss_win_rate": src["peer_loss_win_rate"],
                "truehold_psnr_delta_median_db": src["thold_psnr_delta_median_db"],
                "truehold_psnr_delta_p10_db": src["thold_psnr_delta_p10_db"],
                "invalid_fraction": src["invalid_fraction"],
            }

    summary_path = os.path.join(out_dir, "S0_SUMMARY.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote", summary_path)

    # run the frozen packet analyzer
    subprocess.run([sys.executable, PACKET_ANALYZER, summary_path], check=True)
    verdict_path = os.path.join(out_dir, "S0_VERDICT.json")
    if not os.path.exists(verdict_path):
        # analyzer writes next to the summary (out/) -- confirm
        alt = os.path.join(out_dir, "S0_VERDICT.json")
        raise SystemExit(f"analyzer did not produce {verdict_path}")
    print("verdict:", load(verdict_path))

    # SHA256 manifest of all outputs
    manifest = {}
    for fn in sorted(os.listdir(out_dir)):
        fp = os.path.join(out_dir, fn)
        if os.path.isfile(fp):
            manifest[fn] = {"bytes": os.path.getsize(fp), "sha256": sha256_file(fp)}
    with open(os.path.join(out_dir, "S0_MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("wrote S0_MANIFEST.json with", len(manifest), "entries")


if __name__ == "__main__":
    main()
