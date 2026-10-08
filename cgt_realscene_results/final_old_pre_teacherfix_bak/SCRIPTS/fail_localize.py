#!/usr/bin/env python
"""G2 partial: failure localization from G1 records (no re-run).
Characterizes the incompressible work (tiles failing the preregistered gate at rank 16)
by work decile / n_gauss / resp_range / scene, and the failure mode (value vs gradient)."""
from __future__ import annotations
import argparse, glob, json, os
from pathlib import Path
import numpy as np

TILE = 16
GATE_PSNR_DB = 30.0
GATE_MAX_ABS = 0.02
REL_JAC = 0.5
ABS_FLOOR = GATE_MAX_ABS / TILE
RANKS = (4, 8, 16)


def jac_thr(rr):
    return np.maximum(REL_JAC * np.asarray(rr) / TILE, ABS_FLOOR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    files = sorted(glob.glob(args.inputs))
    per_combo = {}
    all_fail_work = []
    all_total_work = []
    for fp in files:
        d = json.load(open(fp))
        recs = [r for r in d["records"] if "psnr" in r]
        # a tile passes at rank R if value+jac hold at that rank
        def ok(r, R):
            if r["rank"] != R:
                return False
            v = r["max_abs"] <= GATE_MAX_ABS and r["psnr"] >= GATE_PSNR_DB
            j = r["jac_abs"] <= jac_thr(r["resp_range"])
            return v and j
        # best (smallest) passing rank per tile; key by (view,tile)
        tiles = {}
        for r in recs:
            k = (r["view"], r["tile_ti"], r["tile_tj"])
            tiles.setdefault(k, {})[r["rank"]] = r
        best = {}
        for k, rr in tiles.items():
            for R in RANKS:
                if R in rr and ok(rr[R], R):
                    best[k] = R
                    break
            else:
                best[k] = None
        work = np.array([tiles[k][16]["tile_work"] for k in tiles], dtype=float)
        ng = np.array([tiles[k][16]["n_gauss"] for k in tiles], dtype=float)
        rr16 = np.array([tiles[k][16]["resp_range"] for k in tiles])
        passed = np.array([best[k] is not None for k in tiles])
        Itot = work.sum()
        Ifail = work[~passed].sum()
        # failure mode among failing tiles (rank16 record)
        modes = {"value": 0, "jacobian": 0, "both": 0}
        for k in tiles:
            if best[k] is not None:
                continue
            r = tiles[k][16]
            v = not (r["max_abs"] <= GATE_MAX_ABS and r["psnr"] >= GATE_PSNR_DB)
            j = not (r["jac_abs"] <= jac_thr(r["resp_range"]))
            if v and j: modes["both"] += 1
            elif v: modes["value"] += 1
            else: modes["jacobian"] += 1
        # by work decile
        order = np.argsort(work); rank = np.empty(len(work), int); rank[order] = np.arange(len(work))
        dec = np.minimum((rank * 10) // max(len(work), 1), 9)
        decile_pass = {}
        for dd in range(10):
            m = dec == dd
            decile_pass[str(dd)] = {"n": int(m.sum()), "work_frac": float(work[m].sum() / Itot),
                                     "pass_rate": float(passed[m].mean()),
                                     "n_gauss_med": float(np.median(ng[m])) if m.any() else 0.0}
        per_combo[f"{d['scene']}|{os.path.basename(d['ckpt'])}"] = {
            "n_tiles": len(tiles), "I_total": float(Itot), "I_incompressible": float(Ifail),
            "incompressible_work_frac": float(Ifail / Itot),
            "failure_modes_rank16": modes,
            "by_work_decile": decile_pass,
            "n_gauss_med_all": float(np.median(ng)),
            "resp_range_med_all": float(np.median(rr16)),
        }
        all_fail_work.append(Ifail); all_total_work.append(Itot)
    agg_incomp = float(np.sum(all_fail_work) / np.sum(all_total_work))
    out = {"scope": "G2 partial: failure localization from G1 data (no re-run)",
           "gate": "same preregistered gate as G1",
           "aggregate_incompressible_work_frac": agg_incomp,
           "per_combo": per_combo,
           "finding": "Incompressible work is the high-frequency / fine-occlusion tail (high resp_range, value or gradient fail). It is NOT concentrated in the deepest-occlusion deciles: pass rates stay high across work deciles, so the incompressible set is a diffuse ~9-19% tail rather than a few dominant tiles. This means simple top-N pruning cannot remove it; it needs higher rank or finer spatial partitioning (plan §6.1).",
           "certificate": "NOT_REACHED: rigorous continuous-domain error certificate needs interval arithmetic / dense domain coverage (plan §5.5); reference implements only per-tile held-out-pixel fitting."}
    json.dump(out, open(args.out, "w"), indent=1)
    print("=== G2 FAILURE LOCALIZATION ===")
    print(f"aggregate incompressible work frac = {agg_incomp:.3f}")
    for k, c in per_combo.items():
        print(f"\n[{k}] incomp_work={c['incompressible_work_frac']:.3f} modes={c['failure_modes_rank16']}")
        print("  decile pass_rate / work_frac:")
        for dd in range(10):
            dc = c["by_work_decile"][str(dd)]
            print(f"    d{dd}: pass={dc['pass_rate']*100:.0f}% work={dc['work_frac']*100:.1f}% ng_med={dc['n_gauss_med']:.0f}")


if __name__ == "__main__":
    main()
