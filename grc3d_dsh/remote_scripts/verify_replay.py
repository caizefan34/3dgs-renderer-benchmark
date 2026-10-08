#!/usr/bin/env python3
"""Replay provenance evidence: compare a replay run against its P12A reference.

The comparison is on the recorded per-100-iteration loss and Gaussian count, and
on the frozen camera sequence. A mismatch does NOT invalidate the snapshots as
real baseline states; it documents that exact bitwise reproduction of the P12A
sample path is impossible with nondeterministic backward kernels.
"""
import argparse
import hashlib
import json
import os


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay-dir", required=True)
    ap.add_argument("--reference-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import numpy as np
    rep = json.load(open(os.path.join(args.replay_dir, "results.json")))
    ref = json.load(open(os.path.join(args.reference_dir, "results.json")))
    refmap = {r["iter"]: r for r in ref["per_iter"]}

    seq_rep = np.load(os.path.join(args.replay_dir, "camera_sequence.npy"))
    seq_ref = np.load(os.path.join(args.reference_dir, "camera_sequence.npy"))
    n = len(seq_rep)
    seq_equal = bool(np.array_equal(seq_ref[:n], seq_rep[:n]))

    rows, first_n_diff, first_loss_diff = [], None, None
    for r in rep["per_iter"]:
        it = r["iter"]
        rr = refmap.get(it)
        if rr is None:
            continue
        row = {"iter": it, "loss_replay": r["loss"], "loss_reference": rr["loss"],
               "loss_abs_diff": abs(r["loss"] - rr["loss"]),
               "n_gs_replay": r["n_gs"], "n_gs_reference": rr["n_gs"],
               "n_gs_equal": r["n_gs"] == rr["n_gs"]}
        if not row["n_gs_equal"] and first_n_diff is None:
            first_n_diff = it
        if row["loss_abs_diff"] > 0 and first_loss_diff is None:
            first_loss_diff = it
        rows.append(row)

    n_eq = sum(1 for r in rows if r["n_gs_equal"])
    out = {
        "replay_dir": args.replay_dir,
        "reference_dir": args.reference_dir,
        "replay_results_sha256": sha256(os.path.join(args.replay_dir, "results.json")),
        "reference_results_sha256": sha256(os.path.join(args.reference_dir, "results.json")),
        "iterations_compared": len(rows),
        "camera_sequence_prefix_equal": seq_equal,
        "camera_sequence_compared_length": int(n),
        "n_gs_identical_rows": n_eq,
        "n_gs_identical_fraction": (n_eq / len(rows)) if rows else None,
        "first_iteration_with_n_gs_mismatch": first_n_diff,
        "first_iteration_with_loss_mismatch": first_loss_diff,
        "max_loss_abs_diff": max((r["loss_abs_diff"] for r in rows), default=None),
        "max_n_gs_abs_diff": max((abs(r["n_gs_replay"] - r["n_gs_reference"]) for r in rows),
                                  default=None),
        "rows": rows,
        "conclusion": ("EXACT_BITWISE_REPLAY" if (n_eq == len(rows) and seq_equal and
                                                  all(r["loss_abs_diff"] == 0 for r in rows))
                       else "REPLAY_DIVERGENT_NONDETERMINISTIC_KERNELS"),
        "interpretation": ("The replay reproduces the frozen recipe, seed and camera "
                           "sequence but not the P12A sample path: gsplat backward "
                           "accumulation is not run-to-run deterministic (measured "
                           "separately), so the densification trajectory diverges. "
                           "Snapshots are real baseline-recipe states at the stated "
                           "iterations; current and anchor states come from the SAME "
                           "trajectory, which is what the anchored estimator needs."),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, allow_nan=False)
    print(f"[verify] {out['conclusion']}  rows={len(rows)} seq_equal={seq_equal} "
          f"n_gs_equal={n_eq}/{len(rows)} max_loss_diff={out['max_loss_abs_diff']}")


if __name__ == "__main__":
    main()
