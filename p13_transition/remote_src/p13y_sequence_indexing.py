#!/usr/bin/env python3
"""P13-Y Y0a/Y0b: camera-sequence indexing proof + revisit distribution + plan.

Three independent evidence layers (no guessing about 0/1-based indexing):

  L1 STATIC      exact source lines of the frozen P12 trainer that GENERATE,
                 SAVE and CONSUME camera_sequence.npy (including the 1-based
                 loop header), extracted textually + line numbers + file sha.
  L2 REGENERATION the sequence is re-derived from the documented recipe
                 (random.Random(seed=42), pop-from-stack, one full refill per
                 cycle, np.int32) and compared bitwise against the .npy files
                 written by the original 30K P12 runs.
  L3 EMPIRICAL   (available after a capture replay) the capture trainer's
                 per-iteration camera log is asserted against the stored
                 sequence; parity probes tie the logged cam_idx to a bitwise
                 reproduced render in a separate process.

Y0b: adjacent same-camera revisits (camera_i, t0, t1, lag) -> full + in-window
lag distributions; 8 pairs per scene selected by lag quantiles
[0.05, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 0.98] of the IN-WINDOW pair set
(t0 >= 15000 and t1 <= 17000), tie-break (abs(lag-target), camera_id, t0, t1),
distinct pairs enforced in quantile order. Plan frozen here; Y2/Y3 must not
alter it.
"""
import argparse
import hashlib
import json
import os
import random
import sys

import numpy as np

P12_TRAINER = "/root/epic05_p12/publication_trainer_p12.py"
P12_RUNS = "/root/epic05_p12/runs/p12a/p12a_c0f_s42_{scene}/camera_sequence.npy"
ART = "/root/epic05_p13y/artifacts"
LOGS = "/root/epic05_p13y/logs"
SCENES = ["room", "bicycle", "garden"]
SEED = 42
WINDOW = (15000, 17000)
QUANTILES = [0.05, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 0.98]

STATIC_ANCHORS = [
    "viewpoint_stack = list(range(n_cameras))",
    "rng = random.Random(config.seed)",
    "for iteration in range(1, config.iterations + 1):",
    "rand_idx = rng.randint(0, len(viewpoint_stack) - 1)",
    "camera_sequence.append(viewpoint_stack.pop(rand_idx))",
    'np.save(os.path.join(args.outdir, "camera_sequence.npy"), camera_sequence)',
    "cam_idx = int(camera_sequence[iteration - 1])",
    "cam, gt_image = dataset.get_item(cam_idx)",
]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def regenerate(n_cameras, n_iterations, seed=SEED):
    viewpoint_stack = list(range(n_cameras))
    seq = []
    rng = random.Random(seed)
    for _ in range(1, n_iterations + 1):
        if not viewpoint_stack:
            viewpoint_stack = list(range(n_cameras))
        rand_idx = rng.randint(0, len(viewpoint_stack) - 1)
        seq.append(viewpoint_stack.pop(rand_idx))
    return np.array(seq, dtype=np.int32)


def static_evidence():
    src = open(P12_TRAINER).read().splitlines()
    ev = {"file": P12_TRAINER, "sha256": sha256_file(P12_TRAINER), "lines": {}}
    for a in STATIC_ANCHORS:
        hits = [i + 1 for i, ln in enumerate(src) if a in ln]
        ev["lines"][a] = {
            "line_numbers": hits,
            "source": [src[i - 1].strip() for i in hits],
        }
        if not hits:
            ev["lines"][a]["error"] = "ANCHOR_NOT_FOUND"
    return ev


def adjacent_revisits(seq):
    """All adjacent-occurrence pairs of the same camera, in time order."""
    last = {}
    out = []
    for i, c in enumerate(seq.tolist()):
        if c in last:
            out.append({"camera_id": int(c), "t0": int(last[c] + 1),
                        "t1": int(i + 1), "lag": int(i - last[c])})
        last[c] = i
    return out


def lag_quantile_table(lags):
    a = np.asarray(lags, dtype=np.float64)
    if a.size == 0:
        return {}
    qs = [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]
    return {"n": int(a.size), "min": float(a.min()), "max": float(a.max()),
            "mean": float(a.mean()), "std": float(a.std()),
            "quantiles": {str(q): float(np.quantile(a, q)) for q in qs}}


def select_pairs(pairs, window=WINDOW, quantiles=QUANTILES):
    """8 pairs by lag quantiles of the in-window pair set; frozen tie-break."""
    lo, hi = window
    inw = [p for p in pairs if p["t0"] >= lo and p["t1"] <= hi]
    lags = np.asarray([p["lag"] for p in inw], dtype=np.float64)
    targets = [float(np.quantile(lags, q)) for q in quantiles] if lags.size else []
    order = sorted(inw, key=lambda p: (p["lag"], p["camera_id"], p["t0"], p["t1"]))
    chosen, chosen_keys = [], []
    for q, tgt in zip(quantiles, targets):
        cands = sorted(order, key=lambda p: (abs(p["lag"] - tgt), p["camera_id"],
                                             p["t0"], p["t1"]))
        pick = None
        for c in cands:
            key = (c["camera_id"], c["t0"], c["t1"])
            if key not in chosen_keys:
                pick = c
                break
        entry = {"quantile": q, "lag_target": tgt, "pair": pick}
        if pick is None:
            entry["error"] = "NO_DISTINCT_PAIR_LEFT"
            entry["chosen"] = sorted(cands, key=lambda p: (abs(p["lag"] - tgt),
                                                           p["camera_id"], p["t0"], p["t1"]))[0]
        else:
            chosen_keys.append((pick["camera_id"], pick["t0"], pick["t1"]))
        chosen.append(entry)
    return inw, chosen


def empirical_update(cam_log_path, scene, run_dir=None):
    """L3: fold a capture run's per-iteration camera log into the indexing proof."""
    idx_path = f"{ART}/P13Y_SEQUENCE_INDEXING.json"
    indexing = json.load(open(idx_path))
    ps = indexing["per_scene"][scene]
    log = json.load(open(cam_log_path))
    rows = log["cam_log"]
    rec = {
        "cam_log": cam_log_path, "n_logged": len(rows),
        "run_iterations": log.get("iterations"),
        "run_camera_sequence_sha256": log.get("camera_sequence_sha256"),
        "alignment_rule": log.get("alignment_rule"),
        "alignment_holds_in_run": bool(log.get("alignment_holds")),
        "cam_log_sha256": sha256_file(cam_log_path),
    }
    if run_dir:
        npy = os.path.join(run_dir, "camera_sequence.npy")
        if os.path.exists(npy):
            run_seq = np.load(npy)
            rec["run_npy"] = npy
            rec["run_npy_sha256"] = sha256_file(npy)
            rec["run_npy_len"] = int(run_seq.shape[0])
            rec["run_npy_matches_logged_cameras"] = bool(
                all(int(run_seq[i]) == int(rows[i][1]) for i in range(len(rows))))
            p12 = np.load(ps["npy"])
            n = min(int(run_seq.shape[0]), int(p12.shape[0]))
            rec["p12_prefix_len_compared"] = n
            rec["run_npy_is_p12_prefix"] = bool(np.array_equal(run_seq[:n], p12[:n]))
    ok = bool(rec["alignment_holds_in_run"]
              and rec.get("run_npy_matches_logged_cameras")
              and rec.get("run_npy_is_p12_prefix"))
    rec["conclusion"] = (
        "camera_sequence.npy[i] is the camera of training iteration i+1 (1-based); "
        "the replay's own sequence is the frozen P12 30K prefix" if ok
        else "EMPIRICAL_CHECK_FAILED")
    ps["empirical"] = rec
    with open(idx_path, "w") as f:
        json.dump(indexing, f, indent=2)
    print(json.dumps(rec, indent=2))
    if not ok:
        sys.exit(5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--empirical", action="store_true",
                    help="L3: fold a capture run's camera log into the JSON")
    ap.add_argument("--cam-log", default=None,
                    help="path to p13y_cam_log.json (default: <run-dir>/p13y_cam_log.json)")
    ap.add_argument("--run-dir", default=None, help="capture run dir")
    ap.add_argument("--scene", default=None, choices=SCENES)
    args = ap.parse_args()
    if args.empirical:
        if not args.scene or not (args.cam_log or args.run_dir):
            sys.exit("--empirical requires --scene and --cam-log/--run-dir")
        empirical_update(args.cam_log
                         or os.path.join(args.run_dir, "p13y_cam_log.json"),
                         args.scene, args.run_dir)
        return
    os.makedirs(ART, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)

    ev = static_evidence()
    indexing = {"phase": "P13-Y Y0a", "date": "2026-09-27",
                "question": ("which training iteration does camera_sequence.npy[0] "
                             "correspond to (0-based vs 1-based)?"),
                "static_evidence": ev, "per_scene": {}}
    revisit = {}
    plans = {}

    for scene in SCENES:
        npy = P12_RUNS.format(scene=scene)
        seq = np.load(npy)
        n_cameras = int(seq.max()) + 1
        n_uniq = int(np.unique(seq).size)
        n_iterations = int(seq.shape[0])
        regen = regenerate(n_cameras, n_iterations)
        exact = bool(np.array_equal(seq, regen)) and seq.dtype == regen.dtype
        indexing["per_scene"][scene] = {
            "npy": npy, "npy_sha256": sha256_file(npy),
            "dtype": str(seq.dtype), "shape": list(seq.shape),
            "n_iterations_saved": n_iterations,
            "max_camera_id": int(seq.max()),
            "n_unique_cameras_in_seq": n_uniq,
            "n_cameras_implied": n_cameras,
            "every_camera_once_per_cycle": bool(
                all(sorted(seq[k * n_cameras:(k + 1) * n_cameras].tolist())
                    == list(range(n_cameras))
                    for k in range(n_iterations // n_cameras))),
            "regeneration": {
                "recipe": ("list(range(n_cameras)); refill when empty; "
                           "rng=random.Random(42); rand_idx=rng.randint(0,len-1); "
                           "append(pop(rand_idx)); np.array(dtype=int32)"),
                "matches_bitwise": exact,
                "diversity": int(np.unique(seq).size),
            },
            "empirical": None,
        }
        if not exact:
            indexing["per_scene"][scene]["regeneration"]["first_mismatch"] = int(
                np.flatnonzero(seq != regen)[0]) if seq.shape == regen.shape else None

        pairs = adjacent_revisits(seq)
        inw, chosen = select_pairs(pairs)
        revisit[scene] = {
            "npy": npy, "n_cameras": n_cameras, "n_iterations": n_iterations,
            "n_adjacent_revisits_full": len(pairs),
            "n_adjacent_revisits_in_window": len(inw),
            "window": list(WINDOW),
            "lag_quantile_table_in_window": lag_quantile_table(
                [p["lag"] for p in inw]),
            "lag_quantile_table_full": lag_quantile_table(
                [p["lag"] for p in pairs]),
            "pairs_in_window": inw,
        }
        endpoints = sorted(
            {e["pair"]["t0"] for e in chosen if e.get("pair")}
            | {e["pair"]["t1"] for e in chosen if e.get("pair")})
        plans[scene] = {
            "phase": "P13-Y Y0b frozen pair plan", "scene": scene,
            "date": "2026-09-27", "seed": SEED, "window": list(WINDOW),
            "selection_rule": ("8 pairs at lag quantiles "
                               "[0.05,0.15,0.30,0.45,0.60,0.75,0.90,0.98] of the "
                               "in-window adjacent-revisit lag distribution; "
                               "tie-break (abs(lag-target), camera_id, t0, t1); "
                               "distinct pairs enforced in quantile order"),
            "sequence_sha256": sha256_file(npy),
            "quantile_selection": chosen,
            "n_distinct_pairs_selected": len(
                {(c["pair"]["camera_id"], c["pair"]["t0"], c["pair"]["t1"])
                 for c in chosen if c.get("pair")}),
            "capture_iterations_union": endpoints,
            "capture_iterations_note": ("union of the t0/t1 endpoints of the 8 "
                                        "SELECTED pairs only; the full in-window "
                                        "revisit set stays in "
                                        "P13Y_REVISIT_DISTRIBUTION_{scene}.json"),
            "frozen": True,
            "note": ("FROZEN BEFORE any Y2/Y3 result was produced. Y2/Y3 must run "
                     "on ALL pairs listed here, graph-stable and graph-changed."),
        }

    with open(f"{ART}/P13Y_SEQUENCE_INDEXING.json", "w") as f:
        json.dump(indexing, f, indent=2)
    for scene in SCENES:
        with open(f"{ART}/P13Y_REVISIT_DISTRIBUTION_{scene}.json", "w") as f:
            json.dump(revisit[scene], f, indent=2)
        with open(f"{ART}/P13Y_PAIR_PLAN_{scene}.json", "w") as f:
            json.dump(plans[scene], f, indent=2)
    print(json.dumps({
        s: {"n_revisits_in_window": revisit[s]["n_adjacent_revisits_in_window"],
            "lags_selected": [c["pair"]["lag"] if c.get("pair") else None
                              for c in plans[s]["quantile_selection"]],
            "capture_iters": plans[s]["capture_iterations_union"],
            "regen_exact": indexing["per_scene"][s]["regeneration"]["matches_bitwise"],
            "static_ok": all("error" not in v for v in ev["lines"].values())}
        for s in SCENES}, indent=2))
    print("WROTE", ART)


if __name__ == "__main__":
    main()
