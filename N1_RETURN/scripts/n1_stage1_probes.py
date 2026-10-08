#!/usr/bin/env python3
"""N1 Stage 1: deterministic probe construction (frozen, geometry-only).

Construction rules (FROZEN before any probe gradient is computed):
- anchors: for each scene and checkpoint C in {15000, 20000, 25000}, the 64
  consecutive canonical camera-sequence opportunities C+1 .. C+64
  (deterministic seed-42 stream of the canonical p12a run; the N1 snapshot
  run's 25000-entry sequence save is verified prefix-identical, and anchors
  at 25000 read its true continuation 25001..25064). anchor k has
  v1 = seq[C + k - 1] (0-based array), k = 1..64.
- held-out set: 8 cameras chosen FIRST from the complement of the union of
  all anchor views (all 3 checkpoints), even-stride deterministic:
  heldout[k] = candidates[(k * len(candidates)) // 8].
- partner pool per anchor (7 partners): all valid cameras except {v1} union
  heldout, ranked by the frozen geometric-overlap score
      s(a,b) = cos_dir(a,b) / (1 + d(a,b)/d_ref)
  with cos_dir = dot(unit view dir a, unit view dir b),
  view dir = -viewmatrix[2, 0:3] (world-space viewing direction),
  d = ||camera_center_a - camera_center_b||,
  d_ref = median pairwise camera-center distance of the scene.
  Sort descending, ties broken by camera index ascending.
  high   = ranks 1..3        (indices 0,1,2)
  medium = ranks at sorted positions len//3 and 2*len//3
  low    = the two lowest    (indices len-2, len-1)
  Partners are selected ONLY from camera geometry (no gradients, no renders).
- A2-random partner: one draw per anchor from a per-scene frozen RNG stream
  random.Random(N1_SEED) over that scene's anchors in the frozen order
  (checkpoint ascending, anchor index ascending), partner = pool[rng.randint(0, 6)].
- per-anchor historical features from the canonical sequence:
  backward_age = C - (last position p <= C with seq[p-1] == v1)
  count_last_512 = #{p in [C-511, C]} : seq[p-1] == v1  (1-based positions)

Writes N1_PROBE_LIST.json.
"""
import json
import math
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import n1_core as C

REPO = C.REPO
sys.path.insert(0, os.path.join(REPO, "src"))  # benchmark_framework


def load_valid_cameras(scene):
    """Replicate GTDataset's camera matching WITHOUT loading images."""
    from pathlib import Path
    from benchmark_framework import load_cameras_from_json
    official = Path(REPO) / "data" / "official" / "mipnerf360" / scene
    gt_dir = Path(REPO) / "data" / "datasets" / "mipnerf360" / scene / "images"
    cams = load_cameras_from_json(str(official / "cameras.json"), device="cpu")
    image_index = {f.stem for f in gt_dir.iterdir()
                   if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    valid = []
    for cam in cams:
        name = getattr(cam, "image_name", None)
        if name is None:
            name = getattr(cam, "img_name", None)
        if name and Path(name).stem in image_index:
            valid.append(cam)
    return valid


def camera_geometry(cams):
    centers = np.stack([c.camera_center.detach().cpu().numpy().reshape(-1)[:3]
                        for c in cams])
    # world viewing direction = -(R^T e_z) = -viewmatrix[2, 0:3]
    dirs = np.stack([-c.viewmatrix.detach().cpu().numpy()[2, :3] for c in cams])
    dirs = dirs / np.clip(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12, None)
    return centers, dirs


def median_pairwise_distance(centers):
    n = len(centers)
    diffs = []
    for i in range(n):
        d = np.linalg.norm(centers[i + 1:] - centers[i], axis=1)
        diffs.append(d)
    alld = np.concatenate(diffs) if diffs else np.array([1.0])
    return float(np.median(alld))


def overlap_scores(centers, dirs, d_ref, a, others):
    cos_dir = dirs[others] @ dirs[a]
    d = np.linalg.norm(centers[others] - centers[a], axis=1)
    return cos_dir / (1.0 + d / d_ref), cos_dir, d


def build_scene(scene):
    cams = load_valid_cameras(scene)
    n = len(cams)
    centers, dirs = camera_geometry(cams)
    d_ref = median_pairwise_distance(centers)

    # canonical 30000-long sequence (prefix-verified against the snapshot
    # run's 25000-long save); anchors at 25000 need positions 25001..25064,
    # which exist only in the canonical continuation. Both are the same
    # deterministic seed-42 stream.
    seq = np.load(os.path.join(C.CANONICAL_RUNS[scene], "camera_sequence.npy"))
    snap = np.load(os.path.join(C.SNAP_RUNS[scene], "camera_sequence.npy"))
    assert len(snap) == 25000 and (snap == seq[:25000]).all(), \
        f"{scene}: snapshot camera sequence is not a prefix of the canonical"

    # anchor views
    anchors = []
    v1_union = set()
    for Ck in C.ANCHOR_CKPTS:
        for k in range(1, C.N_ANCHORS + 1):
            v1 = int(seq[Ck + k - 1])
            anchors.append({"checkpoint": Ck, "anchor_index": k,
                            "iteration": Ck + k, "v1": v1})
            v1_union.add(v1)

    # held-out: 8 cameras from the complement of the anchor-view union
    cand = [i for i in range(n) if i not in v1_union]
    assert len(cand) >= C.HELDOUT_N, \
        f"{scene}: only {len(cand)} non-anchor cameras for held-out set"
    heldout = [cand[(k * len(cand)) // C.HELDOUT_N] for k in range(C.HELDOUT_N)]
    assert len(set(heldout)) == C.HELDOUT_N
    heldout_set = set(heldout)

    # historical sequence features per anchor
    for a in anchors:
        v1, Ck = a["v1"], a["checkpoint"]
        positions = np.nonzero(seq[:Ck] == v1)[0] + 1  # 1-based positions <= Ck
        a["backward_age"] = int(Ck - (int(positions[-1]) if len(positions) else 0))
        lo = max(1, Ck - 511)
        a["count_backwarded_last_512"] = int(
            ((positions >= lo) & (positions <= Ck)).sum())

    # partner pools (geometry only)
    for a in anchors:
        v1 = a["v1"]
        others = [i for i in range(n) if i != v1 and i not in heldout_set]
        s, cosd, d = overlap_scores(centers, dirs, d_ref, v1, others)
        order = sorted(range(len(others)), key=lambda j: (-s[j], others[j]))
        L = len(order)
        pool_idx = ([others[order[0]], others[order[1]], others[order[2]]] +
                    [others[order[L // 3]], others[order[(2 * L) // 3]]] +
                    [others[order[L - 2]], others[order[L - 1]]])
        tiers = (["high", "high", "high", "medium", "medium", "low", "low"])
        a["pool"] = [{"partner": int(p), "tier": t,
                      "overlap_score": float(s[others.index(p)]),
                      "cos_dir": float(cosd[others.index(p)]),
                      "dist": float(d[others.index(p)])}
                     for p, t in zip(pool_idx, tiers)]
        assert len({int(p) for p in pool_idx}) == 7

    # A2-random partner: one frozen RNG stream over the frozen anchor order
    rng = random.Random(C.N1_SEED)
    for a in anchors:  # anchors already in frozen order (ckpt asc, idx asc)
        a["a2_random_partner"] = a["pool"][rng.randint(0, 6)]["partner"]

    # per-checkpoint union of probe views for the sketch/grad passes:
    # views(S) = views of anchors AT S  (fresh sketches, current grads)
    #          U views of anchors AT S+5000 (stale-sketch source)
    union = {}
    for S in C.CKPTS:
        u = set()
        for a in anchors:
            if a["checkpoint"] == S or a["checkpoint"] == S + 5000:
                u.add(a["v1"])
                u.update(p["partner"] for p in a["pool"])
        union[str(S)] = sorted(u)

    return {
        "scene": scene,
        "n_cameras": n,
        "d_ref_median_pairwise_dist": d_ref,
        "camera_sequence_len": int(len(seq)),
        "camera_sequence_source": "canonical p12a run (30000, deterministic seed-42 "
                                  "stream; snapshot run's 25000-entry save verified "
                                  "prefix-identical)",
        "heldout_cameras": [int(h) for h in heldout],
        "n_anchors": len(anchors),
        "anchors": anchors,
        "probe_view_union_by_checkpoint": union,
    }


def main():
    out = {
        "campaign": "N1_ReBudgetGS_mechanism_measurement",
        "stage": "1_probe_list",
        "frozen_date": "2026-09-28",
        "spec": {
            "anchors_per_checkpoint": C.N_ANCHORS,
            "anchor_checkpoints": list(C.ANCHOR_CKPTS),
            "anchor_rule": "v1 = camera_sequence[C + k - 1] for k=1..64 "
                           "(canonical seed-42 sequence; positions C+1..C+64)",
            "heldout_rule": "8 cameras, even stride over sorted complement of "
                            "the anchor-view union; excludes all action views",
            "pool_rule": "7 partners by geometric overlap s=cos_dir/(1+d/d_ref), "
                         "ties by camera index; high=ranks 1-3, "
                         "medium=positions len//3 and 2len//3, low=two lowest",
            "a2_random_rule": "per-scene frozen RNG stream random.Random(N1_SEED) "
                              "over that scene's anchors in frozen order "
                              "(checkpoint asc, anchor index asc), randint(0,6)",
            "backward_age_rule": "C - last 1-based position <= C with that view",
            "count_512_rule": "occurrences in 1-based positions [C-511, C]",
            "geometry_only": "no gradients or renders used for any selection",
            "a1_replicas": C.A1_REPLICAS,
            "stale_sketch_source": {str(k): v for k, v in C.SKETCH_SRC.items()},
        },
        "n1_seed": C.N1_SEED,
        "scenes": {},
    }
    for scene in C.SCENES:
        print(f"[stage1] building {scene} ...")
        out["scenes"][scene] = build_scene(scene)
        sc = out["scenes"][scene]
        print(f"  cams={sc['n_cameras']} anchors={sc['n_anchors']} "
              f"heldout={sc['heldout_cameras']} "
              f"union_sizes={ {k: len(v) for k, v in sc['probe_view_union_by_checkpoint'].items()} }")
        # sanity: heldout disjoint from every action view
        hs = set(sc["heldout_cameras"])
        for a in sc["anchors"]:
            assert a["v1"] not in hs
            for p in a["pool"]:
                assert p["partner"] not in hs
    C.json_dump(out, os.path.join(C.ARTIFACTS, "N1_PROBE_LIST.json"))
    print("[stage1] N1_PROBE_LIST.json written BEFORE any probe gradient.")


if __name__ == "__main__":
    main()
