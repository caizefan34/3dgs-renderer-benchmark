#!/usr/bin/env python3
"""MG0 Stage 1 — freeze deterministic training-camera sets (CPU only).

Frozen rule (prereg §2): 64 cameras per scene, spread uniformly over the
canonical training-camera ordering (GTDataset _valid_cameras order ==
cameras.json order, validated in Stage 0):

    j_k = floor((k + 0.5) * N / 64),  k = 0..63

Also frozen here (for §9, BEFORE any gradient is computed):
- 32 deterministic anchor views: quantile rule (n=32) over the 64-list
- 16-camera disjoint internal training-view evaluation set: quantile rule
  (n=16) over the remaining 32 (even-position) views

The selection rule is purely structural (index-space quantiles); no gradient
information was used or inspected at freeze time.
"""
import hashlib
import json
import os
import sys

import torch  # noqa: F401  (import parity with the rest of the suite)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mg0_core as C

CAMERAS_JSON = os.path.join(C.REPO, "data", "official", "mipnerf360", "{scene}",
                            "cameras.json")


def main():
    out = {"campaign": "MG0_OneView_GS_HotCV",
           "stage": "1_frozen_camera_sets", "frozen_date": "2026-09-29",
           "rule": "j_k = floor((k+0.5)*N/n) over the canonical training-camera "
                   "ordering (GTDataset index space = cameras.json order)",
           "n_views": C.N_VIEWS,
           "note": "identical 64 cameras used at 15K/20K/25K; no formal test "
                   "views; no camera replaced after gradients are inspected",
           "scenes": {}}
    for scene in C.SCENES:
        cams = json.load(open(CAMERAS_JSON.format(scene=scene)))
        N = len(cams)
        names = [str(c.get("img_name") or c.get("image_name")) for c in cams]
        ids = C.quantile_indices(N, C.N_VIEWS)
        assert len(set(ids)) == C.N_VIEWS, f"quantile rule not injective: {scene}"

        anchors = [ids[k] for k in C.quantile_indices(C.N_VIEWS, 32)]
        anchors_set = set(anchors)
        rest = [i for i in ids if i not in anchors_set]
        assert len(rest) == 32
        internal = [rest[k] for k in C.quantile_indices(len(rest), 16)]
        assert not (set(internal) & anchors_set)
        assert len(set(internal)) == 16

        out["scenes"][scene] = {
            "N_cameras": N,
            "camera_ids": ids,
            "camera_names": [names[i] for i in ids],
            "anchors_32": anchors,
            "anchors_32_names": [names[i] for i in anchors],
            "internal_eval_16": internal,
            "internal_eval_16_names": [names[i] for i in internal],
            "cameras_json_sha256": hashlib.sha256(
                open(CAMERAS_JSON.format(scene=scene), "rb").read()).hexdigest(),
        }
        print(f"[stage1] {scene}: N={N} ids={ids[:6]}... anchors={anchors[:4]}... "
              f"internal={internal[:4]}...", flush=True)

    out["script_sha256"] = {"mg0_stage1_cameras.py": C.sha256_file(os.path.abspath(__file__)),
                            "mg0_core.py": C.sha256_file(os.path.join(
                                os.path.dirname(os.path.abspath(__file__)), "mg0_core.py"))}
    C.json_dump(out, os.path.join(C.ARTIFACTS, "MG0_CAMERA_SET.json"))


if __name__ == "__main__":
    main()
