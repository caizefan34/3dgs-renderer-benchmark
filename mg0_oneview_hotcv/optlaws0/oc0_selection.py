#!/usr/bin/env python3
"""OPTLAWS0 / OC0 step 1 — frozen row + camera selection.

Emits
-----
  OC0_SELECTION.json

Contents (exactly the OC0_PREREG construction):

  * cameras: every second camera of the frozen MG0 64-camera training set, in
    frozen order -> 32 cameras.  Frozen BEFORE any renderer-feature collection.
  * rows per scene/checkpoint cell:
      - hot     : the 1024 highest-V_j rows of the MG0-S frozen variance
                  ranking (stable descending argsort, ties resolved by row id);
      - uniform : 1024 rows drawn WITHOUT replacement from the nonzero-V_j rows
                  that are not already hot, seed 20260929, stored sorted
                  (the same frozen rule MG0-S used, at the OC0 count);
      - union is therefore distinct and exactly 2048 rows.

Row ids and the camera subset are frozen here and only here.  No renderer or
gradient quantity enters this file.

Usage: oc0_selection.py
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.environ.get(
    "MG0_SCRIPTS_DIR",
    "/root/epic05_mg0/scripts" if os.path.isdir("/root/epic05_mg0/scripts")
    else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")))
import mg0_core as C  # noqa: E402
import oc0_common as K  # noqa: E402


def _ids_sha(ids):
    return hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()


def select_rows(v_j):
    """(hot_ids, uniform_ids, sample_ids) from the frozen V_j vector."""
    v = torch.as_tensor(np.asarray(v_j, dtype=np.float64))
    order = torch.argsort(v, descending=True, stable=True).to(torch.int64)
    hot = order[:K.N_HOT]
    n_nonzero = int((v > 0).sum())
    hot_set = set(hot.tolist())
    pool = [int(i) for i in torch.nonzero(v > 0).flatten().tolist()
            if int(i) not in hot_set]
    if len(pool) < K.N_UNIFORM:
        raise RuntimeError(f"nonzero pool {len(pool)} < {K.N_UNIFORM}")
    rng = np.random.default_rng(K.SELECTION_SEED)
    pick = np.sort(rng.choice(np.asarray(pool, dtype=np.int64),
                              size=K.N_UNIFORM, replace=False))
    unif = torch.tensor(pick, dtype=torch.int64)
    sample = torch.cat([hot, unif])
    assert sample.numel() == K.SELECTION_SIZE
    assert len(set(sample.tolist())) == K.SELECTION_SIZE
    return hot.tolist(), unif.tolist(), sample.tolist(), n_nonzero


def main():
    t0 = time.time()
    cams_sha = C.sha256_file(K.artifact("MG0_CAMERA_SET.json"))
    cameras = {}
    out = {
        "campaign": "OPTLAWS0", "stage": "OC0_selection",
        "laws": "L3 Occlusion-Gradient Conflict Predictability",
        "prereg": "OC0_PREREG.md",
        "creation_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "camera_rule": f"every {K.CAM_STRIDE}nd camera of the frozen MG0 "
                       f"{C.N_VIEWS}-camera set, in frozen order; frozen before "
                       f"renderer-feature collection",
        "row_rule": {
            "hot": f"top-{K.N_HOT} rows of the frozen MG0-S V_j ranking "
                   f"(stable descending argsort, ties by row id)",
            "uniform": f"{K.N_UNIFORM} rows drawn without replacement from the "
                       f"nonzero-V_j rows not already hot, seed {K.SELECTION_SEED}, "
                       f"stored sorted",
            "union": "distinct, exactly "
                     f"{K.SELECTION_SIZE} rows; dedup is structural (uniform pool "
                     f"excludes the hot rows)",
        },
        "cells": {},
        "source_files": {"MG0_CAMERA_SET.json": cams_sha},
        "script_sha256": {os.path.basename(__file__):
                          C.sha256_file(os.path.abspath(__file__))},
    }

    for scene in C.SCENES:
        cam_ids = [int(v) for v in C.load_camera_set(scene)["camera_ids"]]
        assert len(cam_ids) == C.N_VIEWS, (scene, len(cam_ids))
        sub = [cam_ids[i] for i in range(0, len(cam_ids), K.CAM_STRIDE)]
        assert len(sub) == K.CAM_SUBSET_SIZE, (scene, len(sub))
        cameras[scene] = {"oc0_camera_ids": sub,
                          "positions_in_frozen_order":
                              list(range(0, len(cam_ids), K.CAM_STRIDE)),
                          "n_cameras": len(sub),
                          "ids_sha256": _ids_sha(sub)}

    for scene, stage in K.CELLS:
        npz = K.artifact(f"MG0S_VARIANCE_{scene}_{stage}.npz")
        if not os.path.exists(npz):
            raise SystemExit(f"missing {npz} -- run MG0-S first (row ids must be "
                             f"frozen before renderer-feature collection)")
        z = np.load(npz)
        v_j = z["V_j"]
        hot, unif, sample, n_nonzero = select_rows(v_j)
        free = z["sample_ids"].tolist() if "sample_ids" in z else []
        out["cells"][K.cell_tag(scene, stage)] = {
            "scene": scene, "checkpoint": stage,
            "N_rows": int(v_j.shape[0]),
            "n_nonzero_variance_rows": n_nonzero,
            "hot_ids": hot, "uniform_ids": unif, "selection_ids": sample,
            "n_hot": len(hot), "n_uniform": len(unif), "n_selection": len(sample),
            "hot_ids_sha256": _ids_sha(hot),
            "uniform_ids_sha256": _ids_sha(unif),
            "selection_ids_sha256": _ids_sha(sample),
            "oc0_camera_ids": cameras[scene]["oc0_camera_ids"],
            "source_variance_npz": os.path.basename(npz),
            "source_variance_npz_sha256": C.sha256_file(npz),
            "mg0s_frozen_sample_ids_sha256": _ids_sha(free) if free else None,
        }
        print(f"[oc0_selection] {scene}@{stage}: hot={len(hot)} "
              f"uniform={len(unif)} cams={len(cameras[scene]['oc0_camera_ids'])}",
              flush=True)

    out["cameras"] = cameras
    out["timing_s"] = time.time() - t0
    C.json_dump(out, K.artifact("OC0_SELECTION.json"))
    print(f"[oc0_selection] COMPLETE {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()