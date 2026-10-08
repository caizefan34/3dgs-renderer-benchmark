import json
def frozen_camera_ids(n, k=8):
    return [int(round(n*(i+0.5)/k)) for i in range(k)]

# camera counts confirmed from the EPIC-05 parity run logs (GTDataset: room 311, bicycle 194)
N_CAMS = {"room": 311, "bicycle": 194}

def cyclic_offsets(pool, K):
    # prereg sec.3: "Within each fold use four fixed cyclic offsets.
    #   For K=4: one four-camera cycle. For K=8: same four-camera cycle repeated twice."
    cyc = list(pool)  # the four-camera cycle
    seq = []
    while len(seq) < K:
        seq.extend(cyc)
    return seq[:K]

folds = {}
cam_ids = {}
for scene, n in N_CAMS.items():
    ids = frozen_camera_ids(n, 8)
    cam_ids[scene] = ids
    c = {f"c{i}": ids[i] for i in range(8)}
    foldA_update = [c["c0"], c["c1"], c["c2"], c["c3"]]
    foldA_hold   = [c["c4"], c["c5"], c["c6"], c["c7"]]
    foldB_update = [c["c4"], c["c5"], c["c6"], c["c7"]]
    foldB_hold   = [c["c0"], c["c1"], c["c2"], c["c3"]]
    folds[scene] = {
        "n_cameras_total": n,
        "camera_selection": "frozen_camera_ids(n,k=8)=[round(n*(i+0.5)/8) for i in 0..7] (authoritative: /root/epic05_moonshot/a0/a0_lib.py:125)",
        "ordered_c0_c7": c,
        "foldA": {"update_pool": foldA_update, "heldout_eval": foldA_hold,
                  "K4_update_seq": cyclic_offsets(foldA_update, 4),
                  "K8_update_seq": cyclic_offsets(foldA_update, 8)},
        "foldB": {"update_pool": foldB_update, "heldout_eval": foldB_hold,
                  "K4_update_seq": cyclic_offsets(foldB_update, 4),
                  "K8_update_seq": cyclic_offsets(foldB_update, 8)},
        "n_paired_rollouts_per_family_per_K": 8,  # 2 folds x 4 cyclic offsets
    }

out = {
    "audit_type": "H0_CAMERA_FOLDS",
    "authority": "H0_PREREG.md section 3 (camera folds / rollouts) + section 12 (required artifact)",
    "camera_set": "formal MOONSHOT frozen 8-camera set (deterministic, even-stride over the canonical seed-42 camera list)",
    "selection_fn": "frozen_camera_ids(n_cams, k=8) = [round(n_cams*(i+0.5)/8) for i in range(8)]",
    "selection_source": "/root/epic05_moonshot/a0/a0_lib.py:125 (authoritative MOONSHOT definition)",
    "n_cameras_total": N_CAMS,
    "camera_ids_c0_c7": cam_ids,
    "folds": folds,
    "note": ("8 paired rollouts per (cell,family,K) = 2 folds x 4 fixed cyclic offsets. "
             "All branches in a rollout start from byte-identical model+optimizer state. "
             "This artifact is decision-independent; it does NOT require the missing 5K cells."),
}
json.dump(out, open("h0_run/H0_CAMERA_FOLDS.json","w"), indent=2, ensure_ascii=False)
print(json.dumps(out, indent=2, ensure_ascii=False))
