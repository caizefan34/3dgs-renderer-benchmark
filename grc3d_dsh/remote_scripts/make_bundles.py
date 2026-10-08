#!/usr/bin/env python3
"""M1 bundle builder: canonical grc3d bundles from replay snapshots.

Camera selection is the pre-registered rule: the first two DISTINCT camera
indices in the frozen camera_sequence (never chosen by looking at results).

State provenance: snapshots are end-of-iteration model states of a real
seed-42 frozen-recipe GMAIN_FUSED run (replay_gmain_snapshots.py). Exact
bitwise reproduction of the P12A sample path is impossible (nondeterministic
backward kernels), documented in the replay comparison evidence; current and
anchor states always come from the SAME trajectory, which is what the anchored
estimator requires.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gx_common as G  # noqa: E402

KEYS = ("means", "quats", "log_scales", "opacity_logits", "sh0", "shN")


def capture_to_params(snap):
    """Audited raw-variable mapping (m0 mapping_audit):
    means<-xyz, quats<-rotation (raw), log_scales<-scaling (log),
    opacity_logits<-opacity (logit, [N]), sh0<-shs[:, :1], shN<-shs[:, 1:]."""
    shs = snap["shs"]
    return {"means": snap["xyz"].detach().float().clone(),
            "quats": snap["rotation"].detach().float().clone(),
            "log_scales": snap["scaling"].detach().float().clone(),
            "opacity_logits": snap["opacity"].detach().float().reshape(-1).clone(),
            "sh0": shs[:, :1].detach().float().clone(),
            "shN": shs[:, 1:].detach().float().clone()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--snapshot-dir", required=True)
    ap.add_argument("--camera-seq", required=True)
    ap.add_argument("--states", required=True, help="comma-separated current iterations")
    ap.add_argument("--lags", default="0,16,64")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()

    G.add_grc3d_path()
    from grc3d.export import save_bundle

    dataset = G.build_dataset(args.scene)
    seq = np.load(args.camera_seq)
    cam_ids = []
    for v in seq:
        iv = int(v)
        if iv not in cam_ids:
            cam_ids.append(iv)
        if len(cam_ids) == 2:
            break
    print(f"[bundles] {args.scene}: pre-registered cameras {cam_ids} "
          f"(first two distinct in frozen sequence)", flush=True)

    states = [int(s) for s in args.states.split(",")]
    lags = [int(s) for s in args.lags.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)
    manifest = {"scene": args.scene, "camera_rule": "first two distinct camera indices "
                "of the frozen camera_sequence (pre-registered)",
                "cameras": cam_ids, "states": states, "lags": lags,
                "snapshot_dir": args.snapshot_dir, "bundles": {}}

    snaps = {}
    for it in sorted(set(states + [s - l for s in states for l in lags])):
        p = os.path.join(args.snapshot_dir, f"snap_{args.scene}_{it:06d}.pt")
        snaps[it] = torch.load(p, map_location="cpu", weights_only=False)
        snaps[it]["_file"] = p
        print(f"[bundles] snapshot {it}: N={snaps[it]['n_gaussians']:,} "
              f"sh_deg={snaps[it]['active_sh_degree']} loss={snaps[it]['loss']:.6f}", flush=True)

    for it in states:
        for lag in lags:
            aid = it - lag
            cur, anc = snaps[it], snaps[aid]
            assert int(cur["active_sh_degree"]) == int(anc["active_sh_degree"]), \
                "SH degree must match between current and anchor state"
            cp, ap_ = capture_to_params(cur), capture_to_params(anc)
            for ci in cam_ids:
                cam, gt = dataset.get_item(ci)
                target = gt.detach().float().cpu()
                name = f"{args.scene}_it{it}_cam{ci}_lag{lag}.pt"
                path = os.path.join(args.out_dir, name)
                meta = {"scene": args.scene, "iteration": it, "anchor_iteration": aid,
                        "lag": lag, "camera_index": ci, "seed": 42,
                        "arm": "gmain_fused (strong baseline)",
                        "current_snapshot_sha256": G.sha256_file(cur["_file"]),
                        "anchor_snapshot_sha256": G.sha256_file(anc["_file"]),
                        "current_snapshot_loss": float(cur["loss"]),
                        "anchor_snapshot_loss": float(anc["loss"]),
                        "n_gaussians_current": int(cp["means"].shape[0]),
                        "n_gaussians_anchor": int(ap_["means"].shape[0]),
                        "provenance": "REAL_BASELINE_TRAJECTORY_REPLAY "
                                      "(nondeterministic-kernel divergence from the P12A "
                                      "sample path documented in replay_comparison.json)",
                        "resolution_hw": [int(cam.image_height), int(cam.image_width)]}
                save_bundle(path, cp, ap_, cam.viewmatrix, cam.K, target,
                            int(cur["active_sh_degree"]), meta)
                manifest["bundles"][name] = {
                    "path": path, "sha256": G.sha256_file(path),
                    "camera_index": ci, "camera_identity": G.camera_identity(
                        cam, int(cam.image_width), int(cam.image_height)),
                    "iteration": it, "anchor_iteration": aid, "lag": lag,
                    "sh_degree": int(cur["active_sh_degree"]),
                    "n_gaussians_current": meta["n_gaussians_current"],
                    "n_gaussians_anchor": meta["n_gaussians_anchor"],
                    "gt_file": str(dataset._valid_image_paths[ci]),
                    "gt_file_sha256": G.sha256_file(dataset._valid_image_paths[ci]),
                    "loss_current": float(cur["loss"]), "loss_anchor": float(anc["loss"])}
                print(f"[bundles] wrote {name} "
                      f"({os.path.getsize(path)/2**20:.1f} MiB)", flush=True)
                del target
    G.write_json(args.manifest, manifest)
    print(f"[bundles] manifest -> {args.manifest}", flush=True)


if __name__ == "__main__":
    main()
