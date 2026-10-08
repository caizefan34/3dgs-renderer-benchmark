#!/usr/bin/env python3
"""P13-X pre-pilot probe: real-scale metrics at the room 30K C0_FUSED state.

Measures, on a real training state (not init state):
  (1) standard rasterization() meta sizes: N, n_visible(all/any), n_isects, and
      snapshot_from_meta NPZ write time/size (pilot budget);
  (2) renderer consistency: classic-path visible set vs the HIGS training path's
      culled visible set on the same anchor cameras (Jaccard).
Read-only with respect to the campaign repo. No training loop, no optimizer.
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch

REPO = "/root/3dgs-renderer-benchmark"
PROBE_TRAINER = "/root/epic05_p13x/publication_trainer_p13x_probe.py"
CKPT = "/root/epic05_p12/runs/p12a/p12a_c0f_s42_room/ckpt_30000.pt"
P13X_DIR = "/root/epic05_p13x"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--anchors", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(P13X_DIR, "artifacts",
                                                 "P13X_RENDERER_CROSSCHECK_room30k.json"))
    ap.add_argument("--snapshot-dir", default=os.path.join(P13X_DIR, "artifacts",
                                                           "preflight_snapshots"))
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    spec = importlib.util.spec_from_file_location("p13x_probe_trainer", PROBE_TRAINER)
    t = importlib.util.module_from_spec(spec)
    sys.modules["p13x_probe_trainer"] = t
    spec.loader.exec_module(t)
    ident = t.bootstrap_c0()
    sys.path.insert(0, P13X_DIR)
    from p13x_probe_core import snapshot_from_meta

    from gaussian_model import GaussianModel
    from config import ReferenceV1Config
    from dataset import GTDataset
    from gsplat.rendering import rasterization
    from gsplat.experimental import rasterize_gaussian_higs_dynamic
    from gsplat.experimental.render.functional.gaussian_inference import _HIGS_DYNAMIC_SCENE

    cfg = ReferenceV1Config()
    cfg.scene = "room"
    cfg.repo_root = REPO
    dataset = GTDataset(scene="room", repo_root=REPO, resolution="1080p",
                        device="cuda", background="black")
    n_cameras = len(dataset)

    lr_cfg = {"position_lr_init": cfg.position_lr_init,
              "position_lr_final": cfg.position_lr_final,
              "position_lr_delay_mult": cfg.position_lr_delay_mult,
              "position_lr_max_steps": cfg.position_lr_max_steps,
              "feature_lr": cfg.feature_lr, "opacity_lr": cfg.opacity_lr,
              "scaling_lr": cfg.scaling_lr, "rotation_lr": cfg.rotation_lr,
              "percent_dense": cfg.percent_dense}
    print(f"loading ckpt {args.ckpt}", flush=True)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = GaussianModel(max_sh_degree=3)
    model.restore(ck, lr_cfg)
    model.active_sh_degree = 3
    _HIGS_DYNAMIC_SCENE.mark_dirty()
    N = int(model._xyz.shape[0])
    print(f"restored: N={N:,} cameras={n_cameras} "
          f"ckpt_iter={ck.get('iteration')} arm={ck.get('arm')}", flush=True)

    anchors = t.p13x_anchor_indices(n_cameras, args.anchors)
    report = {
        "phase": "P13-X", "probe": "renderer_crosscheck_room30k",
        "ckpt": args.ckpt, "ckpt_iteration": ck.get("iteration"),
        "ckpt_arm": ck.get("arm"), "N": N, "n_cameras": n_cameras,
        "anchors": anchors, "binary_identity": ident, "per_anchor": [],
    }
    for aid, cam_idx in enumerate(anchors):
        cam, _gt = dataset.get_item(cam_idx)
        row = {"anchor_id": aid, "camera_index": int(cam_idx),
               "camera_sha256": t.p13x_camera_fingerprint(cam)}
        with torch.no_grad():
            t0 = time.perf_counter()
            out = rasterize_gaussian_higs_dynamic(
                model.get_xyz, model.get_rotation, model.get_scaling,
                model.get_opacity, model.get_features,
                sh_degree=model.active_sh_degree,
                viewmats=cam.viewmatrix.unsqueeze(0).unsqueeze(0),
                Ks=cam.K.unsqueeze(0).unsqueeze(0),
                width=cam.image_width, height=cam.image_height,
                backward_mode="higs_native", enable_culling=True,
                camera_model="pinhole", render_mode="RGB", eps2d=0.3)
            torch.cuda.synchronize()
            row["higs_ms"] = (time.perf_counter() - t0) * 1000
            info = out["densification_info"]
            hmeta = out["metadata"]
            higs_visible = info["visible_gaussian_ids"].detach().reshape(-1).cpu().numpy()
            row["higs_n_visible"] = int(hmeta["n_visible"])
            row["higs_n_isects"] = int(hmeta.get("n_isects", -1))
            row["higs_culling_ratio"] = float(hmeta["culling_ratio"])
            row["higs_meta_keys"] = sorted(hmeta.keys())
            del out, info, hmeta

            t0 = time.perf_counter()
            _r, _a, meta = rasterization(
                means=model.get_xyz, quats=model.get_rotation,
                scales=model.get_scaling, opacities=model.get_opacity,
                colors=model.get_features,
                viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
                width=cam.image_width, height=cam.image_height,
                tile_size=16, packed=False, sh_degree=model.active_sh_degree,
                radius_clip=0.0, eps2d=0.3, render_mode="RGB",
                rasterize_mode="classic", camera_model="pinhole")
            torch.cuda.synchronize()
            row["classic_ms"] = (time.perf_counter() - t0) * 1000
            radii2d = meta["radii"].detach().reshape(-1, 2)
            vis_all = (radii2d > 0).all(dim=-1)
            vis_any = (radii2d > 0).any(dim=-1)
            classic_visible = torch.nonzero(vis_all, as_tuple=False).reshape(-1).cpu().numpy()
            row["classic_n_visible_all"] = int(vis_all.sum())
            row["classic_n_visible_any"] = int(vis_any.sum())
            row["n_mixed_axis_radii"] = int((vis_any & ~vis_all).sum())
            row["n_isects"] = int(meta["flatten_ids"].reshape(-1).numel())
            row["isect_ids_shape"] = list(meta["isect_ids"].shape)
            row["flatten_ids_shape"] = list(meta["flatten_ids"].shape)
            row["isect_offsets_shape"] = list(meta["isect_offsets"].shape)
            row["peak_vram_gb"] = torch.cuda.max_memory_allocated() / (1024 ** 3)

            # classic vs HIGS visible set
            s_c = np.zeros(N, dtype=bool)
            s_c[classic_visible] = True
            s_h = np.zeros(N, dtype=bool)
            s_h[higs_visible[higs_visible < N]] = True
            inter = int((s_c & s_h).sum())
            union = int((s_c | s_h).sum())
            row["visible_jaccard_classic_vs_higs"] = float(inter / union) if union else 1.0
            row["classic_only"] = int((s_c & ~s_h).sum())
            row["higs_only"] = int((s_h & ~s_c).sum())

            t0 = time.perf_counter()
            probe_meta = {"radii": radii2d.min(dim=-1).values,
                          "depths": meta["depths"], "isect_offsets": meta["isect_offsets"],
                          "flatten_ids": meta["flatten_ids"], "isect_ids": meta["isect_ids"]}
            summary = snapshot_from_meta(probe_meta, iteration=30000, scene="room",
                                         anchor_id=aid, out_dir=args.snapshot_dir)
            row["snapshot_ms"] = (time.perf_counter() - t0) * 1000
            row["snapshot"] = summary
            row["snapshot_npz_bytes"] = os.path.getsize(summary["raw_npz"])
            del meta, probe_meta, _r, _a
            torch.cuda.empty_cache()
        report["per_anchor"].append(row)
        print(json.dumps(row, indent=1), flush=True)

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()
