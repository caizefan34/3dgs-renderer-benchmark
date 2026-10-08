#!/usr/bin/env python3
"""P11 Gate T: current-main active-interaction sanity (measurement only).

Verifies whether old C28 conclusions still hold on gsplat main 512d366b.
Uses existing metadata / contribution-count utilities (NO renderer change):
  - tile candidate intersections (n_isects, per-tile lengths);
  - pixels' last_ids (equivalently: per-pixel num_contributing, since the
    num-contributing utility replays the exact forward accumulate walk;
    validated against rasterization's render_alphas);
  - estimated interactions actually traversed by current backward
    (warp-uniform truncation at warp_bin_final);
  - exact contributing interactions (sum of per-pixel num_contributing, plus a
    rasterize_contributing_gaussian_ids cross-check on one camera);
  - memory lower bound for a compact (gaussian_id, pixel_id) worklist.

No worklist is built. REOPEN_TAIL only if a newly measured >=3% E2E conservative
opportunity exists that last_ids/warp termination does not explain.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/root/epic05_p11")
import p11_common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default="/root/epic05_p11/dumps/P11_ACTIVE_INTERACTION_AUDIT.json")
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    torch.manual_seed(0)

    C.bootstrap_gmain()
    ns = C.import_recipe()
    gpu_rec = C.assert_clean_gpu(args.gpu)
    from gsplat import rasterization
    from gsplat.cuda._wrapper import (
        rasterize_num_contributing_gaussians, rasterize_contributing_gaussian_ids)
    from dataset import GTDataset

    per_scene = {}
    for scene in C.SCENES:
        dataset = GTDataset(scene=scene, repo_root=C.REPO, resolution="1080p",
                            device="cuda", background="black")
        seq = C.camera_sequence_for(scene)
        cells = {}

        state_list = [("late_30k", os.path.join(
            C.FULL30K, f"gmain03_s42_{scene}", "ckpt_30000.pt"), int(seq[0]))]
        sp10k = os.path.join(C.STATEGEN_DIR, scene, "state_10000.pt")
        if os.path.exists(sp10k):
            state_list.append(("pre15k_10000", sp10k, int(seq[10000 % len(seq)])))

        for label, state_path, cam_idx in state_list:
            model, _, meta = C.load_model_for_dataset(scene, dataset, state_path=state_path)
            cam, gt = dataset.get_item(cam_idx)
            W_, H_ = int(cam.image_width), int(cam.image_height)

            with torch.no_grad():
                r, render_alphas, m = rasterization(
                    means=model.get_xyz, quats=model.get_rotation,
                    scales=model.get_scaling, opacities=model.get_opacity,
                    colors=model.get_features,
                    viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
                    width=W_, height=H_,
                    tile_size=16, packed=False, sh_degree=model.active_sh_degree,
                    radius_clip=0.0, eps2d=C.EPS2D, render_mode="RGB",
                    absgrad=False)

                n_isects = int(m["flatten_ids"].numel())
                # gsplat's isect_offsets are EXCLUSIVE-START prefix sums per
                # tile (intersect_offset_kernel: offsets[t] = start of tile t's
                # segment; empty tiles repeat the next nonempty start). The
                # last tile's pairs are NOT reflected in off[-1] — its segment
                # ends at n_isects. Per-tile count = start[t+1] - start[t],
                # with the virtual start[n_tiles] = n_isects.
                isect_offsets = m["isect_offsets"]  # [C, TH, TW] start offsets
                off = isect_offsets.flatten()
                ends = torch.cat([off[1:],
                                  torch.full_like(off[:1], n_isects)])
                tiles_counts = (ends - off)
                assert int(tiles_counts.sum()) == n_isects, \
                    (f"per-tile counts total {int(tiles_counts.sum())} != "
                     f"n_isects {n_isects}")
                assert int(tiles_counts.min()) >= 0
                n_tiles_nonzero = int((tiles_counts > 0).sum())

                num_contrib, alphas_util = rasterize_num_contributing_gaussians(
                    m["means2d"], m["conics"], m["opacities"],
                    m["isect_offsets"], m["flatten_ids"], W_, H_, 16)
                # validation: utility replays the same walk as the forward.
                # The utility kernel is an FP-reordered accumulation of the
                # identical front-to-back walk: individually bit-deterministic
                # (repeat calls differ by 0), but vs the render kernel a small
                # fraction of pixels differ in the last bits (measured on
                # room@30K: 732/2.07M pixels, max_abs 3.9e-4, rel ~5e-6).
                # Tolerance documents FP reassociation, not semantic drift.
                ra_fwd = render_alphas[0, :, :, 0]
                ra_util = alphas_util[0]
                alpha_diff = (ra_fwd - ra_util).abs()
                max_alpha_diff = float(alpha_diff.max())
                mismatch_fraction = float((alpha_diff > 1e-3).sum()) / alpha_diff.numel()
                alpha_match = bool(max_alpha_diff <= 1e-3 and
                                   mismatch_fraction <= 1e-3)

                nc = num_contrib[0]  # [H, W]
                total_contrib = int(nc.sum())
                # last_ids[p] = nc[p] - 1 (the walk accumulates exactly the
                # contributors; the last accumulate position is nc-1)
                # warp structure inside a 16x16 tile: 256 threads row-major ->
                # warp = 32 consecutive in-tile pixels = 2 rows of 16.
                # H/W may not be tile-aligned (1080 = 67.5*16): pad to tiles;
                # outside pixels have bin_final=0 in the kernel, so padding 0
                # matches kernel behavior for warp-max purposes.
                TH, TW = int(m["tile_height"]), int(m["tile_width"])
                P = torch.zeros(TH * 16, TW * 16, dtype=nc.dtype, device=nc.device)
                P[:H_, :W_] = nc
                nc_warp = P.reshape(TH, 16, TW, 16).permute(0, 2, 1, 3).reshape(
                    TH, TW, 8, 32)  # 8 warps per tile
                warp_max = nc_warp.max(dim=-1).values  # [TH, TW, 8]
                warp_traversed = int(((warp_max + 1) * 32).sum())
                # per-pixel exact traversed (bin_final+1 per pixel); outside
                # pixels contribute nothing
                pixel_traversed = int((nc + 1).sum())
                # full-list loads (whole tile list is staged per tile)
                full_list_loads = int(tiles_counts.sum())

                # compact worklist memory lower bound (gaussian_id, pixel_id)
                wl_bytes_pair32 = total_contrib * 8
                wl_bytes_packed = total_contrib * 5  # 32-bit id + 8-bit subpixel? no:
                # honest lower bounds: two int32 = 8B/entry; single int64 = 8B;
                # theoretical minimum (pixel-major, ids varint) not lower-bounded
                # below 4B/entry for dense ids
                wl_bytes_min4 = total_contrib * 4

                # backward-kernel time from L3 dump (external-adjoint cell)
                rbwd_ms = None
                titer_ms = None
                l3path = "/root/epic05_p11/dumps/P11_L3_RENDERER_BWD.json"
                if os.path.exists(l3path):
                    try:
                        l3 = json.load(open(l3path))
                        key = "late_30k" if label == "late_30k" else "pre15k_10000"
                        cell = l3["per_scene"][scene][key]
                        per_cam = cell["per_camera"][0]
                        rbwd_ms = per_cam["renderer_backward_external_adjoint"]["median_of_medians_ms"]
                        titer_ms = cell["titer"]["T_iter_ms"]
                    except Exception:
                        pass

                # residual opportunity estimate (conservative):
                # waste vs per-pixel-exact = warp slack; waste vs contributing =
                # non-contributing positions inside the traversed range.
                warp_slack = warp_traversed - pixel_traversed
                noncontrib_in_range = pixel_traversed - total_contrib
                # conservative removable: assume ALL traversed iterations cost
                # equally and compute dominates; count only the warp slack +
                # in-range non-contributors as removable FRACTION of the raster
                # bwd kernel. This generously assumes zero cost for building the
                # compact list (which is why it is an upper bound, then compared
                # to the 3% REOPEN threshold).
                frac_warp_slack = warp_slack / max(pixel_traversed, 1)
                frac_noncontrib = noncontrib_in_range / max(pixel_traversed, 1)
                cons_removable_ms = None
                cons_removable_pct = None
                if rbwd_ms and titer_ms:
                    cons_removable_ms = rbwd_ms * (frac_warp_slack + frac_noncontrib)
                    cons_removable_pct = cons_removable_ms / titer_ms * 100.0

                cells[label] = {
                    "state": {"source": state_path, "N": meta["state_N"],
                              "iteration": meta["state_iteration"]},
                    "camera_index": cam_idx, "image_wh": [W_, H_],
                    "n_gaussians": int(model._xyz.shape[0]),
                    "n_visible_radii_pos": int((m["radii"] > 0).any(dim=-1).sum()),
                    "tile_candidate_intersections": n_isects,
                    "n_tiles_nonzero": n_tiles_nonzero,
                    "mean_tile_len": n_isects / max(n_tiles_nonzero, 1),
                    "max_tile_len": int(tiles_counts.max()),
                    "exact_contributing_interactions": total_contrib,
                    "pixel_traversed_by_bwd_est": pixel_traversed,
                    "warp_traversed_by_bwd_est": warp_traversed,
                    "full_list_loads_staged": full_list_loads,
                    "warp_slack_over_pixel_exact": warp_slack,
                    "noncontributing_in_traversed_range": noncontrib_in_range,
                    "frac_warp_slack": frac_warp_slack,
                    "frac_noncontrib_in_range": frac_noncontrib,
                    "validation_alphas_match_forward": alpha_match,
                    "validation_max_alpha_diff": max_alpha_diff,
                    "validation_mismatch_fraction_gt_1e-3": mismatch_fraction,
                    "validation_max_alpha_diff": max_alpha_diff,
                    "last_ids_note": "last_ids are forward-internal (not exposed); "
                                     "per-pixel num_contributing replays the identical "
                                     "accumulate walk so last_ids[p] = nc[p]-1; "
                                     "validated via render_alphas equality",
                    "compact_worklist_memory_lower_bound_bytes": {
                        "pair_int32_8B": wl_bytes_pair32,
                        "min_4B_per_entry": wl_bytes_min4,
                    },
                    "raster_bwd_ms": rbwd_ms,
                    "T_iter_ms": titer_ms,
                    "conservative_residual_removable_ms": cons_removable_ms,
                    "conservative_residual_removable_pct_e2e": cons_removable_pct,
                }
                print(f"[{scene}/{label}] isects={n_isects:,} contrib={total_contrib:,} "
                      f"pixel_trav={pixel_traversed:,} warp_trav={warp_traversed:,} "
                      f"slack={frac_warp_slack*100:.1f}% noncontrib={frac_noncontrib*100:.1f}% "
                      f"cons_removable={cons_removable_pct:.2f}% E2E")

            # one-camera cross-check with the exact ids utility (dense path)
            if label == "late_30k":
                with torch.no_grad():
                    ids, weights = rasterize_contributing_gaussian_ids(
                        m["means2d"], m["conics"], m["opacities"],
                        m["isect_offsets"], m["flatten_ids"], W_, H_, 16, num_contrib)
                    valid = ids[0] >= 0
                    cells[label]["contributing_ids_utility_check"] = {
                        "max_num_contributing_padded": int(ids.shape[-1]),
                        "total_valid_entries": int(valid.sum()),
                        "matches_num_contributing_sum":
                            int(valid.sum()) == total_contrib,
                        "worklist_gib_at_8B": total_contrib * 8 / 2**30,
                    }
                    del ids, weights
        per_scene[scene] = cells

    # ---- T-GATE ----
    worst = None
    for scene, cells in per_scene.items():
        for label, c in cells.items():
            p = c.get("conservative_residual_removable_pct_e2e")
            if p is not None and (worst is None or p > worst[0]):
                worst = (p, scene, label)
    tgate = {
        "worst_conservative_residual_pct_e2e": worst[0] if worst else None,
        "worst_cell": f"{worst[1]}/{worst[2]}" if worst else None,
        "reopen_threshold_pct": 3.0,
        "decision": ("REOPEN_TAIL" if worst and worst[0] >= 3.0 else "KEEP_DROP"),
        "rationale": (
            "current backward already truncates compute at warp granularity "
            "(warp_bin_final) and stages whole tile lists; the conservative "
            "residual (warp slack + in-range non-contributors, costed generously "
            "as a pure fraction of the raster-bwd kernel with ZERO compact-list "
            "build cost) is below the 3% REOPEN threshold -> C28 closure holds"
            if (not worst or worst[0] < 3.0) else
            "residual exceeds threshold; requires compact-list build cost "
            "analysis before any reopen"),
    }
    record = {
        "phase": "P11 Gate T - current-main active-interaction sanity",
        "gpu": gpu_rec,
        "method_note": "measurement only; no worklist built; existing "
                       "no-grad contribution utilities used; renderer untouched",
        "per_scene": per_scene,
        "t_gate": tgate,
    }
    C.dump_json(record, args.out)


if __name__ == "__main__":
    main()
