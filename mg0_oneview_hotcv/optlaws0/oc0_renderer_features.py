#!/usr/bin/env python3
"""OPTLAWS0 / OC0 step 3 — renderer-state features for the frozen row/camera
pairs.

Usage: oc0_renderer_features.py <room|bicycle> <20000|25000>

For every frozen OC0 selection row (2048) x OC0 camera (32) pair this script
replays the native forward that the frozen c0 render branch already produced
(see oc0_patch_renderer.py + oc0_replay.py) and records the per-pair summaries
required by OC0_PREREG.md:

    visible, accepted_hits, mean_alpha, mean_T_before, p10_T_before,
    p50_T_before, mean_depth_percentile, mean_prior_contrib_count,
    mean_remaining_contrib_count, projected radius / footprint summary.

No quantity is approximated: a summary that cannot be derived from the native
tensors is written as null, never substituted.

Emits
-----
  OC0_RENDERER_FEATURES_{scene}_{ckpt}.parquet
      one row per (selection_pos, camera_slot); schema metadata carries the
      per-camera replay provenance and the frozen definitions.

Diagnostic only: no training, no renderer change, no loss/resolution change,
no threshold tuning.
"""
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
import oc0_replay as R  # noqa: E402


def bootstrap_oc0_worktree():
    """Point mg0_core at the instrumented OC0 copy of the frozen worktree.

    Only C0_WT changes: the compiled extensions are the SAME build-cache
    binaries the frozen run used, so the instrumentation is a Python-level,
    additive publication of tensors the native forward already computed.
    """
    assert os.path.isdir(K.OC0_WT), (
        f"missing instrumented worktree {K.OC0_WT} -- run oc0_patch_renderer.py")
    C.C0_WT = K.OC0_WT
    info = C.bootstrap_c0()
    import gsplat.experimental.render.functional.gaussian_inference as GI
    f = os.path.abspath(GI.__file__)
    if not f.startswith(os.path.abspath(K.OC0_WT)):
        raise RuntimeError(f"instrumented module not resolved from OC0 worktree: {f}")
    if not K.instrumentation_present(GI):
        raise RuntimeError("OC0 instrumentation absent from the resolved module")
    info["resolved_module"] = f
    info["worktree"] = K.OC0_WT
    return GI, info


def main():
    scene, stage = sys.argv[1], int(sys.argv[2])
    assert scene in C.SCENES and stage in (20000, 25000)
    t0 = time.time()
    import json
    par_path = K.artifact("OC0_INSTRUMENTATION_PARITY.json")
    if not os.path.exists(par_path):
        raise SystemExit("OC0_INSTRUMENTATION_PARITY.json missing -- the parity "
                         "step must run before renderer features are accepted")
    parity = json.load(open(par_path))
    if not parity.get("parity_ok"):
        raise SystemExit("parity_ok is false -- renderer features are NOT accepted "
                         "(see OC0_INSTRUMENTATION_PARITY.json)")
    with open(K.artifact("OC0_SELECTION.json")) as f:
        sel = json.load(f)
    cell = sel["cells"][K.cell_tag(scene, stage)]
    cams = [int(v) for v in cell["oc0_camera_ids"]]
    sel_ids = [int(v) for v in cell["selection_ids"]]
    ng = len(sel_ids)

    GI, binfo = bootstrap_oc0_worktree()
    dataset = C.load_scene_dataset(scene)
    ck = C.load_checkpoint(scene, stage)
    model = C.build_model(ck)
    C.scene_reset()
    del ck
    torch.cuda.empty_cache()
    sel_t = torch.tensor(sel_ids, dtype=torch.int64, device="cuda")
    print(f"[oc0_features] {scene}@{stage} N={model._xyz.shape[0]} ng={ng} "
          f"nc={len(cams)} mem={C.gpu_mem_gb():.1f}GB", flush=True)

    cols = {k: [] for k in (
        "selection_pos", "gaussian_row", "partition", "camera_slot", "camera_id",
        "visible", "accepted_hits", "mean_alpha", "mean_T_before",
        "p10_T_before", "p50_T_before", "mean_depth_percentile",
        "mean_prior_contrib_count", "mean_remaining_contrib_count",
        "radius_x", "radius_y", "radius_max", "mean_radius")}
    cam_stats = []

    hot_set = set(int(r) for r in cell["hot_ids"])
    part = np.asarray([0 if int(r) in hot_set else 1 for r in sel_ids],
                      dtype=np.int8)

    for slot, v in enumerate(cams):
        GI._OC0_LAST = None
        cam, gt = dataset.get_item(v)
        with torch.no_grad():
            frame, radii_info, means2d_info = C.render_view(model, cam, model.active_sh_degree)
        last = getattr(GI, "_OC0_LAST", None)
        if last is None:
            raise RuntimeError("instrumented forward did not publish _OC0_LAST "
                               f"(camera slot {slot})")
        captured = last["captured"]
        visible_ids = last["visible_ids"].detach().to("cuda", torch.int64)
        ctx = last["ctx"]
        ratio = float(getattr(ctx, "sampled_tile_ratio", 1.0))
        if ratio != 1.0:
            raise RuntimeError(f"tile sampling active (ratio={ratio}); the "
                               f"native replay would not be exact")

        res = R.replay_camera(captured, visible_ids, sel_t,
                              tile_size=16, force_all_tiles=False,
                              want_rgb=False)
        agg = R.finalize_aggregates(res, ng, acc_total=res["acc_total"])

        slot_sel = res["slot_sel_index"].numpy()          # [nvis] -> -1 or pos
        sel_slot = np.full(ng, -1, dtype=np.int64)
        nz = np.nonzero(slot_sel >= 0)[0]
        sel_slot[slot_sel[nz]] = nz
        in_frustum = sel_slot >= 0

        hits = agg["accepted_hits"].numpy()
        rx = np.full(ng, np.nan, dtype=np.float32)
        ry = np.full(ng, np.nan, dtype=np.float32)
        rxv = res["radius_x"].numpy()
        ryv = res["radius_y"].numpy()
        rx[in_frustum] = rxv[sel_slot[in_frustum]]
        ry[in_frustum] = ryv[sel_slot[in_frustum]]
        rmax = np.maximum(rx, ry)
        rmean = 0.5 * (rx + ry)

        cols["selection_pos"].append(np.arange(ng, dtype=np.int32))
        cols["gaussian_row"].append(np.asarray(sel_ids, dtype=np.int64))
        cols["partition"].append(np.where(part == 0, "hot", "uniform"))
        cols["camera_slot"].append(np.full(ng, slot, dtype=np.int32))
        cols["camera_id"].append(np.full(ng, v, dtype=np.int32))
        cols["visible"].append((hits > 0))
        cols["accepted_hits"].append(hits.astype(np.int32))
        cols["mean_alpha"].append(agg["mean_alpha"].numpy().astype(np.float32))
        cols["mean_T_before"].append(agg["mean_T_before"].numpy().astype(np.float32))
        cols["p10_T_before"].append(agg["p10_T_before"].numpy().astype(np.float32))
        cols["p50_T_before"].append(agg["p50_T_before"].numpy().astype(np.float32))
        cols["mean_depth_percentile"].append(
            agg["mean_depth_percentile"].numpy().astype(np.float32))
        cols["mean_prior_contrib_count"].append(
            agg["mean_prior_contrib_count"].numpy().astype(np.float32))
        cols["mean_remaining_contrib_count"].append(
            agg["mean_remaining_contrib_count"].numpy().astype(np.float32))
        cols["radius_x"].append(rx)
        cols["radius_y"].append(ry)
        cols["radius_max"].append(rmax)
        cols["mean_radius"].append(rmean)

        cam_stats.append({
            "camera_slot": slot, "camera_id": int(v),
            "n_visible_rows": int((hits > 0).sum()),
            "n_in_frustum_rows": int(in_frustum.sum()),
            "n_isects": int(res["n_isects"]),
            "n_isects_sampled_ctx": int(getattr(ctx, "n_isects_sampled", -1)),
            "n_processed_tiles": int(res["n_processed_tiles"]),
            "n_candidate_pairs": int(res["n_candidate_pairs"]),
            "n_accepted_pair_entries": int(agg["n_pair_entries"]),
            "max_abs_alpha_err_processed_tiles": res["max_abs_alpha_err"],
            "mean_abs_alpha_err_processed_tiles": res["mean_abs_alpha_err"],
            "max_abs_lastid_err": res["max_abs_lastid_err"],
            "n_lastid_compared": res["n_lastid_compared"],
            "n_lastid_mismatch": res["n_lastid_mismatch"],
        })
        print(f"[oc0_features] cam {slot + 1}/{len(cams)} id={v} "
              f"vis={cam_stats[-1]['n_visible_rows']}/{ng} "
              f"aerr_max={res['max_abs_alpha_err']:.3e} "
              f"lastid_err={res['max_abs_lastid_err']}", flush=True)
        del last, captured, visible_ids, res, agg
        torch.cuda.empty_cache()

    # ---- parquet (pyarrow only; pandas is not installed on the node) --------
    import pyarrow as pa
    import pyarrow.parquet as pq

    n_tot = ng * len(cams)

    def cat(key, dtype):
        # Row order is selection_pos major / camera_slot minor -- the contract
        # declared in the parquet metadata below and relied on by every
        # consumer (oc0_analysis.load_cell).  The columns are appended one
        # camera at a time, so they must be stacked to [ng, nc] before being
        # flattened; np.concatenate alone yields camera-major order.
        return np.stack(cols[key], axis=1).astype(dtype).reshape(-1)

    arrays = {
        "cell": pa.array([K.cell_tag(scene, stage)] * n_tot),
        "scene": pa.array([scene] * n_tot),
        "checkpoint": pa.array(np.full(n_tot, stage, dtype=np.int32)),
        "selection_pos": pa.array(cat("selection_pos", np.int32)),
        "gaussian_row": pa.array(cat("gaussian_row", np.int64)),
        "partition": pa.array(cat("partition", object)),
        "camera_slot": pa.array(cat("camera_slot", np.int32)),
        "camera_id": pa.array(cat("camera_id", np.int32)),
        "visible": pa.array(cat("visible", bool)),
        "accepted_hits": pa.array(cat("accepted_hits", np.int32)),
    }
    for k in ("mean_alpha", "mean_T_before", "p10_T_before", "p50_T_before",
              "mean_depth_percentile", "mean_prior_contrib_count",
              "mean_remaining_contrib_count", "radius_x", "radius_y",
              "radius_max", "mean_radius"):
        arrays[k] = pa.array(cat(k, np.float32), type=pa.float32(),
                             from_pandas=True)

    meta = {
        b"campaign": b"OPTLAWS0",
        b"stage": b"OC0_renderer_features",
        b"prereg": b"OC0_PREREG.md",
        b"cell": K.cell_tag(scene, stage).encode(),
        b"n_rows": str(n_tot).encode(),
        b"n_selected_gaussians": str(ng).encode(),
        b"n_cameras": str(len(cams)).encode(),
        b"row_order": b"selection_pos major, camera_slot minor",
        b"partition_rule": (b"selection positions [0,1024) are the MG0-S "
                            b"top-V_j hot rows; [1024,2048) are the frozen "
                            b"uniform nonzero-V_j rows"),
        b"null_rule": (b"per-camera-summary columns are null when the row has "
                       b"no accepted hit; radius columns are null when the row "
                       b"is not in the visible subset of that camera"),
        b"definitions": (
            b"accepted_hits: native accepted pixel interactions; "
            b"mean_alpha: mean min(0.999,opac*exp(-sigma)) over accepted hits; "
            b"mean_T_before/p10/p50: transmittance immediately before the "
            b"Gaussian over accepted hits; "
            b"mean_depth_percentile: mean of lpos/(L-1) over accepted hits, "
            b"L = tile isect count; "
            b"mean_prior_contrib_count: mean accepted contributors strictly "
            b"before this Gaussian in its pixel; "
            b"mean_remaining_contrib_count: mean accepted contributors "
            b"strictly after it; "
            b"radius_x/radius_y: native projected radius; "
            b"radius_max/mean_radius: derived from them"),
        b"replay": (b"vectorized exact re-derivation of the production "
                    b"rasterize_to_pixels_3dgs loop from the tensors the native "
                    b"forward published; alpha compared against native "
                    b"render_alphas, and the last accepted isect index against "
                    b"native last_ids"),
        b"camera_stats": json.dumps(cam_stats).encode(),
        b"bootstrap": json.dumps(binfo).encode(),
        b"gpu_uuid": os.environ.get("MG0_GPU_UUID", "unknown").encode(),
        b"creation_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()).encode(),
    }
    table = pa.Table.from_arrays(list(arrays.values()),
                                 names=list(arrays.keys()))
    table = table.replace_schema_metadata(meta)
    os.makedirs(K.PARQUET_DIR, exist_ok=True)
    path = K.parquet_path(scene, stage)
    pq.write_table(table, path, compression="zstd")
    print(f"[oc0_features] {scene}@{stage} wrote {path} rows={n_tot} "
          f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()