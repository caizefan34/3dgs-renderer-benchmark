#!/usr/bin/env python3
"""D0 Stage 5: final-state descriptive audit aggregation.

Reads per scene:
  runs/TRACE_<scene>/d0_final_audit.npz       (in-process 16-camera audit)
  runs/TRACE_<scene>/d0_uid_table.npz         (uid lineage bookkeeping)
  artifacts/D0_LINEAGE_OUTCOMES_<scene>.parquet   (stage-3 root outcomes)

Aggregates the per-row contribution proxies by surviving lineage (post-init
birth root, same partition as stage 3) and writes:
  artifacts/D0_FINAL_CONTRIBUTION_AUDIT.json
  artifacts/D0_FINAL_CONTRIBUTION_AUDIT_<scene>.parquet  (full table)

Interpretation guard (prereg): a lineage with low recorded contribution is
NOT "safe to remove"; this stage makes no causal or removal claim.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage3_accounting import lineage_partition  # noqa: E402

ROOT = "/root/epic05_d0"
RUNS = os.path.join(ROOT, "runs")
ART = os.path.join(ROOT, "artifacts")

GUARD = ("descriptive only: low recorded contribution does NOT mean 'safe to "
         "remove'; no causal or selection claim is made by this stage")


def analyze_scene(scene, runs_dir, art_dir):
    d = os.path.join(runs_dir, "TRACE_%s" % scene)
    au = np.load(os.path.join(d, "d0_final_audit.npz"), allow_pickle=True)
    ut = np.load(os.path.join(d, "d0_uid_table.npz"))

    row_uid = au["row_uid"].astype(np.int64)
    vis = au["visibility_count"].astype(np.int64)
    vis_mask = au["visibility_mask"].astype(np.uint16)
    tiles = au["tiles_total"].astype(np.int64)
    fp = au["alpha_footprint"].astype(np.float64)
    N = int(au["n_rows"])
    audit_cams = au["audit_cameras"].astype(np.int64)
    keys = [str(k) for k in au["per_camera_keys"]]
    pc = au["per_camera"]
    iters = int(au["iterations"])

    bt = ut["birth_type"].astype(np.int64)
    parent = ut["parent_uid"].astype(np.int64)
    birth_iter = ut["birth_iter"].astype(np.int64)
    birth_event = ut["birth_event"].astype(np.int64)
    death_iter = ut["death_iter"].astype(np.int64)
    U = int(bt.shape[0])
    root_of, _order = lineage_partition(bt, parent, birth_event, "post")

    alive_uid = np.nonzero(death_iter == -1)[0]
    gates = {
        "audit_written_at_iterations": iters,
        "n_rows": N,
        "n_alive_uids": int(alive_uid.size),
        "rows_match_final_N": bool(
            N == row_uid.size == vis.size == tiles.size == fp.size),
        "row_uid_unique": bool(np.unique(row_uid).size == row_uid.size),
        "row_uid_set_equals_alive_uids": bool(
            np.array_equal(np.sort(row_uid), alive_uid)),
        "uid_range_ok": bool(row_uid.size and row_uid.min() >= 0
                             and row_uid.max() < U),
        "audit_cameras_distinct_ascending": bool(
            audit_cams.size == 16 and np.all(np.diff(audit_cams) > 0)),
    }
    pop = np.zeros(vis_mask.size, dtype=np.int64)
    for j in range(audit_cams.size):
        pop += (vis_mask >> np.uint16(j)) & np.uint16(1)
    gates["visibility_mask_popcount_equals_count"] = bool(
        np.array_equal(pop, vis))

    k = {nm: i for i, nm in enumerate(keys)}
    per_cam = [{nm: float(pc[j, k[nm]]) for nm in keys} for j in range(pc.shape[0])]
    gates["per_camera_tiles_vs_isects_max_abs_delta"] = int(max(
        abs(int(c["sum_tiles"]) - int(c["n_isects"])) for c in per_cam))

    rid = root_of[row_uid]
    is_post = rid >= 0

    import pandas as pd
    lin = pd.read_parquet(
        os.path.join(art_dir, "D0_LINEAGE_OUTCOMES_%s.parquet" % scene))
    surv = set(int(r) for r in
               lin.loc[lin["has_final_descendant"], "root_uid"].to_numpy())
    roots, inv = np.unique(rid[is_post], return_inverse=True)
    gates["audit_covers_all_surviving_lineages"] = bool(
        set(int(r) for r in roots) == surv)

    n_rows = np.bincount(inv, minlength=roots.size).astype(np.int64)
    vis_t = np.bincount(inv, minlength=roots.size,
                        weights=vis[is_post].astype(np.float64)).astype(np.int64)
    tiles_t = np.bincount(inv, minlength=roots.size,
                          weights=tiles[is_post].astype(np.float64)).astype(np.int64)
    fp_t = np.bincount(inv, minlength=roots.size,
                       weights=fp[is_post]).astype(np.float64)

    lin_idx = lin.set_index("root_uid")
    birth = lin_idx["birth_iter"].reindex(roots).to_numpy(dtype=np.int64)
    n_members = lin_idx["n_members"].reindex(roots).to_numpy(dtype=np.int64)
    n_desc = lin_idx["n_final_descendants"].reindex(roots).to_numpy(dtype=np.int64)

    cam_union = np.zeros((audit_cams.size, roots.size), dtype=bool)
    for j in range(audit_cams.size):
        m = ((vis_mask[is_post] >> np.uint16(j)) & np.uint16(1)).astype(np.float64)
        cam_union[j] = np.bincount(inv, minlength=roots.size,
                                   weights=m) > 0

    tab = pd.DataFrame({
        "scene": scene,
        "root_uid": roots.astype(np.int64),
        "birth_iter": birth,
        "n_final_rows": n_rows,
        "visibility_count_total": vis_t,
        "visibility_count_mean_per_row": vis_t / np.maximum(n_rows, 1),
        "cameras_with_visible_row": cam_union.sum(axis=0).astype(np.int64),
        "tile_intersections_total": tiles_t,
        "tiles_mean_per_row": tiles_t / np.maximum(n_rows, 1),
        "alpha_footprint_total": fp_t,
        "alpha_footprint_mean_per_row": fp_t / np.maximum(n_rows, 1),
        "n_members": n_members,
        "n_final_descendants": n_desc,
    })

    def q(a):
        if a.size == 0:
            return None
        qs = np.percentile(a, [0, 25, 50, 75, 90, 99, 100])
        return {nm: float(v) for nm, v in zip(
            ("p0", "p25", "p50", "p75", "p90", "p99", "p100"), qs)}

    total_fp = float(fp_t.sum())
    order_fp = np.sort(fp_t)[::-1]
    n_l = int(roots.size)
    share = {}
    for frac in (0.01, 0.05, 0.10):
        m = max(int(round(frac * n_l)), 1)
        share["top%.0f%%_fp_share" % (frac * 100)] = (
            float(order_fp[:m].sum()) / total_fp if total_fp > 0 else None)
    tiny = np.sort(fp_t)
    small_frac = {thr: float((fp_t <= thr * max(total_fp, 1e-30)).mean())
                  for thr in (1e-6, 1e-4)}

    top = tab.sort_values("alpha_footprint_total", ascending=False).head(20)
    bot = tab.sort_values("alpha_footprint_total", ascending=True).head(20)

    init_rows = int((~is_post).sum())
    out = {
        "scene": scene,
        "run_dir": d,
        "audit_cameras": [int(c) for c in audit_cams],
        "rgb_contribution_proxy": {
            "status": "unavailable",
            "reason": "not cheaply exposed by the C0 renderer; recorded as "
                      "unavailable rather than approximated (spec 6)",
        },
        "gates": gates,
        "surviving_lineages": {
            "n": n_l,
            "n_with_zero_recorded_footprint": int((fp_t == 0).sum()),
            "alpha_footprint_total_sum": total_fp,
            "tile_intersections_total_sum": int(tiles_t.sum()),
            "visibility_count_total_sum": int(vis_t.sum()),
            "fp_total_quantiles": q(fp_t),
            "tiles_total_quantiles": q(tiles_t.astype(np.float64)),
            "fp_concentration": share,
            "tiny_footprint_fraction": small_frac,
            "lineages_visible_per_camera": [
                int(v) for v in cam_union.sum(axis=1)],
        },
        "init_population_bucket": {
            "n_rows": init_rows,
            "visibility_count_total": int(vis[~is_post].sum()),
            "tile_intersections_total": int(tiles[~is_post].sum()),
            "alpha_footprint_total": float(fp[~is_post].sum()),
            "note": "rows whose uid has no post-init birth root (init "
                    "ancestors); not part of the stage-3 birth-root partition",
        },
        "top20_by_alpha_footprint": top.to_dict(orient="records"),
        "bottom20_by_alpha_footprint": bot.to_dict(orient="records"),
        "per_camera": per_cam,
    }
    tab.to_parquet(os.path.join(
        art_dir, "D0_FINAL_CONTRIBUTION_AUDIT_%s.parquet" % scene),
        index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*")
    ap.add_argument("--runs-dir", default=RUNS)
    ap.add_argument("--art-dir", default=ART)
    args = ap.parse_args()
    scenes = args.scenes or ["room", "bicycle"]
    os.makedirs(args.art_dir, exist_ok=True)

    out = {"artifact": "D0_FINAL_CONTRIBUTION_AUDIT.json",
           "stage": "5 (final-state descriptive audit)",
           "interpretation_guard": GUARD,
           "protocol": {
               "audit_cameras": "round(k*(n_cam-1)/15), k=0..15 (spec 6)",
               "proxies": ["visibility_count (# audit cameras visible)",
                           "tile_intersections_total",
                           "alpha_footprint_proxy = sum_cam sigma(opacity)*"
                           "min(pi*r_max_px^2, W*H)"],
               "aggregation": "sum over final-state rows of each surviving "
                              "post-init birth-root lineage (stage-3 "
                              "partition); init lineage ancestors reported "
                              "as a separate bucket",
           },
           "runs": {}}
    for s in scenes:
        out["runs"][s] = analyze_scene(s, args.runs_dir, args.art_dir)
        g = out["runs"][s]["gates"]
        print("== %s audit: rows=%d survivors=%d gates=%s" %
              (s, g["n_rows"], out["runs"][s]["surviving_lineages"]["n"],
               {kk: g[kk] for kk in ("rows_match_final_N",
                                     "row_uid_set_equals_alive_uids",
                                     "audit_covers_all_surviving_lineages")}),
              flush=True)
    with open(os.path.join(args.art_dir, "D0_FINAL_CONTRIBUTION_AUDIT.json"),
              "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("wrote D0_FINAL_CONTRIBUTION_AUDIT.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
