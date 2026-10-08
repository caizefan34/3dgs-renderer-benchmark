#!/usr/bin/env python3
"""D0 Stage 3: per-lineage work accounting + tail quality + headroom bound.

Run on EPIC-05 with the system python3 (numpy/pandas/pyarrow).  Reads the
TRACE_<scene> 30K runs and writes into /root/epic05_d0/artifacts:

  D0_LINEAGE_SCHEMA.json          column definitions + interpretation notes
  D0_BIRTH_TABLE.parquet          per post-init birth: ids + 30 pre-birth feats
  D0_LINEAGE_OUTCOMES.parquet     per post-init birth root: spec-3 outcomes
  D0_POPULATION_TRAJECTORY.parquet  every-iteration trajectory (both scenes)
  D0_RENDER_COST_SAMPLES.parquet    sampled (iter, uid, tiles) rows
  D0_COST_ACCOUNTING.json         r_I_dead / r_N_dead + reconstruction gates
  D0_TAIL_QUALITY.json            checkpoint evals + t_star
  D0_HEADROOM.json                R_upper / S_upper + CI + per-scene gates
  D0_SENSITIVITY_INIT_ROOTS.json  spec-4.5 variant (reported, never substituted)

Exit 0 iff every analyzed scene passes the accounting validity gates
(sample-shard crosscheck + exact per-iteration alignment + weighted
reconstruction error <= 3%; isect sumcheck relative delta <= 1e-6 with the
raw mismatch count reported verbatim).  On failure the headroom verdict is
withheld (accounting invalid -> STOP before verdict).
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = "/root/epic05_d0"
RUNS = os.path.join(ROOT, "runs")
ART = os.path.join(ROOT, "artifacts")

F_FB = 13.915 / 19.369
F_OPT = 4.140 / 19.369
T_FULL = 30000
CKPTS = [5000, 10000, 15000, 20000, 25000]
ENV = {"psnr": 0.10, "ssim": 0.003, "lpips": 0.005}
BOOT_N = 2000
BOOT_SEED = 20260928
RECON_MAX = 0.03

FEATURE_NAMES = [
    "parent_age", "parent_opacity", "parent_xyz_norm", "depth_last_visible",
    "scale_x", "scale_y", "scale_z", "scale_volume", "radius_last_px",
    "sh_degree_active",
    "grad_accum_row", "grad_denom_row", "grad_mean_norm", "threshold_margin",
    "max_radii2D_row", "last_absgrad_norm_px", "branch",
    "vis_count_32", "vis_count_128", "vis_frac_32", "vis_frac_128",
    "n_distinct_cameras", "view_dir_spread_32",
    "res_mean_current_tile", "res_p90_current_tile", "res_roll_mean_32",
    "res_roll_var_32", "grad_dir_stability_32", "footprint_tiles_last_obs",
    "footprint_tiles_last32_mean",
]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def lineage_partition(birth_type, parent, birth_event, mode):
    """Root uid per post-init birth ('post') or per init ancestor
    ('init_sensitivity').  Parents always have a smaller uid and belong to an
    earlier event (or init), so processing events in ascending order resolves
    every parent's root before its children.
    """
    U = int(birth_type.shape[0])
    root = np.full(U, -1, dtype=np.int64)
    order = np.nonzero(birth_type > 0)[0]
    if order.size == 0:
        return root, order
    ev = birth_event[order]
    if mode != "post":
        root[np.nonzero(birth_type == 0)[0]] = np.nonzero(birth_type == 0)[0]
    srt = np.argsort(ev, kind="stable")
    o_sorted, ev_sorted = order[srt], ev[srt]
    uniq, starts = np.unique(ev_sorted, return_index=True)
    for i, e in enumerate(uniq):
        lo = starts[i]
        hi = starts[i + 1] if i + 1 < len(starts) else o_sorted.size
        idx = o_sorted[lo:hi]
        p = parent[idx]
        p_init = birth_type[p] == 0
        if p_init.any():
            root[idx[p_init]] = (idx[p_init] if mode == "post"
                                 else p[p_init])
        rest = ~p_init
        if rest.any():
            root[idx[rest]] = root[p[rest]]
    return root, order


def r_upper(r_i, r_n, ts):
    num = (F_FB * r_i + F_OPT * r_n) * ts + (F_FB + F_OPT) * (T_FULL - ts)
    return num / ((F_FB + F_OPT) * T_FULL)


def analyze_scene(scene, scene_id, writer=None, pa=None):
    d = os.path.join(RUNS, "TRACE_%s" % scene)
    out = {"scene": scene, "run_dir": d}
    ut = np.load(os.path.join(d, "d0_uid_table.npz"))
    traj = np.load(os.path.join(d, "d0_population_trajectory.npz"))
    meta = load_json(os.path.join(d, "d0_trace_meta.json"))
    results = load_json(os.path.join(d, "results.json"))

    bt = ut["birth_type"].astype(np.int64)
    parent = ut["parent_uid"].astype(np.int64)
    birth_iter = ut["birth_iter"].astype(np.int64)
    birth_event = ut["birth_event"].astype(np.int64)
    death_iter = ut["death_iter"].astype(np.int64)
    U = int(bt.shape[0])
    n_init = int((bt == 0).sum())

    # ---------------- lineage partitions ------------------------------------
    root_of, order = lineage_partition(bt, parent, birth_event, "post")
    root_s, _ = lineage_partition(bt, parent, birth_event, "init_sensitivity")
    members = np.bincount(root_of[order], minlength=U)
    root_has = members > 0
    alive_m = death_iter[order] == -1
    n_alive = np.bincount(root_of[order][alive_m], minlength=U)
    root_alive = n_alive > 0
    root_dead = (~root_alive) & root_has
    last = np.where(death_iter >= 0, death_iter, T_FULL)
    integ = np.maximum(0, np.minimum(last, T_FULL) - birth_iter + 1).astype(np.float64)
    root_integ = np.bincount(root_of[order], weights=integ[order], minlength=U)
    root_death = np.full(U, -1.0)
    np.maximum.at(root_death, root_of[order], death_iter[order].astype(np.float64))
    root_death[root_alive] = -1.0
    root_birth = np.where(bt > 0, birth_iter, -1)
    n_roots = int(root_has.sum())
    out["lineage"] = {
        "n_init": n_init, "n_uids": U, "n_post_init_births": int(order.size),
        "n_roots": n_roots, "n_dead_roots": int(root_dead.sum()),
        "dead_root_frac": float(root_dead.sum()) / max(n_roots, 1),
        "n_final_descendants_total": int(n_alive.sum()),
        "lineage_population_integral_total": float(root_integ[root_has].sum()),
        "mean_members_per_root": float(members[root_has].mean()),
    }

    # sensitivity partition aggregation
    m_s = np.bincount(root_s, minlength=U)
    alive_idx = np.nonzero(death_iter == -1)[0]
    n_alive_s = np.bincount(root_s[alive_idx], minlength=U)
    root_dead_s = (n_alive_s == 0) & (m_s > 0) & (bt == 0)

    # ---------------- single shard pass -------------------------------------
    sdir = os.path.join(d, "d0_cost_samples")
    shards = sorted(f for f in os.listdir(sdir) if f.startswith("sample_"))
    s_iters = np.array([int(f.split("_")[-1].split(".")[0]) for f in shards],
                       dtype=np.int64)
    n_s = len(shards)
    s_all = np.zeros(n_s)
    s_dead = np.zeros(n_s)
    s_dead_s = np.zeros(n_s)
    per_root_tiles = np.zeros(U, dtype=np.float64)
    pos_uid = root_of >= 0
    root_clamped = np.where(pos_uid, root_of, 0)
    for i, fn in enumerate(shards):
        z = np.load(os.path.join(sdir, fn))
        uid = z["uid"].astype(np.int64)
        tiles = z["tiles"]
        tf = tiles.astype(np.float64)
        s_all[i] = tf.sum()
        s_dead[i] = tf[root_dead[root_clamped[uid]] & pos_uid[uid]].sum()
        s_dead_s[i] = tf[root_dead_s[root_s[uid]]].sum()
        per_root_tiles += np.bincount(root_clamped[uid],
                                      weights=tf * pos_uid[uid], minlength=U)
        if writer is not None:
            writer.write_table(pa.table({
                "scene_id": pa.array(np.full(uid.size, scene_id, dtype=np.int8)),
                "iter": pa.array(np.full(uid.size, s_iters[i], dtype=np.int32)),
                "uid": pa.array(uid), "tiles": pa.array(tiles)}))
        if i % 250 == 0:
            print("   %s shard %d/%d" % (scene, i, n_s), flush=True)
    summary_path = os.path.join(d, "d0_sample_summary.npz")
    summary_ok = None
    if os.path.exists(summary_path):
        summ = np.load(summary_path)
        if int(summ["iter"].size) == n_s:
            summary_ok = bool(np.array_equal(summ["iter"].astype(np.int64), s_iters)
                              and np.allclose(summ["sum_tiles"], s_all))
    exact_I = np.array(traj["I_t"], dtype=np.float64)
    traj_iter = np.array(traj["iter"], dtype=np.int64)
    pos = np.searchsorted(traj_iter, s_iters)
    aligned = bool(s_iters.size and bool((pos < traj_iter.size).all())
                   and np.array_equal(traj_iter[np.minimum(pos, traj_iter.size - 1)],
                                      s_iters))
    exact_at_samples = exact_I[np.minimum(pos, traj_iter.size - 1)]
    per_sample_rel = np.abs(s_all - exact_at_samples) / np.maximum(exact_at_samples, 1.0)
    # samples land every 20th iteration; the sampled estimator of the exact
    # total I_t is 20 * sum(sampled).  The 3% gate applies to that estimator
    # (prereg stage-3 accounting validity).
    recon_err = abs(1.0 - float(s_all.sum()) * 20.0 / float(exact_I.sum()))
    mism = meta.get("sumcheck_mismatches") or []
    n_mism = int(meta.get("n_sumcheck_mismatches", -1))
    sumcheck_rel = 0.0
    if mism:
        sumcheck_rel = max(abs(int(m["sum"]) - int(m["n_isects"]))
                           / max(int(m["n_isects"]), 1) for m in mism)
    # +-1-unit deltas between the pass-1 AccuTile count and the pass-2 emit
    # count are frozen-op-internal (see D0_LINEAGE_SCHEMA notes); gate on
    # relative magnitude, report the raw count verbatim.
    sumcheck_ok = bool(n_mism == 0 or (n_mism > 0 and sumcheck_rel <= 1e-6))
    out["accounting"] = {
        "n_samples": int(n_s), "expected_samples": 1500,
        "exact_I_t_total": float(exact_I.sum()),
        "sampled_tiles_total": float(s_all.sum()),
        "weighted_estimate_total": float(s_all.sum()) * 20.0,
        "reconstruction_error": float(recon_err),
        "reconstruction_gate_ok": bool(recon_err <= RECON_MAX),
        "sample_iter_alignment_ok": aligned,
        "per_sample_rel_max": float(per_sample_rel.max()) if per_sample_rel.size else None,
        "per_sample_rel_median": (float(np.median(per_sample_rel))
                                  if per_sample_rel.size else None),
        "sample_summary_crosscheck_ok": summary_ok,
        "sumcheck_mismatches_n": n_mism,
        "sumcheck_max_rel": float(sumcheck_rel),
        "sumcheck_examples": mism[:3],
        "sumcheck_gate_ok": sumcheck_ok,
        "grad_missing_iters": int(meta.get("grad_missing_iters", -1)),
    }

    # ---------------- tail quality / t_star ---------------------------------
    ev = results.get("eval_rows", [])
    ck = {r["step"]: r for r in ev if r.get("tag") == "d0_ckpt"}
    final = ck.get(30000)
    tail = {"checkpoints": {str(k): {kk: ck[k][kk] for kk in
                                     ("psnr", "ssim", "lpips", "n_eval_cameras",
                                      "n_gaussians")}
                            for k in sorted(ck)},
            "final_source": "d0_ckpt@30000" if final else None}
    t_star, t_star_reason = T_FULL, "no earlier checkpoint inside the frozen envelope"
    if final is not None:
        box = {"psnr_min": final["psnr"] - ENV["psnr"],
               "ssim_min": final["ssim"] - ENV["ssim"],
               "lpips_max": final["lpips"] + ENV["lpips"]}
        tail["envelope"] = box
        for k in CKPTS:
            r = ck.get(k)
            if r and r["psnr"] >= box["psnr_min"] and r["ssim"] >= box["ssim_min"] \
                    and r.get("lpips", 9) <= box["lpips_max"]:
                t_star, t_star_reason = k, "earliest checkpoint inside envelope"
                break
    tail["t_star"] = int(t_star)
    tail["t_star_reason"] = t_star_reason
    qp = os.path.join(d, "quality.json")
    if os.path.exists(qp):
        q = load_json(qp)
        tail["final_quality_json"] = {k: q.get(k) for k in
                                      ("psnr", "ssim", "lpips", "n_eval_cameras")}
    out["tail"] = tail

    # ---------------- pre-tail dead fractions + bootstrap -------------------
    pre = s_iters <= t_star
    n_pre = int(pre.sum())
    pool = np.nonzero(pre)[0]
    r_i_dead = float(s_dead[pre].sum() / max(s_all[pre].sum(), 1e-12))
    pop_T = np.maximum(0, np.minimum(last, t_star) - birth_iter + 1).astype(np.float64)
    root_pop_T = np.bincount(root_of[order], weights=pop_T[order], minlength=U)
    r_n_dead = float(root_pop_T[root_dead].sum() /
                     max(root_pop_T[root_has].sum(), 1e-12))
    R = r_upper(r_i_dead, r_n_dead, t_star)
    rng = np.random.default_rng(BOOT_SEED)
    R_boot = np.empty(BOOT_N)
    for b in range(BOOT_N):
        pi = pool[rng.integers(0, n_pre, size=n_pre)]
        R_boot[b] = r_upper(s_dead[pi].sum() / max(s_all[pi].sum(), 1e-12),
                            r_n_dead, t_star)
    ci = np.percentile(R_boot, [2.5, 97.5])
    out["headroom"] = {
        "t_star": int(t_star), "n_pretail_samples": n_pre,
        "r_I_dead": r_i_dead, "r_N_dead": r_n_dead,
        "R_dead_pretail": F_FB * r_i_dead + F_OPT * r_n_dead,
        "R_upper": R,
        "S_upper": 1.0 / (1.0 - R) if R < 1 else float("inf"),
        "R_upper_ci95": [float(ci[0]), float(ci[1])],
        "S_upper_at_ci95_low": (1.0 / (1.0 - ci[0]) if ci[0] < 1 else float("inf")),
        "bootstrap": {"n": BOOT_N, "seed": BOOT_SEED,
                      "resample": "pre-tail sampled iterations with replacement",
                      "r_N_dead": "held exact"},
        "gate_headroom_ge_070": bool(ci[0] >= 0.70),
        "gate_no_headroom_lt_060": bool(ci[0] < 0.60),
    }

    # ---------------- spec-4.5 init-root sensitivity ------------------------
    pop_root_s = np.bincount(root_s, weights=integ, minlength=U)
    pop_root_s_T = np.bincount(root_s, weights=pop_T, minlength=U)
    r_i_s = float(s_dead_s[pre].sum() / max(s_all[pre].sum(), 1e-12))
    r_n_s = float(pop_root_s_T[root_dead_s].sum() /
                  max(pop_root_s_T[bt == 0].sum(), 1e-12))
    R_s = r_upper(r_i_s, r_n_s, t_star)
    out["sensitivity_init_roots"] = {
        "n_init_roots": int((bt == 0).sum()),
        "n_dead_init_roots": int(root_dead_s.sum()),
        "r_I_dead": r_i_s, "r_N_dead": r_n_s, "R_upper": R_s,
        "S_upper": 1.0 / (1.0 - R_s) if R_s < 1 else float("inf"),
        "note": "reported only; verdict uses the post-init partition (spec 4.5)",
    }

    # ---------------- per-scene tables --------------------------------------
    try:
        import pandas as pd
        birth = {
            "scene": scene,
            "uid": order, "parent_uid": parent[order], "birth_iter": birth_iter[order],
            "birth_type": bt[order], "birth_event_id": birth_event[order],
            "root_uid": root_of[order],
        }
        br = np.load(os.path.join(d, "d0_birth_records.npz"), allow_pickle=True)
        feat = br["feat"]
        assert feat.shape[1] == order.size, (feat.shape, order.size)
        for i, nm in enumerate(FEATURE_NAMES):
            birth[nm] = feat[i].astype(np.float32)
        pd.DataFrame(birth).to_parquet(
            os.path.join(ART, "D0_BIRTH_TABLE_%s.parquet" % scene), index=False)
        lin_death = root_death[root_has]
        rb = root_birth[root_has]
        lin = {
            "scene": scene,
            "root_uid": np.nonzero(root_has)[0],
            "birth_iter": rb, "birth_type": bt[root_has],
            "n_members": members[root_has],
            "n_final_descendants": n_alive[root_has],
            "has_final_descendant": (n_alive > 0)[root_has],
            "lineage_death_iter": lin_death,
            "lineage_lifetime": np.where(lin_death >= 0, lin_death - rb, -1),
            "lineage_population_integral": root_integ[root_has],
            "pop_integral_pretail": root_pop_T[root_has],
            "retrospectively_dead": root_dead[root_has],
            "tiles_sampled_full": per_root_tiles[root_has],
        }
        pd.DataFrame(lin).to_parquet(
            os.path.join(ART, "D0_LINEAGE_OUTCOMES_%s.parquet" % scene), index=False)
        out["_birth"] = birth
        out["_lineage"] = lin
        out["artifacts"] = {"birth_table_rows": int(order.size),
                            "lineage_rows": n_roots}
    except ImportError:
        out["artifacts"] = {"parquet": "pandas/pyarrow unavailable"}

    out["_trajectory"] = {k: traj[k] for k in traj.files}
    return out


def write_merged(scenes_out, keys=("iter", "N_t", "n_visible", "I_t",
                                   "I_t_sumchecked", "loss")):
    import pandas as pd
    frames = []
    for s, o in scenes_out.items():
        t = o.pop("_trajectory")
        df = pd.DataFrame({k: t[k] for k in t})
        df.insert(0, "scene", s)
        frames.append(df)
    pd.concat(frames, ignore_index=True).to_parquet(
        os.path.join(ART, "D0_POPULATION_TRAJECTORY.parquet"), index=False)


def write_merged_tables(scenes_out):
    # Canonical unsuffixed prereg artifacts (concatenation of the per-scene
    # tables written inside analyze_scene).
    import pandas as pd
    frames = {k: [] for k in ("_birth", "_lineage")}
    for s, o in scenes_out.items():
        for key in frames:
            d = o.pop(key, None)
            if d is not None:
                frames[key].append(pd.DataFrame(d))
    names = {"_birth": "D0_BIRTH_TABLE.parquet",
             "_lineage": "D0_LINEAGE_OUTCOMES.parquet"}
    for key, name in names.items():
        if frames[key]:
            pd.concat(frames[key], ignore_index=True).to_parquet(
                os.path.join(ART, name), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*")
    ap.add_argument("--no-parquet", action="store_true")
    args = ap.parse_args()
    scenes = args.scenes or ["room", "bicycle"]

    os.makedirs(ART, exist_ok=True)
    scenes_out, invalid = {}, []
    writer = None
    pa = None
    if not args.no_parquet:
        import pyarrow as pa_
        import pyarrow.parquet as pq
        pa = pa_
        schema = pa.schema([("scene_id", pa.int8()), ("iter", pa.int32()),
                            ("uid", pa.int64()), ("tiles", pa.uint16())])
        writer = pq.ParquetWriter(os.path.join(ART, "D0_RENDER_COST_SAMPLES.parquet"),
                                  schema, compression="zstd")
    try:
        for si, s in enumerate(scenes):
            print("== stage3 %s" % s, flush=True)
            o = analyze_scene(s, si, writer=writer, pa=pa)
            scenes_out[s] = o
            a = o["accounting"]
            if not (a["reconstruction_gate_ok"] and a["sumcheck_gate_ok"]
                    and a["sample_iter_alignment_ok"]
                    and a["sample_summary_crosscheck_ok"]):
                invalid.append(s)
            print("   recon_err=%.6f gate=%s sumcheck_rel=%.2e(sum %s) "
                  "align=%s crosscheck=%s samples=%d" %
                  (a["reconstruction_error"], a["reconstruction_gate_ok"],
                   a["sumcheck_max_rel"], a["sumcheck_mismatches_n"],
                   a["sample_iter_alignment_ok"],
                   a["sample_summary_crosscheck_ok"], a["n_samples"]), flush=True)
            h = o["headroom"]
            print("   t_star=%d r_I=%.4f r_N=%.4f R_upper=%.4f ci_low=%.4f S=%.2f" %
                  (h["t_star"], h["r_I_dead"], h["r_N_dead"], h["R_upper"],
                   h["R_upper_ci95"][0], h["S_upper"]), flush=True)
    finally:
        if writer is not None:
            writer.close()
    write_merged(scenes_out)
    write_merged_tables(scenes_out)

    cost = {"artifact": "D0_COST_ACCOUNTING.json",
            "runs": {s: {"accounting": scenes_out[s]["accounting"],
                         "lineage": scenes_out[s]["lineage"]} for s in scenes}}
    head = {"artifact": "D0_HEADROOM.json",
            "F_FB": F_FB, "F_OPT": F_OPT, "T": T_FULL,
            "runs": {s: scenes_out[s]["headroom"] for s in scenes},
            "sensitivity_init_roots": {s: scenes_out[s]["sensitivity_init_roots"]
                                       for s in scenes},
            "outputs": {s: scenes_out[s]["artifacts"] for s in scenes}}
    tail = {"artifact": "D0_TAIL_QUALITY.json",
            "runs": {s: scenes_out[s]["tail"] for s in scenes}}
    if invalid:
        cost["accounting_invalid"] = invalid
        head["verdict"] = "ACCOUNTING_INVALID"
        head["note"] = ("reconstruction/self-check gate failed -> no headroom "
                        "verdict (prereg: STOP before headroom)")
    else:
        hl = [scenes_out[s]["headroom"]["gate_headroom_ge_070"] for s in scenes]
        nh = [scenes_out[s]["headroom"]["gate_no_headroom_lt_060"] for s in scenes]
        head["verdict"] = ("BOTH_SCENES_HEADROOM" if all(hl) else
                           "D0_NO_3X_HEADROOM" if any(nh) else
                           "D0_INCONCLUSIVE_BORDERLINE")
    schema = {
        "artifact": "D0_LINEAGE_SCHEMA.json",
        "uid_table": ["uid", "parent_uid(-1 init)", "birth_iter(0 init)",
                      "birth_type(0 init,1 clone,2 split_child)",
                      "birth_event_id(-1 init)", "death_iter(-1 alive)",
                      "terminal_reason(0 alive,1 opacity,2 screen,3 world,"
                      "4 split_parent_removed)",
                      "tiles_cum", "cam_count", "hist_n"],
        "birth_table": ["scene", "uid", "parent_uid", "birth_iter", "birth_type",
                        "birth_event_id", "root_uid"] + FEATURE_NAMES,
        "feature_order_note": "f00..f29 in the order above (spec 3.3)",
        "lineage_outcomes": ["scene", "root_uid", "birth_iter", "birth_type",
                             "n_members", "n_final_descendants",
                             "has_final_descendant", "lineage_death_iter",
                             "lineage_lifetime", "lineage_population_integral",
                             "pop_integral_pretail", "retrospectively_dead",
                             "tiles_sampled_full(unweighted; x20 = estimate)"],
        "cost_samples": ["scene_id(0 room,1 bicycle by CLI order)", "iter", "uid",
                         "tiles(nonzero only, uint16)"],
        "iteration_semantics": ("rows born in the densify event of iteration t "
                                "are updated at t; rows removed at d are not "
                                "updated at d (frozen spec 3.1)"),
        "implementation_notes": [
            "vis_stamp uses int32 + sentinel -(1<<20) instead of the spec "
            "table's uint16/0xFF00 encoding; identical semantics (a slot is "
            "valid iff its stamp lies within the last 128 iterations).",
            "f9: the frozen recipe renders the baseline-native branch for "
            "sh_degree<3 (iterations 1-2999) and the gatherless F9 branch from "
            "iteration 3000; both expose identical visible-row metadata at the "
            "shared capture site (recorded per iteration in the trajectory).",
            "terminal_reason=2 (screen) is structurally unreachable: the "
            "unconditional clone densification_postfix zeroes max_radii2D "
            "before the prune mask is built (frozen behavior; all non-opacity "
            "prunes in the P12 reference runs are world prunes, first at 3100).",
            "lineage_population_integral follows the frozen spec formula "
            "max(0, min(last_alive,30000)-birth+1) verbatim.",
            "isect sumcheck: sum(tiles_per_gauss) vs n_isects can differ by "
            "+-1 unit at rare geometries (frozen-op-internal; relative "
            "~1e-8).  The gate uses relative |delta|/n_isects <= 1e-6 and "
            "reports the raw mismatch count verbatim.",
            "run-to-run nondeterminism: iteration-1 loss is bit-identical "
            "across runs (forward deterministic) but the backward's "
            "atomicAdd reduction order injects ~1e-7 noise per iteration "
            "from iteration 2, amplifying at densification thresholds to "
            "~1% final-N spread between two frozen clean runs.  Trace-"
            "validity is judged with an intrinsic envelope (traced-clean "
            "divergence <= max within-group clean-clean / traced-traced "
            "divergence).",
            "Stage-5 audit export (d0_final_audit.npz, written in-process "
            "after the final eval): row_uid = final row->uid map (model."
            "_gaussian_ids; restore() does not checkpoint ids, so this is "
            "the only authoritative row<->uid linkage), per-row "
            "visibility_count / tiles_total / alpha_footprint over the 16 "
            "deterministic audit cameras, plus per-camera scalars.  RGB "
            "contribution proxy recorded unavailable (not exposed by the "
            "C0 renderer).  Descriptive only; no removal claim.",
        ],
    }
    for name, obj in (("D0_COST_ACCOUNTING.json", cost), ("D0_HEADROOM.json", head),
                      ("D0_TAIL_QUALITY.json", tail),
                      ("D0_SENSITIVITY_INIT_ROOTS.json",
                       {"artifact": "D0_SENSITIVITY_INIT_ROOTS.json",
                        "runs": {s: scenes_out[s]["sensitivity_init_roots"]
                                 for s in scenes}}),
                      ("D0_LINEAGE_SCHEMA.json", schema)):
        with open(os.path.join(ART, name), "w") as f:
            json.dump(obj, f, indent=2, default=str)
        print("wrote", name)

    print("STAGE3", "ACCOUNTING_INVALID" if invalid else head["verdict"])
    return 1 if invalid else 0


if __name__ == "__main__":
    sys.exit(main())
