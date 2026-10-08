"""VOS Phase-4 analysis: visibility gap statistics + traffic oracle inputs.

Reads the per-iteration visibility bitmap traces captured by vos_trace_driver.py
(no trainer modification), computes per-scene / per-stage statistics, and emits:

  VOS_VISIBILITY_GAPS.json  — visibility fractions, reactivation counts, gap
                              distributions (censoring-aware, fence-segment aware),
                              per-row visibility frequency, never-visible fractions.
  vos_visibility_series.npz — per-iteration series (visible count, reactivation
                              count, N) for downstream coupling with microbench.

Row-identity rules honored:
  * Densify/prune/reset run at every 100 iters in 1..15000 -> row identity only
    stable WITHIN a 100-iter stretch [k*100+1, (k+1)*100]; those windows are
    processed per-stretch, gap stats left-censored at each stretch start.
  * 15001..30000 has NO topology events -> stable identity; fence segments are
    bounded by eval iterations 20000/25000/30000 (materialization fences for
    shs/opacity). Max deferral K inside a segment = segment length.

Gap definition (matches VOS_EXACTNESS_MODEL): row reactivated at t (visible at
t, not at t-1) with last visible at u < t-1 has gap g = t - u - 1 deferred
zero-gradient steps to replay (steps u+1..t-1), then the real transition at t.
"""
import argparse
import json
import os

import numpy as np

SCENES = ["room", "bicycle", "garden"]
ELEMS = {"xyz": 3, "shs": 48, "opacity": 1, "scaling": 3, "rotation": 4}
DEFERRED = ["shs", "opacity"]          # EVENT_MATERIALIZABLE (v1 scope)
EAGER = ["xyz", "scaling", "rotation"]  # EAGER_REQUIRED
BYTES_PER_ELEM_ADAM = 28               # read grad+m+v+theta (16B), write m+v+theta (12B)
TAU_BYTES_PER_ACTIVE_ROW = 8           # read tau + write tau (int32 each)
DENSIFY_INTERVAL = 100


def load_meta(trace_dir):
    meta = json.load(open(os.path.join(trace_dir, "vos_trace_meta.json")))
    recs = meta["records"]  # [iter, N, nbytes, sh_degree]
    offsets = np.zeros(len(recs) + 1, dtype=np.int64)
    for i, r in enumerate(recs):
        offsets[i + 1] = offsets[i] + 16 + r[2]
    iters = np.array([r[0] for r in recs], dtype=np.int64)
    Ns = np.array([r[1] for r in recs], dtype=np.int64)
    nbs = np.array([r[2] for r in recs], dtype=np.int64)
    return meta, iters, Ns, nbs, offsets


def stretch_bounds(it_lo, it_hi, stretch):
    if stretch is None:
        return [(it_lo, it_hi)]
    out = []
    t = it_lo
    while t <= it_hi:
        end = min(((t - 1) // stretch + 1) * stretch, it_hi)
        out.append((t, end))
        t = end + 1
    return out


def analyze_window(trace_dir, it_lo, it_hi, stretch, label):
    """One analysis window; per-stretch processing honors row-identity breaks."""
    meta, iters, Ns, nbs, offsets = load_meta(trace_dir)
    fences = sorted(t for t in (20000, 25000, 30000) if it_lo < t <= it_hi) \
        if stretch is None else []

    gaps = []
    cens = 0
    react_per_iter = np.zeros(it_hi - it_lo + 1, dtype=np.int64)
    vis_per_iter = np.zeros(it_hi - it_lo + 1, dtype=np.int64)
    seg_fences = []
    never_list = []
    hist = np.zeros(20, dtype=np.int64)
    N_window = None

    with open(os.path.join(trace_dir, "vos_trace.bin"), "rb") as f:
        for (a, b) in stretch_bounds(it_lo, it_hi, stretch):
            sel = (iters >= a) & (iters <= b)
            idxs = np.nonzero(sel)[0]
            assert len(idxs) == b - a + 1, f"missing iters in [{a},{b}]"
            N = int(Ns[idxs[0]])
            assert int(Ns[idxs].max()) == N and int(Ns[idxs].min()) == N, \
                f"N not constant in stretch [{a},{b}]: {Ns[idxs].min()}..{Ns[idxs].max()}"
            if N_window is None:
                N_window = N
            else:
                assert stretch is not None or N == N_window, \
                    f"N changed inside stable window: {N_window} -> {N}"

            vis_counts = np.zeros(N, dtype=np.int32)
            last_vis = np.full(N, -1, dtype=np.int64)
            tau_end = np.full(N, a - 1, dtype=np.int64)  # materialized at stretch start
            n_iters = b - a + 1

            for ri in idxs:
                t = int(iters[ri])
                f.seek(int(offsets[ri]) + 16)
                raw = f.read(int(nbs[ri]))
                bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8),
                                     bitorder="big")[:N]
                bm = bits.astype(bool)
                wpos = t - it_lo
                idx = np.flatnonzero(bm)
                vis_per_iter[wpos] = len(idx)
                if len(idx):
                    vis_counts[idx] += 1
                    newly = idx[last_vis[idx] < t - 1]
                    react_per_iter[wpos] = len(newly)
                    if len(newly):
                        lv = last_vis[newly]
                        ok = lv >= 0
                        if ok.any():
                            gaps.append((t - lv[ok] - 1).astype(np.int64))
                        cens += int((~ok).sum())
                    last_vis[idx] = t
                    tau_end[idx] = t
                if t in fences:
                    deferred = tau_end < t
                    kf = (t - tau_end[deferred]).astype(np.int64)
                    seg_fences.append({
                        "fence_t": t, "n_rows_fenced": int(deferred.sum()),
                        "sum_K_fence": int(kf.sum()) if len(kf) else 0,
                        "K_p50": float(np.percentile(kf, 50)) if len(kf) else 0.0,
                        "K_p90": float(np.percentile(kf, 90)) if len(kf) else 0.0,
                        "K_max": int(kf.max()) if len(kf) else 0})
                    tau_end[:] = t
            # per-stretch row stats (identity valid only inside the stretch)
            never_list.append(int((vis_counts == 0).sum()) / N)
            hist += np.bincount(
                (vis_counts * 20 // n_iters).clip(0, 19), minlength=20)
            if stretch is not None:
                # rows carried across stretch boundaries keep identity in the
                # *dense* optimizer; we censor, per protocol.
                pass

    gaps = np.concatenate(gaps) if gaps else np.zeros(0, dtype=np.int64)
    n_iters_win = it_hi - it_lo + 1
    # tau_end after the last stretch/fence: rows never visible replay to the
    # window end only in stable windows (fence); censored otherwise.
    out = {
        "label": label, "it_lo": it_lo, "it_hi": it_hi, "N": int(N_window),
        "iters": n_iters_win,
        "N_note": "constant (stable window)" if stretch is None else
                  "grows with densification; per-stretch N",
        "vis_frac_mean": float(vis_per_iter.mean() / N_window),
        "vis_frac_p10": float(np.percentile(vis_per_iter, 10) / N_window),
        "vis_frac_p50": float(np.percentile(vis_per_iter, 50) / N_window),
        "vis_frac_p90": float(np.percentile(vis_per_iter, 90) / N_window),
        "vis_frac_min": float(vis_per_iter.min() / N_window),
        "vis_frac_max": float(vis_per_iter.max() / N_window),
        "reactivations_per_iter_mean": float(react_per_iter.mean()),
        "reactivations_per_iter_p50": float(np.percentile(react_per_iter, 50)),
        "reactivations_per_iter_p90": float(np.percentile(react_per_iter, 90)),
        "gap_count": int(len(gaps)),
        "gap_censored_count": cens,
        "gap_mean": float(gaps.mean()) if len(gaps) else None,
        "gap_p50": float(np.percentile(gaps, 50)) if len(gaps) else None,
        "gap_p75": float(np.percentile(gaps, 75)) if len(gaps) else None,
        "gap_p90": float(np.percentile(gaps, 90)) if len(gaps) else None,
        "gap_p95": float(np.percentile(gaps, 95)) if len(gaps) else None,
        "gap_p99": float(np.percentile(gaps, 99)) if len(gaps) else None,
        "gap_max": int(gaps.max()) if len(gaps) else None,
        "gap_sum_per_iter_mean": float(gaps.sum() / n_iters_win) if len(gaps) else 0.0,
        "never_visible_frac_per_stretch_mean": float(np.mean(never_list)),
        "row_vis_freq_hist_20bins": (hist / max(hist.sum(), 1)).tolist(),
        "fences": seg_fences,
    }
    series = {"vis_per_iter": vis_per_iter, "react_per_iter": react_per_iter,
              "N": np.full(n_iters_win, N_window, dtype=np.int64)}
    return out, series


def traffic_model(stats, label):
    """Byte/FLOP model for one window (v1 scope: defer shs+opacity only)."""
    N = stats["N"]
    n_act = stats["vis_frac_mean"] * N
    deferred_elems = sum(ELEMS[a] for a in DEFERRED)
    eager_elems = sum(ELEMS[a] for a in EAGER)
    dense_bytes = N * (deferred_elems + eager_elems) * BYTES_PER_ELEM_ADAM
    vos_bytes = (N * eager_elems * BYTES_PER_ELEM_ADAM
                 + n_act * deferred_elems * BYTES_PER_ELEM_ADAM
                 + n_act * TAU_BYTES_PER_ACTIVE_ROW)
    fence_sumK = sum(fc["sum_K_fence"] for fc in stats["fences"])
    fence_rows = sum(fc["n_rows_fenced"] for fc in stats["fences"])
    iters = stats["iters"]
    fence_K_per_iter = fence_sumK / iters
    fence_write_bytes_per_iter = (fence_rows * deferred_elems * 12) / iters
    react_K_per_iter = stats["gap_sum_per_iter_mean"]
    return {
        "label": label,
        "N": N, "n_active_mean": n_act,
        "deferred_elems_per_row": deferred_elems,
        "eager_elems_per_row": eager_elems,
        "dense_adam_bytes_per_iter": dense_bytes,
        "vos_bytes_per_iter": vos_bytes + fence_write_bytes_per_iter,
        "vos_byte_saving_frac":
            (dense_bytes - vos_bytes - fence_write_bytes_per_iter) / dense_bytes,
        "replay_steps_per_iter_reactivation": react_K_per_iter,
        "replay_steps_per_iter_fence_amortized": fence_K_per_iter,
        "replay_steps_per_iter_total": react_K_per_iter + fence_K_per_iter,
        "replay_flops_per_iter": (react_K_per_iter + fence_K_per_iter)
            * deferred_elems * 8.0,
        "tau_metadata_bytes_per_iter": n_act * TAU_BYTES_PER_ACTIVE_ROW,
        "fence_materialization_write_bytes_per_iter": fence_write_bytes_per_iter,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-root", default="/root/vos_traces")
    ap.add_argument("--out-dir", default="/root/vos_traces/analysis")
    ap.add_argument("--scenes", default="room,bicycle,garden")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    scenes = args.scenes.split(",")

    windows = [
        (4501, 5000, DENSIFY_INTERVAL, "at_5K"),
        (14501, 15000, DENSIFY_INTERVAL, "at_15K"),
        (29001, 30000, None, "at_30K"),
        (15001, 20000, None, "fence_seg_15K_20K"),
        (20001, 25000, None, "fence_seg_20K_25K"),
        (25001, 30000, None, "fence_seg_25K_30K"),
        (15001, 30000, None, "vos_region_15K_30K"),
    ]

    result = {"scenes": {}, "model": {
        "bytes_per_elem_adam": BYTES_PER_ELEM_ADAM,
        "elements": ELEMS, "deferred": DEFERRED, "eager": EAGER,
        "tau_bytes_per_active_row": TAU_BYTES_PER_ACTIVE_ROW,
        "fence_events_in_vos_region": [20000, 25000, 30000],
        "densify_until_iter": 15000,
    }}
    series_out = {}
    for scene in scenes:
        tdir = os.path.join(args.trace_root, f"{scene}_T30")
        result["scenes"][scene] = {}
        series_out[scene] = {}
        print(f"[analyze] {scene}: streaming trace ...", flush=True)
        for (lo, hi, stretch, label) in windows:
            st, se = analyze_window(tdir, lo, hi, stretch, label)
            result["scenes"][scene][label] = st
            result["scenes"][scene][label]["traffic"] = traffic_model(st, label)
            for k, v in se.items():
                series_out[scene][f"{label}__{k}"] = v
            print(f"  {label}: N={st['N']} vis={st['vis_frac_mean']:.4f} "
                  f"react/iter={st['reactivations_per_iter_mean']:.0f} "
                  f"gap p50={st['gap_p50']} p90={st['gap_p90']} p99={st['gap_p99']} "
                  f"max={st['gap_max']} never={st['never_visible_frac_per_stretch_mean']:.4f} "
                  f"sumK/iter={st['gap_sum_per_iter_mean']:.0f}", flush=True)
        np.savez_compressed(os.path.join(args.out_dir, "vos_visibility_series.npz"),
                            **{f"{s}__{k}": v for s, d in series_out.items()
                               for k, v in d.items()})

    with open(os.path.join(args.out_dir, "VOS_VISIBILITY_GAPS.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"WROTE {args.out_dir}/VOS_VISIBILITY_GAPS.json")


if __name__ == "__main__":
    main()
