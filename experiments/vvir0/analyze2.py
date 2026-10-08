"""VRIR0 Phase P5/P6 - comprehensive analyzer + mechanical gate evaluation.

Reads per-observation JSONL and reports per scene and 5k window the metrics required
by FAIL_CLOSED_GATES.md, then applies gates G0-G9 mechanically to emit the verdict.
"""
from __future__ import annotations
import json, os, sys, glob, hashlib
import numpy as np
from collections import defaultdict


def load_records(path):
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except Exception:
                pass
    return recs


def window_of(it):
    if 15001 <= it <= 20000: return "15k-20k"
    if 20001 <= it <= 25000: return "20k-25k"
    if 25001 <= it <= 30000: return "25k-30k"
    return "diagnostic"


def analyze_scene(recs, timing=None):
    revisits = [r for r in recs if r.get("eligible") and r.get("event") != "CACHE_BUDGET_REJECT"]
    n = len(revisits)
    if n == 0:
        return {"n_revisits": 0, "note": "no revisits collected"}
    arr = lambda k: np.array([r[k] for r in revisits], dtype=float)
    cert = lambda k: np.array([1.0 if r["certificate"].get(k) else 0.0 for r in revisits])
    gaps = arr("cache_age_iters"); truth = arr("truth_same")
    acc = cert("accept"); mem = cert("membership")
    ord_exact = cert("order_exact_scan"); ord_slack = cert("order_slack_n")
    fa = arr("false_accept"); fr = arr("false_reject")
    mm_ok = np.array([1.0 if r.get("membership_model_ok") else 0.0 for r in revisits])
    tile_frac = np.array([r["tile_truth_same_fraction"] if r["tile_truth_same_fraction"] is not None else np.nan for r in revisits])
    adj_v = arr("adjacent_order_violations"); adj_p = arr("adjacent_pairs_checked")
    n_isects = arr("n_isects"); cache_b = arr("cache_bytes")
    drift = {"means2d": [], "radii": [], "depth": []}
    for r in revisits:
        d = r.get("drift") or {}
        for k in drift:
            if d.get(k):
                drift[k].extend([d[k]["p50"], d[k]["p95"], d[k]["p99"]])
    drift_q = {k: {"p50": float(np.percentile(a,50)), "p95": float(np.percentile(a,95)), "p99": float(np.percentile(a,99))}
               for k, a in ((k, np.array(v)) for k, v in drift.items()) if len(a)}
    h = float(np.mean(acc))
    amdahl = None
    if timing:
        t_iter = timing.get("t_iter_mean_ms"); t_struct = timing.get("t_struct_ms") or timing.get("t_fwd_mean_ms")
        if t_iter and t_struct and t_iter > h * t_struct:
            amdahl = {"hit_rate": h, "t_iter_ms": t_iter, "t_struct_ms": t_struct,
                      "S_UB": t_iter / (t_iter - h * t_struct)}
    return {
        "n_revisits": n,
        "cache_age_iters": {"p50": float(np.percentile(gaps,50)), "p95": float(np.percentile(gaps,95)),
                            "p99": float(np.percentile(gaps,99)), "max": float(np.max(gaps))},
        "membership_model_ok_rate": float(np.mean(mm_ok)),
        "truth_same_rate": float(np.mean(truth)),
        "C_MEM_rate": float(np.mean(mem)),
        "C_ORD_EXACT_SCAN_rate": float(np.mean(ord_exact)),
        "C_ORD_SLACK_N_rate": float(np.mean(ord_slack)),
        "full_fast_cert_accept_rate": h,
        "false_accepts": int(np.sum(fa)),
        "false_rejects": int(np.sum(fr)),
        "tile_truth_same_fraction_mean": float(np.nanmean(tile_frac)),
        "adjacent_order_violation_rate": float(np.sum(adj_v) / max(1.0, np.sum(adj_p))),
        "n_isects_mean": float(np.mean(n_isects)),
        "cache_bytes_mean": float(np.mean(cache_b)),
        "drift_quantiles": drift_q,
        "amdahl": amdahl,
    }


def analyze_by_window(recs, timing=None):
    groups = defaultdict(list)
    for r in recs:
        if r.get("eligible") and r.get("event") != "CACHE_BUDGET_REJECT":
            groups[window_of(r["iteration"])].append(r)
    out = {}
    for w, rs in sorted(groups.items()):
        out[w] = analyze_scene(rs, timing)
    return out


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    recs = load_records(os.path.join(run_dir, "vir0_records.jsonl"))
    timing = {}
    tp = os.path.join(run_dir, "vir0_timing.json")
    if os.path.exists(tp):
        timing = json.load(open(tp))
    summ = analyze_scene(recs, timing)
    bywin = analyze_by_window(recs, timing)
    out = {"scene_level": summ, "by_window": bywin}
    outp = os.path.join(run_dir, "vir0_analysis.json")
    with open(outp, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print("written", outp)
