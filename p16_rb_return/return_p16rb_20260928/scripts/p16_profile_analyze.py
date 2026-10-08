#!/usr/bin/env python3
"""P16-RB: turn raw torch.profiler chrome traces into the profile JSON census.

Attribution protocol (p16-prof-v1):
  - iteration regions  <- P16.iter <idx> user_annotation ranges
  - span of a kernel   <- cuda_runtime correlation id -> launch timestamp -> innermost
                          containing P16.* span; fallback: kernel execution ts with
                          time slack against span [ts-50us, ts+dur+1500us]
  - phase of a kernel  <- name table, with span-context overrides (e.g. renderCUDA in a
                          'backward' span -> raster_backward; anything inside a
                          density span -> density_control); unclassified kernels fall back
                          to their span's default phase, else 'other'
  - per phase: per-iteration kernel time sum (or CPU span duration when a phase has no
    kernels in that iteration); phase value = median over window iterations
  - other = median over iterations of max(0, iteration_wall - sum(phase values))
Stdlib only.
"""
import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

PHASES = ["data", "projection_preprocess", "intersection_sort", "raster_forward", "loss",
          "raster_backward", "optimizer", "backward_optimizer_fused", "density_control", "other"]

SPAN_DEFAULT = {
    "data": "data",
    "raster_forward": "raster_forward",
    "loss": "loss",
    "backward": "raster_backward",
    "optimizer": "optimizer",
    "density": "density_control",
    "density_stats": "density_control",
    "cb_densify": "density_control",
    "cb_morton_ordering": "density_control",
    "cb_recompute_3d_filter": "density_control",
    "cb_reset_opacities": "density_control",
    "cb_reset_opacities_extra": "density_control",
    "cb_hard_pruning": "density_control",
}

SORT_TOKENS = ("extract_instance_ranges_cu", "create_instances_cu", "apply_depth_ordering_cu",
               "extract_bucket_counts", "duplicateWithKeys", "identifyTileRanges", "perTileBucketCount",
               "checkFrustum", "RadixSort", "SortPairs", "radix")
DENSITY_TOKENS = ("compute_scores_cu", "relocation_cu", "add_noise_cu", "update_3d_filter_cu")


def classify(name, span):
    if span in ("density", "density_stats"):
        return "density_control"
    n = name
    if "blend_backward" in n:
        return "raster_backward"
    if "preprocess_backward_cu" in n:
        return "raster_backward"
    if "computeCov2DCUDA" in n:
        return "raster_backward"
    if "PerGaussianRender" in n:
        return "raster_backward"
    if "renderCUDA" in n:
        return "raster_backward" if span == "backward" else "raster_forward"
    if "preprocessCUDA" in n:
        return "raster_backward" if span == "backward" else "projection_preprocess"
    if "preprocess_cu" in n:
        return "projection_preprocess"
    if "blend_cu" in n:
        return "raster_forward"
    for t in SORT_TOKENS:
        if t in n:
            return "intersection_sort"
    if "adam" in n.lower():
        return "backward_optimizer_fused" if span == "backward" else "optimizer"
    if "ssim" in n.lower():
        return "loss"
    for t in DENSITY_TOKENS:
        if t in n:
            return "density_control"
    d = SPAN_DEFAULT.get(span)
    return d if d else "other"


def analyze_stage(rawdir: Path, stage: str, base: int):
    trace_path = rawdir / f"trace_{stage}.json"
    if not trace_path.exists():
        return {"status": "TRACE_MISSING", "stage": stage}
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    ev = trace.get("traceEvents") if isinstance(trace, dict) else trace
    if ev is None:
        return {"status": "TRACE_UNREADABLE", "stage": stage}

    iters, spans, kern, rt = [], [], [], {}
    for e in ev:
        cat = e.get("cat")
        nm = e.get("name") or ""
        if cat == "user_annotation":
            if nm.startswith("P16.iter "):
                try:
                    iters.append({"idx": int(nm.split(" ", 1)[1]), "ts": e["ts"], "dur": e.get("dur") or 0})
                except (ValueError, KeyError):
                    pass
            elif nm.startswith("P16."):
                spans.append({"name": nm[4:], "ts": e["ts"], "dur": e.get("dur") or 0})
        elif cat in ("kernel", "gpu_memcpy"):
            kern.append(e)
        elif cat == "cuda_runtime":
            c = (e.get("args") or {}).get("correlation")
            if c is not None:
                rt[c] = e["ts"]
    iters.sort(key=lambda x: x["ts"])
    spans.sort(key=lambda x: x["ts"])
    bounds = [(it["ts"], it["ts"] + it["dur"], it["idx"], it["dur"]) for it in iters]

    def iter_of(ts):
        for lo, hi, idx, _d in bounds:
            if lo <= ts < hi:
                return idx
        best = None
        for lo, _hi, idx, _d in bounds:
            if lo <= ts:
                best = idx
        return best

    spans_by_iter = {}
    for s in spans:
        spans_by_iter.setdefault(iter_of(s["ts"]), []).append(s)

    def span_of(ts, it):
        strict = None
        for s in spans_by_iter.get(it, ()):
            if s["ts"] <= ts <= s["ts"] + s["dur"]:
                if strict is None or s["dur"] < strict["dur"]:
                    strict = s
        if strict is not None:
            return strict["name"]
        slack = None
        for s in spans_by_iter.get(it, ()):
            if s["ts"] - 50 <= ts <= s["ts"] + s["dur"] + 1500:
                if slack is None or s["dur"] < slack["dur"]:
                    slack = s
        return slack["name"] if slack else None

    per_iter_k = {idx: {p: 0.0 for p in PHASES} for _, _, idx, _d in bounds}
    per_iter_s = {idx: {} for _, _, idx, _d in bounds}
    unclassified = Counter()
    corr_hits = corr_total = 0
    for e in kern:
        dur = e.get("dur") or 0.0
        ts = e["ts"]
        c = (e.get("args") or {}).get("correlation")
        corr_total += 1
        launch = rt.get(c) if c is not None else None
        if launch is not None:
            corr_hits += 1
        use = launch if launch is not None else ts
        it = iter_of(use)
        if it is None:
            continue
        sp = span_of(use, it)
        v = classify(e.get("name") or "", sp)
        per_iter_k[it][v] += dur
        if v == "other":
            unclassified[e.get("name") or "?"] += 1

    for s in spans:
        it = iter_of(s["ts"])
        if it is None:
            continue
        p = SPAN_DEFAULT.get(s["name"])
        if p:
            per_iter_s[it][p] = per_iter_s[it].get(p, 0.0) + s["dur"]

    n = len(bounds)
    vals = {p: [] for p in PHASES}
    other_res = []
    for _, _, idx, dur in bounds:
        covered = 0.0
        for p in PHASES:
            if p == "other":
                continue
            k = per_iter_k[idx][p]
            v = k if k > 0 else per_iter_s[idx].get(p, 0.0)
            vals[p].append(v)
            covered += v
        other_res.append(max(0.0, dur - covered))
    for p in PHASES:
        if p != "other":
            continue
        vals[p] = other_res

    def med_ms(xs):
        return round(statistics.median(xs) / 1000.0, 4) if xs else None

    def mean_ms(xs):
        return round(statistics.fmean(xs) / 1000.0, 4) if xs else None

    def max_ms(xs):
        return round(max(xs) / 1000.0, 4) if xs else None

    totals = [d for _, _, _, d in bounds]
    window_idxs = [idx for _, _, idx, _d in bounds]
    return {
        "status": "OK",
        "stage": stage,
        "n_iterations": n,
        "window_iterations": [min(window_idxs), max(window_idxs)] if window_idxs else None,
        "base": base,
        "median_ms": {p: med_ms(vals[p]) for p in PHASES} | {"total": med_ms(totals)},
        "mean_ms": {p: mean_ms(vals[p]) for p in PHASES} | {"total": mean_ms(totals)},
        "max_ms": {p: max_ms(vals[p]) for p in PHASES} | {"total": max_ms(totals)},
        "evidence": {
            "correlation_coverage": round(corr_hits / corr_total, 4) if corr_total else None,
            "n_kernel_events": corr_total,
            "unclassified_kernel_names": [k for k, _ in unclassified.most_common(20)],
            "unclassified_kernel_events": int(sum(unclassified.values())),
            "trace": str(trace_path),
            "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
            "attribution_protocol": "p16-prof-v1 (correlation launch-ts -> span; fallback exec-ts containment; span-context phase overrides)",
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rawdir", required=True, help="dir with trace_<stage>.json + meta.json")
    ap.add_argument("--method", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--base", type=int, default=None, help="defaults to meta.json base")
    args = ap.parse_args()

    rawdir = Path(args.rawdir)
    meta = {}
    if (rawdir / "meta.json").exists():
        meta = json.loads((rawdir / "meta.json").read_text(encoding="utf-8"))
    base = args.base if args.base is not None else int(meta.get("base", 0))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stages = [s for s in ("mid", "final") if (rawdir / f"trace_{s}.json").exists()]
    if not stages:
        print(json.dumps({"error": "no traces found", "rawdir": str(rawdir)}, indent=2))
        return 1
    for stage in stages:
        res = analyze_stage(rawdir, stage, base)
        res["method"] = args.method
        res["scene"] = args.scene
        out = outdir / f"{args.scene}_{stage}.json"
        out.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(json.dumps({"out": str(out), "stage": stage, "status": res.get("status"),
                          "n_iterations": res.get("n_iterations")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
