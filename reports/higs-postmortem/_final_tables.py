import json, glob, os, statistics, math
from collections import defaultdict

def clean(x):
    if isinstance(x, str):
        try: return float(x)
        except Exception: return float("nan")
    try: return float(x)
    except Exception: return float("nan")

def mean(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return statistics.fmean(xs) if xs else float("nan")

def gmean(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x) and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")

def load_grid(d):
    rows = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        r = json.load(open(f))
        rows[(r.get("method"), r.get("scene"), r.get("seed"))] = r
    return rows

g_paper = load_grid(r"artifacts\training-paper\results")
met_arr = sorted({m for (m, _, _) in g_paper})
scn_arr = sorted({s for (_, s, _) in g_paper})
agg = defaultdict(lambda: defaultdict(dict))
for (m, s, seed), r in g_paper.items():
    for key in ["wall_time_seconds", "time_to_quality_seconds", "psnr", "ssim", "lpips", "final_gaussian_count", "peak_gpu_memory_mib"]:
        v = None
        if key == "psnr":
            v = r.get("quality", {}).get("psnr_db")
        elif key in ("ssim", "lpips"):
            v = r.get("quality", {}).get(key)
        elif key in ("final_gaussian_count", "peak_gpu_memory_mib"):
            res = r.get("resources", {})
            v = res.get(key, res.get("peak_memory_mib"))
        else:
            v = r.get("performance", {}).get(key)
        if v is not None:
            agg[(m, s)].setdefault(key, []).append(clean(v))

print("### master grid (psnr key is psnr_db)")
hdr = "%-26s" % "scene"
for m in met_arr:
    hdr += " | %-12s|%9s|%9s|%9s|%9s|%9s" % (m[:10], "w_s", "ttq_s", "psnr", "ssim", "gauss")
print(hdr)
for s in scn_arr:
    line = "%-26s" % s
    for m in met_arr:
        d_ = dict(agg.get((m, s), {}))
        w = mean(d_.get("wall_time_seconds", []))
        t = mean(d_.get("time_to_quality_seconds", []))
        p = mean(d_.get("psnr", []))
        ss = mean(d_.get("ssim", []))
        g = mean(d_.get("final_gaussian_count", []))
        line += " | %-12s|%9.1f|%9.1f|%9.2f|%9.4f|%9.0f" % (m[:10], w, t, p, ss, g)
    print(line)

print("\ngeomeans:")
for m in met_arr:
    ws = []
    for s in scn_arr:
        rs = [r for (mm, ss, _), r in g_paper.items() if mm == m and ss == s]
        if rs:
            ws.append(mean([clean(r["performance"]["wall_time_seconds"]) for r in rs]))
    if len(ws) > 1:
        print("  %-16s wall_gmean=%7.1fs" % (m, gmean(ws)))
    # ttq
    ts = []
    for s in scn_arr:
        rs = [r for (mm, ss, _), r in g_paper.items() if mm == m and ss == s]
        if rs:
            ts.append(mean([clean(r["performance"]["time_to_quality_seconds"]) for r in rs]))
    if len(ts) > 1:
        print("  %-16s ttq_gmean =%7.1fs" % (m, gmean(ts)))

# Stage breakdown from logs
print("\n## STAGE BREAKDOWN from training logs")
files = sorted(glob.glob(r"results/training\*.json"))
scene_stage = defaultdict(lambda: defaultdict(list))
for f in files:
    r = json.load(open(f))
    ml = r.get("metrics_log", [])
    scene = (r.get("config") or {}).get("scene", "?")
    tile = (r.get("config") or {}).get("tile_size", "?")
    for e in ml:
        for k, v in e.items():
            if isinstance(k, str) and isinstance(v, (int, float)) and v > 0:
                if k.endswith("_ms"):
                    scene_stage[scene][k].append(v)
for scene in sorted(scene_stage):
    print("\nscene:", scene, "| recorded stages:", sorted(set(scene_stage[scene])))
    st_ = dict(scene_stage[scene])
    maxes = {k: max(v) for k, v in st_.items()}
    total = sum(v for f_ in ["fwd_ms", "bwd_ms", "opt_ms", "topology_ms"] if (v := maxes.get(f_)))
    for f_ in ["fwd_ms", "bwd_ms", "opt_ms", "topology_ms"]:
        if f_ in maxes:
            print("   %-14s accum=%8.1f ms  share=%.1f%%" % (f_, maxes[f_], 100 * maxes[f_] / total if total else 0))

