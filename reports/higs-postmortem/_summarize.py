import json, glob, os, statistics, math
from collections import defaultdict

def load_grid(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        try:
            r = json.load(open(f))
            out[(r.get("method"), r.get("scene"), r.get("seed"))] = r
        except Exception:
            pass
    return out

def mean(xs): return statistics.fmean(xs) if xs else float("nan")
def gmean(xs):
    xs = [x for x in xs if x and math.isfinite(x) and x > 0]
    return math.exp(sum(math.log(x) for x in xs)/len(xs)) if xs else float("nan")

paths = [("PAPER", r"artifacts\training-paper\results"), ("ALL", r"artifacts\training-all\results")]
for label, d in paths:
    g = load_grid(d)
    print("\n===== %s: %d =====" % (label, len(g)))
    by_method = defaultdict(list)
    for (m, s, seed), r in g.items():
        by_method[m].append(r)
    print("methods:", sorted(by_method.keys()))
    for m in sorted(by_method):
        rs = by_method[m]
        walls = [r["performance"]["wall_time_seconds"] for r in rs]
        ttqs = [r["performance"]["time_to_quality_seconds"] for r in rs]
        ps = [r["quality"]["psnr_db"] for r in rs]
        mems = [r["resources"]["peak_memory_mb"] if "peak_memory_mb" in r["resources"] else r["resources"]["peak_gpu_memory_mb"] if "peak_gpu_memory_mb" in r["resources"] else r["resources"].get("peak_gpu_memory_mib", 0) for r in rs]
        gaus = [r["resources"]["final_gaussian_count"] for r in rs]
        print("  %-16s n=%d wall_gmean=%.1fs ttq_gmean=%.1fs psnr_mean=%.2f mem_mean=%.1fMiB gauss_mean=%.0f" % (m, len(rs), gmean(walls), gmean(ttqs), mean(ps), mean(mems), mean(gaus)))
    def scene_means(m):
        d2 = defaultdict(list)
        for (mm, s, seed), r in g.items():
            if mm == m:
                d2[s].append(r["performance"]["wall_time_seconds"])
        return {s: mean(v) for s, v in d2.items()}
    print("  wall scene-mean ratios (gslam / higher):")
    for tgt in ["higs_full", "higs_proposed"]:
        if "gslam" in by_method and tgt in by_method:
            sm_g, sm_h = scene_means("gslam"), scene_means(tgt)
            common = sorted(set(sm_g) & set(sm_h))
            ratios = [sm_g[s]/sm_h[s] for s in common if sm_h[s] > 0]
            print("    gslam/%s gmean = %.4fx (%d scenes)" % (tgt, gmean(ratios), len(common)))
