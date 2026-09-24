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

def stdev(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return statistics.stdev(xs) if len(xs) > 1 else 0.0

def gmean(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x) and x > 0]
    return math.exp(sum(math.log(x) for x in xs)/len(xs)) if xs else float("nan")

def load_grid(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        r = json.load(open(f))
        out[(r.get("method"), r.get("scene"), r.get("seed"))] = r
    return out

g = load_grid(r"artifacts\training-paper\results")
methods = sorted({m for (m,_,_) in g})
scenes  = sorted({s for (_,s,_) in g})
print("methods in corpus:", methods)

def mono(fn):
    return fn if not isinstance(fn, bool) else None

rows = []
for m in methods:
    for s in scenes:
        rs = [r for (mm,ss,_), r in g.items() if mm==m and ss==s]
        if not rs:
            continue
        w  = [clean(r["performance"]["wall_time_seconds"]) for r in rs]
        t  = [clean(r["performance"]["time_to_second"]) if False else clean(r["performance"]["time_to_quality_seconds"]) for r in rs]
        p  = [clean(r["quality"]["psnr_db"]) for r in rs]
        si = [clean(r["quality"]["ssim"]) for r in rs]
        lp = [clean(r["quality"]["lpips"]) for r in rs]
        ga = [clean(r["resources"]["final_gaussian_count"]) for r in rs]
        rows.append((m, s, mean(w), stdev(w), mean(t), mean(p), mean(si), mean(lp), mean(ga)))
        print("%-24s %-14s wall=%8.1f+-%7.1f  ttq=%8.1f  psnr=%.2f ssim=%.4f lpips=%.4f gauss=%9.0f" %
              (s, m[:14], mean(w), stdev(w), mean(t), mean(p), mean(si), mean(lp), mean(ga)))

# group by scene: mean across seeds of wall, plus ratios vs gsplat
print("\n### SCENE-LEVEL RATIOS (gsplat = 1.0)")
scene_means = defaultdict(dict)
for m,s,w,sw,ttq,p,si,lp,ga in rows:
    scene_means[s][m] = (w,ttq,p,si,lp,ga)
for s in scenes:
    if "gsplat" not in scene_means[s]:
        continue
    g_w = scene_means[s]["gsplat"][0]
    for m in methods[1:]:
        if m not in scene_means[s]:
            continue
        h_w = scene_means[s][m][0]
        print("  %-24s %-14s wall ratio=%.4fx  (%.0fs vs %.0fs)" % (s, m[:14], g_w/h_w, g_w, h_w))
    # seed-level ratios
    rs_g = [clean(r["performance"]["wall_time_seconds"]) for (mm,ss,_), r in g.items() if mm==m and ss==s]
# geomean across scenes of per-scene mean ratios
for m in methods[1:]:
    rts = []
    for s in scenes:
        if "gsplat" in scene_means[s] and m in scene_means[s]:
            g_w = scene_means[s]["gsplat"][0]; h_w = scene_means[s][m][0]
            if g_w and h_w: rts.append(g_w/h_w)
    if rts:
        print("  GEOMEAN wall speedup %s = %.4fx over %d scenes" % (m, math.exp(sum(math.log(x) for x in rts)/len(rts)), len(rts)))
# seed-level pairwise geomean speedup
print("\n### SEED-LEVEL pairwise speedup (all pairs of seeds, per scene, geomean)")
pair_all = []
for m in methods[1:]:
    pair = []
    for s in scenes:
        rg = [r for (mm,ss,seed), r in g.items() if mm=="gsplat" and ss==s]
        rh = [r for (mm,ss,seed), r in g.items() if mm==m and ss==s]
        if not rg or not rh:
            continue
        seeds = sorted({r["seed"] for r in rg} & {r["seed"] for r in rh})
        for seed in seeds:
            wg = clean(rg[0]["performance"]["wall_time_seconds"]) if False else None
        # just use means
        wg = mean([clean(r["performance"]["wall_time_seconds"]) for r in rg])
        wh = mean([clean(r["performance"]["wall_time_seconds"]) for r in rh])
        if wg and wh: pair.append(wg/wh)
    if pair:
        print("  %-14s seed-mean geomean=%.4fx (%d scenes)" % (m, math.exp(sum(math.log(x) for x in pair)/len(pair)), len(pair)))
