import json, glob, os, statistics
from collections import defaultdict

def load_grid(d):
    out = {}
    for f in glob.glob(os.path.join(d, "*.json")):
        r = json.load(open(f))
        out[(r.get("method"), r.get("scene"), r.get("seed"))] = r
    return out

for label, d in [
    ("PAPER", r"artifacts\training-paper\results"),
    ("ALL", r"artifacts\training-all\results"),
]:
    g = load_grid(d)
    print(f"\n== {label}: {len(g)} ==")
    for (m, s, seed), r in sorted(g.items()):
        p = r.get("performance", {})
        q = r.get("quality", {})
        res = r.get("resources", {})
        print(f"{str(m).strip():<14s} {str(s).strip():<28s} wall={p.get('wall_time_seconds',0):.1f}s ttq={p.get('time_to_quality_seconds',0):.1f}s psnr={q.get('psnr_df', q.get('psnr_db', 0)):.2f} gauss={res.get('final_gaussian_count',0):,}")
