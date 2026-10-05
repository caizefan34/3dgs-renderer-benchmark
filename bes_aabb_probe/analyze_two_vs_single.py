#!/usr/bin/env python
"""Fair two-stage vs single-stage AABB-reject comparison.
Single-stage baseline (aabb_probe mode 0) is the gsplat staged walk = common reference.
Two-stage baseline (aabb2_probe mode 0) is a NAIVE global walk (slow) -> NOT fair reference.
Fair speedup of two-stage = single_stage_baseline / two_stage_ts.
Both kernels produce bit-identical output to the true render (maxdiff=0), so the
common reference is valid.
"""
import json, os, sys

SCENES = ["room", "bicycle", "garden"]
SINGLE_DIR = "bes_aabb_probe"
TWO_DIR = sys.argv[1] if len(sys.argv) > 1 else "bes_aabb2_tmp"

def load(d, name):
    p = os.path.join(d, name)
    if not os.path.exists(p):
        return None
    return json.load(open(p))

rows = []
for sc in SCENES:
    s = load(SINGLE_DIR, "aabb_%s.json" % sc)
    t = load(TWO_DIR, "aabb2_%s.json" % sc)
    if s is None:
        print("MISSING single %s" % sc); continue
    sa = s["agg"]; sb = sa["baseline_ms"]; st = sa["tile_ms"]; sf = sa["floor_ms"]
    sp_single = sb / st
    if t is None:
        print("MISSING two %s" % sc); continue
    ta = t["agg"]; tt = ta["ts_ms"]; ttsp = ta["tsp_ms"]; tf = ta["floor_ms"]
    sp_two = sb / tt
    sp_two_tsp = sb / ttsp
    rows.append((sc, sb, st, sp_single, tt, sp_two, ttsp, sp_two_tsp, sf, tf, ta.get("max_abs_diff_ts"), ta.get("max_abs_diff_tsp")))

print()
print("%-8s %10s %10s %8s | %10s %8s %8s | %9s %9s | %8s %8s" % (
    "scene","base_ms","s_tile_ms","sp_s","ts_ms","sp_two_fair","sp_tsp","s_floor","t_floor","md_ts","md_tsp"))
print("-"*105)
for r in rows:
    sc,sb,st,sp_s,tt,sp_two,ttsp,sp_two_tsp,sf,tf,mts,mtsp = r
    print("%-8s %10.1f %10.1f %7.3fx | %10.1f %7.3fx %7.3fx | %9.2f %9.2f | %.1e %.1e" % (
        sc,sb,st,sp_s,tt,sp_two,sp_two_tsp,sf,tf,mts,mtsp))
print()
# Totals
SB=sum(r[1] for r in rows); ST=sum(r[2] for r in rows); TT=sum(r[4] for r in rows)
TSPS=sum(r[6] for r in rows); SF=sum(r[8] for r in rows); TF=sum(r[9] for r in rows)
print("AGG (sum over scenes, common ref = single-stage staged baseline):")
print("  single-stage AABB reject : %.1f -> %.1f ms  = %.3fx" % (SB, ST, SB/ST))
print("  two-stage    AABB reject : %.1f -> %.1f ms  = %.3fx" % (SB, TT, SB/TT))
print("  floor: single %.1f ms  two %.1f ms  (two/single=%.3f)" % (SF, TF, TF/SF))
