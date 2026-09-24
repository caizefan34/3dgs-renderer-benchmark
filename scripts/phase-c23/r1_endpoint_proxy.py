#!/usr/bin/env python3
"""R1 conservative fetch-screening proxy.

Uses C23 batch activity measurements.  This deliberately separates CTA-level
attribute loads from pixel-level evaluations: when a tile batch has any active
pixel, the baseline CTA must load its Gaussian attributes; the existing data
cannot identify whether every fetched attribute makes a material color change.
"""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('inputs',nargs='+');a=p.parse_args()
by={g:[] for g in ('LOW','MEDIUM','HIGH')}
for fn in a.inputs:
 d=json.load(open(fn))
 for c in d['camera_records']:
  for g,gd in c['density_groups'].items():
   for t in gd['tile_records']:
    curve=np.asarray(t['activity_curve'],float)
    # activity_curve is sampled at sorted-range fractions. A CTA-level loader
    # remains necessary while any lane is active; below is a lane-side proxy,
    # not a count of globally fetched attributes.
    by[g].append({'camera':c['camera'],'tile':t['tile'],'intersections':t['total_intersections'],'mean_active_lane_fraction':float(curve.mean()),'tail_active_lane_fraction':float(t['active_lane_fraction_at_90pct']),'tail_low_util_warp_fraction':float(t['warps_le_4_active_lanes_at_90pct'])})
def st(v,k):
 x=np.asarray([r[k] for r in v],float);return {'mean':float(x.mean()),'p50':float(np.percentile(x,50)),'p90':float(np.percentile(x,90)),'p95':float(np.percentile(x,95)),'p99':float(np.percentile(x,99)),'min':float(x.min()),'max':float(x.max())} if len(v) else None
out={'schema_version':1,'phase':'R1 contribution-gated attribute fetch','instrumentation':'Derived solely from real C23 last_ids activity curves; no kernel modification or attribute-gating experiment.','definitions':{'fetched':'CTA-level global-to-shared attribute load for a Gaussian batch while any pixel lane remains active. Not directly counted.','evaluated':'pixel-side Gaussian test before its actual termination endpoint. Not separately counted.','committed':'Gaussian that passes gsplat contribution threshold and updates compositing. Not separately counted by stock output.'},'by_density':{g:{'n_tiles':len(v),'mean_active_lane_fraction_proxy':st(v,'mean_active_lane_fraction'),'tail_active_lane_fraction_at_90pct':st(v,'tail_active_lane_fraction'),'tail_low_util_warp_fraction_at_90pct':st(v,'tail_low_util_warp_fraction')} for g,v in by.items()},'decision':'MAYBE','decision_basis':'Activity measurements establish sparse lane-side demand late in ranges, but they cannot isolate global attribute fetches versus material committed contributions. Exact useful_attribute_work/total_attribute_work needs a minimal counter in the kernel; no attribute fetch gating or optimization was implemented.'}
Path(a.out).parent.mkdir(parents=True,exist_ok=True);json.dump(out,open(a.out,'w'),indent=2);print('saved',a.out)