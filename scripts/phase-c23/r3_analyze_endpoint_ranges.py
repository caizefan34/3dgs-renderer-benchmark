#!/usr/bin/env python3
"""R3 conservative analysis of C22 endpoint data.

Distinguishes logical tile intersections from a proxy for executed per-pixel
range traversal. It deliberately does NOT call endpoint suffixes new removable
work: gsplat baseline early termination has already avoided them. Meaningful
committed-contribution counts are unavailable through this instrumentation.
"""
import argparse, json
from pathlib import Path
import numpy as np

def stat(x):
    a=np.asarray(x,dtype=float)
    return {"mean":float(a.mean()),"p50":float(np.percentile(a,50)),"p90":float(np.percentile(a,90)),"p95":float(np.percentile(a,95)),"p99":float(np.percentile(a,99)),"min":float(a.min()),"max":float(a.max())}
def group(n): return "HIGH" if n>10000 else "MEDIUM" if n>3000 else "LOW"
p=argparse.ArgumentParser();p.add_argument("--out",required=True);p.add_argument("inputs",nargs="+");a=p.parse_args()
rows=[]; by={"LOW":[],"MEDIUM":[],"HIGH":[]}
for fn in a.inputs:
 d=json.load(open(fn))
 for c in d["camera_records"]:
  for t in c["tile_classification"]:
   n=t["total_intersections"]; px=t["pixel_count"]; r=t["useful_prefix_ratio_mean"]
   # Logical pixel-range checks absent baseline termination, and proxy actually reached before termination.
   logical=n*px; executed_proxy=logical*r
   row={"camera":c["camera"],"tile":t["tile"],"density":group(n),"logical_intersections_per_tile":n,"pixels":px,"logical_pixel_gaussian_checks":logical,"executed_before_baseline_termination_proxy":executed_proxy,"executed_over_logical":r,"baseline_avoided_suffix_fraction":1-r}
   rows.append(row);by[row["density"]].append(row)
out={"schema_version":1,"phase":"R3 tile-level work amplification","instrumentation":"Uses C22 actual last_ids aggregates; no kernel change. executed_before_baseline_termination_proxy is not a committed-contribution count.","semantic_limits":"A=logical tile intersections. B=per-pixel range positions reached before gsplat's existing termination. C=meaningful committed contributions is NOT observed. Therefore WA=B/C cannot be reported without further instrumentation.","rows":rows,"by_density":{g:{"n_tiles":len(v),"intersection_count":stat([r["logical_intersections_per_tile"] for r in v]) if v else None,"executed_over_logical":stat([r["executed_over_logical"] for r in v]) if v else None,"baseline_avoided_suffix_fraction":stat([r["baseline_avoided_suffix_fraction"] for r in v]) if v else None} for g,v in by.items()},"decision":"MAYBE","decision_basis":"Endpoint data establishes a stable large logical-to-reached-range gap, but not WA relative to material contributions. A stable execution-work amplification law is therefore not yet established."}
Path(a.out).parent.mkdir(parents=True,exist_ok=True);json.dump(out,open(a.out,"w"),indent=2);print("saved",a.out,"tiles",len(rows))