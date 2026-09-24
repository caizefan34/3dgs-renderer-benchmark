#!/usr/bin/env python3
"""C21 N6/N8 analysis from C19 raw tile decomposition; no renderer modification."""
from __future__ import annotations
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser();p.add_argument('--candidate',choices=['N6','N8'],required=True);p.add_argument('--out',required=True);a=p.parse_args()
 raw=json.loads((ROOT/'results/phase-c19/c19-2_reproducibility.json').read_text())['tile_scaling']
 rows=[]
 for r in raw:
  # C19 did not isolate radix sort; sort bound is explicitly unavailable rather than invented.
  rows.append({'tile_size':r['tile_size'],'total_intersections':r['total_intersections'],'intersect_time_ms':r['isect_tiles_ms'],'sort_time_ms':'UNAVAILABLE: C19 measured combined isect_tiles plus CUB sort','raster_time_ms':r['rasterization_ms'],'forward_time_ms':r['full_forward_ms'],'raster_ns_per_intersection':r['rasterizer_ns_per_intersection'],'mean_intersections_per_tile':r['mean_ints_per_tile'],'p99_intersections_per_tile':r['p99'],'active_tiles':r['active_tiles']})
 if a.candidate=='N6':
  conclusion='The component balance changes sharply with tile density: smaller tiles pay substantially more intersection/sort-input work; larger tiles lower intersections but increase raster cost per intersection. This establishes a coupled trade-off, but one room/camera workload does not establish a stable phase-map boundary.'
  verdict='MAYBE'
 else:
  conclusion='Sort input changes 9.69x (5,465,619 at tile8 to 564,323 at tile32) while density rises 168.69 to 276.63 intersections/tile. C19 does not isolate radix timing by density regime, so it cannot establish a workload-dependent sorting organization.'
  verdict='DROP'
 out={'schema_version':1,'candidate':a.candidate,'timestamp_utc':datetime.now(timezone.utc).isoformat(),'source_raw_file':'results/phase-c19/c19-2_reproducibility.json','rows':rows,'source_limitations':['C19 supplies combined isect_tiles timing, not isolated radix-sort timing.','Single scene/camera tile sweep does not identify independent workload regimes.'],'conclusion':conclusion,'verdict':verdict}
 q=Path(a.out);q.parent.mkdir(parents=True,exist_ok=True);q.write_text(json.dumps(out,indent=2));print(q)
if __name__=='__main__':main()
