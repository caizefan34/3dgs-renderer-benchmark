from __future__ import annotations
import argparse, json, re
from collections import defaultdict
from pathlib import Path
import numpy as np
from p13x_probe_core import compare_snapshots

PAT = re.compile(r"(?P<scene>.+)_a(?P<anchor>\d+)_i(?P<iter>\d+)\.npz$")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--snapshots', required=True)
    ap.add_argument('--scene', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--lags', nargs='+', type=int, default=[250,500,1000,2500])
    args=ap.parse_args()
    files=defaultdict(dict)
    for p in Path(args.snapshots).glob(f"{args.scene}_a*_i*.npz"):
        m=PAT.match(p.name)
        if m:
            files[int(m['anchor'])][int(m['iter'])]=p
    rows=[]
    for aid, by_iter in sorted(files.items()):
        for i0,p0 in sorted(by_iter.items()):
            for lag in args.lags:
                p1=by_iter.get(i0+lag)
                if p1 is None: continue
                r=compare_snapshots(p0,p1)
                r.update(anchor_id=aid, iter_a=i0, iter_b=i0+lag, lag=lag)
                rows.append(r)
    def summary_for(lag):
        rs=[r for r in rows if r['lag']==lag]
        if not rs: return {'n_pairs':0}
        keys=['visible_jaccard','visible_retained_fraction','visible_entered_fraction',
              'membership_jaccard','unchanged_membership_tile_fraction',
              'unchanged_full_order_tile_fraction','order_inversion_rate_median_tile',
              'order_inversion_rate_weighted']
        out={'n_pairs':len(rs)}
        for k in keys:
            a=np.asarray([r[k] for r in rs],dtype=float)
            out[k]={'median':float(np.median(a)),'mean':float(np.mean(a)),
                    'p10':float(np.percentile(a,10)),'p90':float(np.percentile(a,90)),
                    'min':float(np.min(a)),'max':float(np.max(a))}
        return out
    result={'scene':args.scene,'lags':{str(l):summary_for(l) for l in args.lags},'pairs':rows}
    with open(args.out,'w',encoding='utf-8') as f: json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result['lags'].items()},indent=2))
if __name__=='__main__': main()
