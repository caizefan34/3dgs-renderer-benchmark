#!/usr/bin/env python3
"""Aggregate VRIR0 JSONL oracle records. No benchmark claims are produced here."""
import argparse, json, math
from collections import defaultdict


def pct(a,b): return (a/b) if b else float('nan')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('jsonl'); ap.add_argument('--t-struct-ms',type=float); ap.add_argument('--t-iter-ms',type=float)
    args=ap.parse_args()
    rec=[]
    for line in open(args.jsonl,'r',encoding='utf-8'):
        line=line.strip()
        if line: rec.append(json.loads(line))
    if not rec: raise SystemExit('no records')
    groups=defaultdict(list)
    for r in rec:
        it=r['iteration']
        if 15001 <= it <= 20000: w='15k-20k'
        elif 20001 <= it <= 25000: w='20k-25k'
        elif 25001 <= it <= 30000: w='25k-30k'
        else: w='diagnostic'
        groups[(r['scene'],w)].append(r)
    out={}
    for key,rs in sorted(groups.items()):
        n=len(rs); elig=sum(bool(r.get('eligible')) for r in rs)
        ca=sum(bool(r.get('certificate',{}).get('accept')) for r in rs)
        same=sum(bool(r.get('truth_same')) for r in rs)
        fa=sum(bool(r.get('false_accept')) for r in rs)
        fr=sum(bool(r.get('false_reject')) for r in rs)
        h=pct(ca,n)
        row={'n':n,'eligible_rate':pct(elig,n),'truth_same_rate':pct(same,n),'cert_hit_rate':h,'false_accepts':fa,'false_rejects':fr}
        if args.t_struct_ms is not None and args.t_iter_ms is not None and math.isfinite(h):
            denom=args.t_iter_ms-h*args.t_struct_ms
            row['optimistic_speedup_upper_bound']=(args.t_iter_ms/denom) if denom>0 else float('inf')
        out[f'{key[0]}:{key[1]}']=row
    print(json.dumps(out,indent=2,sort_keys=True))

if __name__=='__main__': main()
