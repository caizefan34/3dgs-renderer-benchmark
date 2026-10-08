#!/usr/bin/env python3
import argparse, json, math, os
from pathlib import Path
from statistics import median

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import numpy as np
except Exception:
    np = None


def load_rows(path):
    p = Path(path)
    if p.suffix == '.json':
        x = json.loads(p.read_text())
        return x['rows'] if isinstance(x, dict) and 'rows' in x else x
    if p.suffix == '.csv':
        import csv
        with p.open(newline='') as f:
            return list(csv.DictReader(f))
    if p.suffix == '.parquet':
        if pd is None:
            raise RuntimeError('pandas/parquet engine unavailable; export JSON or CSV')
        return pd.read_parquet(p).to_dict('records')
    raise ValueError(f'unsupported results format: {p}')


def fnum(x, default=float('nan')):
    try:
        return float(x)
    except Exception:
        return default


def canonical_action(a):
    a = str(a).strip().lower().replace('_','-').replace('lr','')
    aliases = {
        'a0':'A0',
        'a1-1.00':'A1-1.00','a1-1.0':'A1-1.00','a1-1':'A1-1.00',
        'a1-0.75':'A1-0.75','a1-.75':'A1-0.75',
        'a1-0.50':'A1-0.50','a1-0.5':'A1-0.50','a1-.5':'A1-0.50',
        'a1-0.25':'A1-0.25','a1-.25':'A1-0.25',
        'a2-r1':'A2-r1','a2-r2':'A2-r2','a2-r3':'A2-r3'
    }
    return aliases.get(a, str(a))


def group_anchors(rows):
    out = {}
    for r in rows:
        scene = str(r['scene'])
        ckpt = int(r['checkpoint'])
        aid = int(r['anchor_id'])
        k = (scene, ckpt, aid)
        a = canonical_action(r['action'])
        out.setdefault(k, {}).setdefault(a, []).append(r)
    return out


def reduce_anchor(actions, noise_floor):
    req = ['A0','A1-1.00','A1-0.75','A1-0.50','A1-0.25','A2-r1','A2-r2','A2-r3']
    missing = [x for x in req if x not in actions]
    if missing:
        return None, missing
    def one(a):
        rr = actions[a]
        # one row expected; if duplicates, use median progress and median cost
        return {
            'delta': median([fnum(x.get('delta_fused', x.get('delta_loss'))) for x in rr]),
            'cost': median([fnum(x.get('wall_ms', x.get('total_ms'))) for x in rr])
        }
    vals = {a:one(a) for a in req}
    a2d = [vals[x]['delta'] for x in ['A2-r1','A2-r2','A2-r3']]
    a2c = [vals[x]['cost'] for x in ['A2-r1','A2-r2','A2-r3']]
    vals['A2-median'] = {'delta':median(a2d),'cost':median(a2c)}
    dacts = ['A0','A1-1.00','A1-0.75','A1-0.50','A1-0.25']
    best = max(dacts, key=lambda a: vals[a]['delta'])
    vals['D-best'] = {'action':best, **vals[best]}
    vals['unique_a2'] = (vals['A2-median']['delta'] - vals['D-best']['delta']) > noise_floor
    vals['a2_partner_positive_count'] = sum(d > vals['D-best']['delta'] + noise_floor for d in a2d)
    return vals, []


def pareto_options(anchor_vals, action_names):
    opts=[]
    for a in action_names:
        v=anchor_vals[a]
        opts.append((a, max(0.0, float(v['cost'])), float(v['delta'])))
    # Always allow A0; keep negative progress because optimal solver can choose A0 instead.
    return opts


def solve_mckp(anchors, action_names, rho, baseline_cost):
    # Exact multiple-choice knapsack via scipy MILP when available.
    from scipy.optimize import milp, Bounds, LinearConstraint
    from scipy.sparse import lil_matrix
    n=len(anchors); m=len(action_names)
    c=[]; costs=[]
    for av in anchors:
        for a in action_names:
            c.append(-float(av[a]['delta']))
            costs.append(float(av[a]['cost']))
    N=n*m
    Aeq=lil_matrix((n,N), dtype=float)
    for i in range(n):
        for j in range(m): Aeq[i,i*m+j]=1.0
    constraints=[LinearConstraint(Aeq.tocsr(), lb=[1.0]*n, ub=[1.0]*n),
                 LinearConstraint([costs], lb=-math.inf, ub=rho*baseline_cost)]
    res=milp(c=c, integrality=[1]*N, bounds=Bounds([0]*N,[1]*N), constraints=constraints,
             options={'time_limit':120})
    if not res.success:
        raise RuntimeError(f'MILP failed: {res.message}')
    chosen=[]; prog=0.0; cost=0.0
    for i,av in enumerate(anchors):
        xs=res.x[i*m:(i+1)*m]
        j=max(range(m), key=lambda q: xs[q])
        a=action_names[j]
        chosen.append(a); prog += av[a]['delta']; cost += av[a]['cost']
    return {'progress':prog,'cost':cost,'chosen':chosen,'rho':rho,'budget':rho*baseline_cost}


def fixed_mixture_frontier(anchors, action_names, rho, baseline_cost):
    # Global mixture upper bound without per-anchor outcome-dependent assignment.
    # Enumerates integer counts of action types and assigns action labels independent of anchor outcomes
    # by using their scene-wide mean progress/cost; this deliberately prevents per-anchor oracle selection.
    n=len(anchors)
    means={}
    for a in action_names:
        means[a]={'delta':sum(x[a]['delta'] for x in anchors)/n,
                  'cost':sum(x[a]['cost'] for x in anchors)/n}
    best=None
    if len(action_names)==2:
        a0,a1=action_names
        for n1 in range(n+1):
            n0=n-n1
            cost=n0*means[a0]['cost']+n1*means[a1]['cost']
            if cost <= rho*baseline_cost+1e-9:
                prog=n0*means[a0]['delta']+n1*means[a1]['delta']
                if best is None or prog>best['progress']:
                    best={'progress':prog,'cost':cost,'counts':{a0:n0,a1:n1}}
    return best


def analyze_scene(scene, anchor_map, noise, rhos=(0.60,0.75,0.90)):
    keys=sorted(k for k in anchor_map if k[0]==scene)
    vals=[]; miss=[]
    for k in keys:
        av,mm=reduce_anchor(anchor_map[k], noise)
        if mm: miss.append({'key':k,'missing':mm})
        else: vals.append((k,av))
    if miss:
        return {'scene':scene,'valid':False,'missing':miss}
    anchors=[v for _,v in vals]
    baseline_cost=sum(v['A1-1.00']['cost'] for v in anchors)
    all_a1_progress=sum(v['A1-1.00']['delta'] for v in anchors)
    out={'scene':scene,'valid':True,'n_anchors':len(anchors),'baseline_cost':baseline_cost,
         'all_a1_progress':all_a1_progress}
    out['unique_a2_fraction']=sum(v['unique_a2'] for v in anchors)/len(anchors)
    out['unique_a2_fraction_by_checkpoint']={}
    for ckpt in sorted(set(k[1] for k,_ in vals)):
        sub=[v for k,v in vals if k[1]==ckpt]
        out['unique_a2_fraction_by_checkpoint'][str(ckpt)]=sum(v['unique_a2'] for v in sub)/len(sub)
    fronts={}
    for rho in rhos:
        damp=solve_mckp(anchors,['A0','A1-1.00','A1-0.75','A1-0.50','A1-0.25'],rho,baseline_cost)
        a2=solve_mckp(anchors,['A0','A1-1.00','A2-median'],rho,baseline_cost)
        gain=(a2['progress']-damp['progress'])/(abs(damp['progress'])+1e-12)
        fronts[str(rho)]={'damping_only':{k:v for k,v in damp.items() if k!='chosen'},
                          'a2_frontier':{k:v for k,v in a2.items() if k!='chosen'},
                          'relative_progress_gain_over_damping':gain,
                          'a2_count':sum(x=='A2-median' for x in a2['chosen'])}
    out['frontiers']=fronts
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    ap.add_argument('--noise', required=False, help='JSON: {"room": value, "bicycle": value}; defaults 0')
    ap.add_argument('--outdir', required=True)
    args=ap.parse_args()
    rows=load_rows(args.results)
    amap=group_anchors(rows)
    noise={'room':0.0,'bicycle':0.0}
    if args.noise:
        noise.update(json.loads(Path(args.noise).read_text()))
    summaries=[analyze_scene(s,amap,float(noise.get(s,0.0))) for s in ('room','bicycle')]
    by={x['scene']:x for x in summaries}
    if not all(by[s].get('valid') for s in ('room','bicycle')):
        verdict='N1D_INVALID_OR_INCOMPLETE'; reason='missing_actions_or_invalid_rows'
    else:
        gains={s:by[s]['frontiers']['0.75']['relative_progress_gain_over_damping'] for s in by}
        uniq={s:by[s]['unique_a2_fraction'] for s in by}
        ckspread={s:sum(v>=0.10 for v in by[s]['unique_a2_fraction_by_checkpoint'].values())>=2 for s in by}
        if gains['room']>=0.05 and gains['bicycle']>=0.05 and uniq['room']>=0.10 and uniq['bicycle']>=0.10 and ckspread['room'] and ckspread['bicycle']:
            verdict='N1D_SECOND_VIEW_VALUE_CONFIRMED'; reason='both_scenes_clear_damping_and_anchor_gates'
        elif gains['room']<0.03 and gains['bicycle']<0.03:
            verdict='N1D_A2_EXPLAINED_BY_DAMPING_OR_NOISE'; reason='both_scenes_below_3pct_matched_cost_gain'
        elif (0.03 <= gains['room'] < 0.05) or (0.03 <= gains['bicycle'] < 0.05) or ((gains['room']>=0.05) != (gains['bicycle']>=0.05)):
            verdict='N1D_WEAK_SECOND_VIEW_VALUE'; reason='borderline_or_one_scene_only'
        else:
            verdict='N1D_WEAK_SECOND_VIEW_VALUE'; reason='strong_gain_but_anchor_or_phase_gate_failed'
    outdir=Path(args.outdir); outdir.mkdir(parents=True,exist_ok=True)
    (outdir/'N1D_ANALYSIS.json').write_text(json.dumps({'scenes':summaries},indent=2))
    (outdir/'N1D_VERDICT.json').write_text(json.dumps({'verdict':verdict,'reason':reason},indent=2))
    print(json.dumps({'verdict':verdict,'reason':reason},indent=2))

if __name__=='__main__': main()
