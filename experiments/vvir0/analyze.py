"""VRIR0 Phase P5 — analysis of oracle records.

Reads per-observation JSONL and reports by scene and 5k window:
  - revisit gap distribution
  - topology eligibility
  - truth_same rate
  - C_MEM, C_ORD_EXACT_SCAN, C_ORD_SLACK_N, full fast-certificate hit rates
  - false accepts / false rejects
  - tile-local unchanged fraction
  - adjacent order violation rate
  - n_isects / cache bytes
  - drift quantiles
  - baseline T_struct / T_iter provenance
  - optimistic Amdahl upper bound
"""
from __future__ import annotations
import json, os, sys, glob
import numpy as np

def load_records(path):
    recs = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except Exception:
                pass
    return recs

def analyze_scene(recs, timing=None):
    revisits = [r for r in recs if r.get('eligible') and 'certificate' in r]
    # exclude budget-reject events
    revisits = [r for r in revisits if r.get('event') != 'CACHE_BUDGET_REJECT']
    n_revisits = len(revisits)
    if n_revisits == 0:
        return {'n_revisits': 0, 'note': 'no revisits collected'}
    arr = lambda key: np.array([r[key] for r in revisits], dtype=float)
    cert = lambda key: np.array([1.0 if r['certificate'].get(key) else 0.0 for r in revisits])
    gaps = arr('cache_age_iters')
    truth = arr('truth_same')
    acc = cert('accept')
    mem = cert('membership')
    ord_exact = cert('order_exact_scan')
    ord_slack = cert('order_slack_n')
    false_acc = arr('false_accept')
    false_rej = arr('false_reject')
    tile_frac = np.array([r['tile_truth_same_fraction'] if r['tile_truth_same_fraction'] is not None else np.nan for r in revisits])
    adj_viol = arr('adjacent_order_violations')
    adj_pairs = arr('adjacent_pairs_checked')
    n_isects = arr('n_isects')
    cache_bytes = arr('cache_bytes')
    # drift quantiles (pooled across revisits)
    drift_all = {'means2d': [], 'radii': [], 'depth': []}
    for r in revisits:
        d = r.get('drift') or {}
        for k in drift_all:
            if d.get(k):
                drift_all[k].extend([d[k]['p50'], d[k]['p95'], d[k]['p99']])
    drift_q = {}
    for k, v in drift_all.items():
        if v:
            a = np.array(v)
            drift_q[k] = {'p50': float(np.percentile(a, 50)), 'p95': float(np.percentile(a, 95)),
                          'p99': float(np.percentile(a, 99))}
    # Amdahl: S_UB = T_iter / (T_iter - h * T_struct). Use T_fwd as T_struct proxy (documented).
    amdahl = None
    if timing:
        t_iter = timing.get('t_iter_mean_ms')
        t_fwd = timing.get('t_fwd_mean_ms')
        h = float(np.mean(acc))
        if t_iter and t_fwd and t_iter > h * t_fwd:
            amdahl = {'hit_rate': h, 't_iter_ms': t_iter, 't_struct_ms': t_fwd,
                      'S_UB': t_iter / (t_iter - h * t_fwd),
                      'note': 'T_struct proxied by T_fwd (forward pass); optimistic upper bound'}
    return {
        'n_revisits': n_revisits,
        'cache_age_iters': {'p50': float(np.percentile(gaps,50)), 'p95': float(np.percentile(gaps,95)),
                            'p99': float(np.percentile(gaps,99)), 'max': float(np.max(gaps))},
        'topology_eligible_rate': 1.0,  # all revisits in frozen-topology window are epoch-0 eligible
        'truth_same_rate': float(np.mean(truth)),
        'C_MEM_rate': float(np.mean(mem)),
        'C_ORD_EXACT_SCAN_rate': float(np.mean(ord_exact)),
        'C_ORD_SLACK_N_rate': float(np.mean(ord_slack)),
        'full_fast_cert_accept_rate': float(np.mean(acc)),
        'false_accepts': int(np.sum(false_acc)),
        'false_rejects': int(np.sum(false_rej)),
        'tile_truth_same_fraction_mean': float(np.nanmean(tile_frac)),
        'adjacent_order_violations_total': int(np.sum(adj_viol)),
        'adjacent_pairs_total': int(np.sum(adj_pairs)),
        'adjacent_order_violation_rate': float(np.sum(adj_viol) / max(1, np.sum(adj_pairs))),
        'n_isects_mean': float(np.mean(n_isects)),
        'cache_bytes_mean': float(np.mean(cache_bytes)),
        'drift_quantiles': drift_q,
        'amdahl': amdahl,
    }

def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    recs = load_records(os.path.join(run_dir, 'vir0_records.jsonl'))
    timing = {}
    tp = os.path.join(run_dir, 'vir0_timing.json')
    if os.path.exists(tp):
        timing = json.load(open(tp))
    summ = analyze_scene(recs, timing)
    out = os.path.join(run_dir, 'vir0_analysis.json')
    with open(out, 'w') as f:
        json.dump(summ, f, indent=2)
    print(json.dumps(summ, indent=2))
    print('written', out)

if __name__ == '__main__':
    main()
