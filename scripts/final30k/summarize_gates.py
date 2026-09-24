import json
import sys

for scene in ("room", "bicycle", "garden"):
    try:
        d = json.load(open(f'/mnt/storage_pool/liaoyuanjun/final30k_gates/gate_results_{scene}.json'))
    except FileNotFoundError:
        print(f'=== {scene}: MISSING ===')
        continue
    print(f'================ {scene.upper()} ================')
    gc = d['gate_c']
    print('GATE C:', gc['classification'])
    if gc['fail_reasons']:
        print('fail_reasons:', gc['fail_reasons'])
    print('discrete:', json.dumps(gc['discrete']))
    print('absgrad_presence:', json.dumps(gc['absgrad_presence']))
    print('new_abs0_vs_new_abs1 max_abs:', {k: f"{v['max_abs']:.3e}" for k, v in gc['new_abs0_vs_new_abs1'].items()})
    print('old_vs_new_abs1 rel_l2:', {k: f"{v['relative_L2']:.2e}" for k, v in gc['old_vs_new_abs1'].items()})
    print('envelope(old,old) max_abs:', {k: f"{v['max_abs']:.3e}" for k, v in gc['envelope_old_old'].items()})
    d2 = d['gate_d2']
    print('GATE D2:', d2['classification'])
    m = d2['absgrad_metrics']
    print('  absgrad: max_abs=%.3e mean_abs=%.3e rel_L2=%.3e cos=%.8f outside_tol=%d support_mismatch=%d NaN=%d' % (
        m['max_abs'], m['mean_abs'], m['relative_L2'], m['cosine'], m['outside_tolerance_count'],
        m['support_mismatch'], m['NaN_count']))
    print('  cos_x=%.8f cos_y=%.8f' % (d2['absgrad_metrics_x']['cosine'], d2['absgrad_metrics_y']['cosine']))
    sc = d2.get('signed_contracted_metrics')
    if sc:
        print('  SIGNED-CONTRACTED (renderer numerics envelope): cos=%.8f rel_L2=%.3e max_abs=%.3e' % (
            sc['cosine'], sc['relative_L2'], sc['max_abs']))
    rd = d2.get('absgrad_row_rel_dev')
    if rd:
        print('  absgrad row rel dev: mean=%.3e median=%.3e p90=%.3e p99=%.3e max=%.3e' % (
            rd.get('mean', -1), rd.get('median', -1), rd.get('p90', -1), rd.get('p99', -1), rd.get('max', -1)))
    cf = d2.get('abs_stage_counterfactual')
    if cf:
        print('  abs-stage counterfactual: c0_abs vs |accum-signed| cos=%.4f ; ref_abs vs |accum-signed| cos=%.4f' % (
            cf['c0_absgrad_vs_abs_after_accum']['cosine'], cf['ref_absgrad_vs_abs_after_accum']['cosine']))
    print('  c0_dist:', json.dumps(d2['c0_absgrad_distribution']))
    print('  ref_dist:', json.dumps(d2['ref_absgrad_distribution']))
    print('  n_vis=%d n_isects=%d' % (d2['n_vis'], d2['n_isects']))
    print('  signed_m2d (moment vs contracted, expected to differ): cos=%.8f rel_l2=%.2e' % (d2['signed_m2d_metrics']['cosine'], d2['signed_m2d_metrics']['relative_L2']))
    d1 = d['gate_d1']
    print('GATE D1 (end-to-end):')
    m = d1['absgrad_metrics_full']
    print('  absgrad: max_abs=%.3e mean_abs=%.3e rel_L2=%.3e cos=%.6f support_mismatch=%d NaN=%d' % (
        m['max_abs'], m['mean_abs'], m['relative_L2'], m['cosine'], m['support_mismatch'], m['NaN_count']))
    if 'signed_gmeans_metrics' in d1:
        sg = d1['signed_gmeans_metrics']
        print('  SIGNED g_means envelope: cos=%.8f rel_L2=%.3e max_abs=%.3e' % (
            sg['cosine'], sg['relative_L2'], sg['max_abs']))
    print('  c0_dist:', json.dumps(d1['c0_absgrad_distribution']))
    print('  b1a_dist:', json.dumps(d1['b1a_absgrad_distribution']))
    print('  n_visible:', json.dumps(d1['n_visible']))
    print()
