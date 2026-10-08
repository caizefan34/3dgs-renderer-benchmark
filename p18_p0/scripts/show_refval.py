import json
d = json.load(open(r'C:\Users\36570\3dgs-renderer-benchmark\p18_p0\artifacts\P18_REFERENCE_VALIDATION.json'))
print('=== SYNTHETIC ===')
for v, dom in d['synthetic_conics']['variants'].items():
    for k in ('exp2', 'exp'):
        r = dom[k]
        print(f"{v}/{k}: max_abs={r['max_abs_G_error']:.3e} max_rel={r['max_rel_G_error']} mean={r['mean_abs_G_error']:.3e} branch_mm={r['alpha_branch_mismatch']} nan={r['nan_inf_count']}")
print('=== REAL ===')
for sc, cd in d['real_conics'].items():
    if sc == 'error':
        print('ERROR:', cd); continue
    print(sc, 'n_vis', cd['n_visible'], 'n_samp', cd['n_sampled'])
    for v in cd['per_variant']:
        for k in ('exp2', 'exp'):
            r = cd['per_variant'][v][k]
            print(f"  {v}/{k}: max_abs={r['max_abs_G_error']:.3e} max_rel={r['max_rel_G_error']} mean={r['mean_abs_G_error']:.3e} branch_mm={r['alpha_branch_mismatch']} nan={r['nan_inf_count']}")
