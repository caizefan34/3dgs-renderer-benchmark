import torch, json, sys

p = sys.argv[1] if len(sys.argv) > 1 else '/root/epic05_d0/runs/TRACE_room/d0_ckpt_15000.pt'
ck = torch.load(p, map_location='cpu', weights_only=False)
print('ckpt keys:', sorted(ck.keys()))
osd = ck.get('optimizer_state_dict')
if osd:
    print('opt state top keys:', list(osd.keys()))
    print('n state entries:', len(osd.get('state', {})))
    st = osd['state'][0] if osd.get('state') else {}
    print('per-param state keys:', sorted(st.keys()))
    if 'step' in st:
        print('step:', st['step'])
    for g in osd['param_groups']:
        print('group:', {k: v for k, v in g.items() if k in ('name', 'lr', 'betas', 'eps', 'weight_decay', 'amsgrad')})
    if osd.get('state'):
        s0 = osd['state'][0]
        if 'exp_avg' in s0:
            print('exp_avg shape:', tuple(s0['exp_avg'].shape), 'exp_avg_sq shape:', tuple(s0['exp_avg_sq'].shape))
for k in ('xyz', 'shs', 'scaling', 'rotation', 'opacity'):
    if k in ck:
        print('param', k, tuple(ck[k].shape), ck[k].dtype)
print('iteration:', ck.get('iteration'), 'active_sh_degree:', ck.get('active_sh_degree'))
