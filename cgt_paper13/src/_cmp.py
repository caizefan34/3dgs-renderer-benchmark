import numpy as np, json
z=np.load('/home/liaoyuanjun/cgt_f0_fused_cuda/results/workload.npz')
sc=z['scene'].astype(str); M=z['M'].astype(int); safe=z['safe'].astype(int)
print('F0 workload: n_tiles',len(sc),'views',sorted(set(z['view'].tolist())),'R',int(z['R']),'Mt',int(z['Mt']))
for s in ['room','garden','bicycle']:
    m=M[sc==s]; sf=safe[sc==s]
    print(f'  F0 {s}: n={len(m)} M_med={int(np.median(m))} M_mean={m.mean():.1f} M_max={m.max()} safe={int(sf.sum())}/{len(m)}')
print()
d=json.load(open('/mnt/storage_pool/liaoyuanjun/cgt_paper13/raw/smoke/cgt_room.json'))
print(f"my room: n_tiles={d['n_tiles']} M_median={d['M_median']} M_mean={round(d['M_mean'],1)} safe_frac={d['certificate']['safe_fraction']} S_mixed={round(d['S_mixed'],4)} T_teacher={round(d['T_teacher_all_ms'],4)}ms T_fused={round(d['T_fused_mixed_ms'],4)}ms T_student={round(d['T_student_all_ms'],4)}ms")