import json, numpy as np
d=json.load(open('C:/Users/36570/3dgs-renderer-benchmark/results/phase-c29/a_step_value.json'))
psnrs=[r['psnr'] for r in d['records']]
t_iters=[r['t_iter_ms'] for r in d['records']]

# steps to reach PSNR threshold
for target in [48.0, 49.0, 50.0]:
    t=0.0; steps=0
    for r in d['records']:
        if r['psnr']>=target: break
        t+=r['t_iter_ms']; steps+=1
    print(f'T_target PSNR>{target}: {steps} steps, {t:.1f}ms total')

# No densification period (steps 0-99): mean T_iter = ~4.4ms
early = [r for r in d['records'] if r['step']<100]
late = [r for r in d['records'] if r['step']>=100 and r['step']<500]
print(f'Steps 0-99: mean T_iter={np.mean([r["t_iter_ms"] for r in early]):.3f}ms, last PSNR={early[-1]["psnr"]:.2f}')
print(f'Steps 100-500: mean T_iter={np.mean([r["t_iter_ms"] for r in late]):.3f}ms, last PSNR={late[-1]["psnr"]:.2f}')
print(f'Overall mean T_iter (0-500): {np.mean(t_iters):.3f}ms')

# Check backward stability
bwds = [r['bwd_ms'] for r in d['records']]
print(f'Backward mean={np.mean(bwds):.3f}ms cv={np.std(bwds)/np.mean(bwds):.3f}')
