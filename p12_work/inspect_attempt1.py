import json
r = json.load(open('/root/epic05_p12/preflight_attempts/attempt1_partial_contaminated_gpus01/pre_gmain_s42_garden/results.json'))
lb = r['loss_backend']
print(json.dumps({k: lb[k] for k in ('name', 'padding', 'so_sha256', 'padding_same_probe', 'formula')}, indent=1))
print('per_iter losses:', [(p['iter'], round(p['loss'], 5)) for p in r['per_iter']])
print('final_eval:', {k: r['final_eval'].get(k) for k in ('psnr', 'ssim', 'lpips')})
print('timing mean_iter_ms:', r['timing']['mean_iter_ms'])
print('phase_ms:', json.dumps(r['timing']['phase_ms']))
print('higs_env_all:', r['higs_env_all'])
print('peak_vram_gb:', json.load(open('/root/epic05_p12/preflight_attempts/attempt1_partial_contaminated_gpus01/pre_gmain_s42_garden/memory.json'))['peak_vram_gb'])
fr = json.load(open('/root/3dgs-publication-epic05/full30k/gmain03_s42_garden/results.json'))
fl = {p['iter']: p['loss'] for p in fr['per_iter']}
print('frozen gmain garden losses at 1/100/200:', [(i, round(fl[i], 5)) for i in (1, 100, 200)])
print('initial_N frozen:', fr['initial_N'])
