import json
d=json.load(open('h0_run/_smoke_room25k_native.json'))
r=d['rollouts'][0]
print('arm',r['arm'],'K',r['K'],'fold',r['fold'],'offset',r['offset'],'seq',r['update_seq'])
print('camera_ids',d['camera_ids_c0_c7'])
for t in r['trajectory']:
    nv=t['native']
    print(f"  step{t['step']}: hold_obj={nv['holdout_obj']:.6f} design_obj={nv['design_obj']:.6f} hold_psnr={nv['holdout_psnr']:.3f} hold_ssim={nv['holdout_ssim']:.4f} upd_g={t['update_norm']['global']:.3e}")
