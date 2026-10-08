import json,numpy as np
d=json.load(open('/home/liaoyuanjun/cgt_g5_native/results/G5A_NATIVE_CAPACITY.json'))
print('VERDICT',d['verdict'],'gate',d['native_variants_gate_pass_all_scenes'])
print('sanity frozen chain:',d['sanity_frozen_chain'])
for sc,r in d['per_scene'].items():
    A=r['variants']['A_frozen_teacher']; B=r['variants']['B_native_from_teacher']; C=r['variants']['C_native_from_scratch']
    print('==',sc,'n_tiles',r['n_tiles'])
    print('   A psnr med=%.3f min=%.3f mean=%.3f ssim=%.4f l1=%.5f'%(A['psnr_median'],A['psnr_min'],A['psnr_mean'],A['ssim_mean'],A['l1_mean']))
    print('   B psnr med=%.3f min=%.3f mean=%.3f ssim=%.4f train=%.3f dPSNR=%+.3f dSSIM=%+.5f pass=%s'%(B['psnr_median'],B['psnr_min'],B['psnr_mean'],B['ssim_mean'],B['psnr_train_mean'],B['dPSNR_vs_A_median'],B['dSSIM_vs_A_mean'],B['gate_pass']))
    print('   C psnr med=%.3f min=%.3f mean=%.3f ssim=%.4f train=%.3f dPSNR=%+.3f dSSIM=%+.5f pass=%s'%(C['psnr_median'],C['psnr_min'],C['psnr_mean'],C['ssim_mean'],C['psnr_train_mean'],C['dPSNR_vs_A_median'],C['dSSIM_vs_A_mean'],C['gate_pass']))
    print('   B train-heldout gap (mean): %.4f dB ; C: %.4f dB'%(B['psnr_train_mean']-B['psnr_mean'],C['psnr_train_mean']-C['psnr_mean']))
