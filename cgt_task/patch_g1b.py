import re
p='src/g1_compressibility.py'
s=open(p,encoding='utf-8').read()

# add response_range + well-posed jac to evaluate_tile
old="""    dyt = teacher_jac(px[te], py[te], gids, means2d, conics, opacities, colors)
    dyt_norm = float(np.linalg.norm(dyt))
    jac_rel = float(np.linalg.norm(jt-dyt)/max(dyt_norm,1e-30))
    jac_abs = float(np.linalg.norm(jt-dyt)/max(np.sqrt(jt.size),1.0))
    return dict(psnr=psnr, max_abs=max_abs, jac_rel=jac_rel, jac_abs=jac_abs,
                teacher_jac_norm=dyt_norm, mse=mse, n_gauss=int(len(gids)))"""
new="""    dyt = teacher_jac(px[te], py[te], gids, means2d, conics, opacities, colors)
    dyt_norm = float(np.linalg.norm(dyt))
    jac_frob = float(np.linalg.norm(jt-dyt))
    jac_rel = jac_frob/max(dyt_norm,1e-30)
    jac_abs = jac_frob/max(np.sqrt(jt.size),1.0)
    # response range: how much (E,T) actually varies across the tile
    resp_range = float(np.max(y_te.max(0)-y_te.min(0)))
    # well-posed jac: normalize by response gradient scale (not raw teacher norm)
    grad_scale = resp_range/max(px[te].max()-px[te].min(),1.0)
    jac_wp = jac_frob/max(dyt_norm, grad_scale*0.1, 1e-12)
    return dict(psnr=psnr, max_abs=max_abs, jac_rel=jac_rel, jac_abs=jac_abs,
                jac_wp=jac_wp, teacher_jac_norm=dyt_norm, resp_range=resp_range,
                mse=mse, n_gauss=int(len(gids)))"""
assert old in s
s=s.replace(old,new)
open(p,'w',encoding='utf-8').write(s)
print("patched")
