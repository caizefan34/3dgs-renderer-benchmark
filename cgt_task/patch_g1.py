import re
p='src/g1_compressibility.py'
s=open(p,encoding='utf-8').read()

# 1) add abs jac + teacher norm to evaluate_tile return
old_ret="""    jt = np.empty((len(te),4,2))
    for d in range(2):
        dq=np.zeros(2); dq[d]=h
        jt[:,:,d]=(predict_and_vjp(p,qte+dq)-predict_and_vjp(p,qte-dq))/(2*h)
    dyt = teacher_jac(px[te], py[te], gids, means2d, conics, opacities, colors)
    jac_rel = float(np.linalg.norm(jt-dyt)/max(np.linalg.norm(dyt),1e-30))
    return dict(psnr=psnr, max_abs=max_abs, jac_rel=jac_rel, mse=mse, n_gauss=int(len(gids)))"""
new_ret="""    jt = np.empty((len(te),4,2))
    for d in range(2):
        dq=np.zeros(2); dq[d]=h
        jt[:,:,d]=(predict_and_vjp(p,qte+dq)-predict_and_vjp(p,qte-dq))/(2*h)
    dyt = teacher_jac(px[te], py[te], gids, means2d, conics, opacities, colors)
    dyt_norm = float(np.linalg.norm(dyt))
    jac_rel = float(np.linalg.norm(jt-dyt)/max(dyt_norm,1e-30))
    jac_abs = float(np.linalg.norm(jt-dyt)/max(np.sqrt(jt.size),1.0))
    return dict(psnr=psnr, max_abs=max_abs, jac_rel=jac_rel, jac_abs=jac_abs,
                teacher_jac_norm=dyt_norm, mse=mse, n_gauss=int(len(gids)))"""
assert old_ret in s
s=s.replace(old_ret,new_ret)

# 2) stratified random sampling instead of top-N
old_sel="""        n_sel = min(args.max_tiles_per_view, len(tiles))
        sel = np.argsort(-counts)[:n_sel]"""
new_sel="""        n_sel = min(args.max_tiles_per_view, len(tiles))
        # stratified random sampling across work deciles (representative, not top-N)
        order = np.argsort(counts)
        ranks_sel = np.empty(len(tiles), dtype=np.int64)
        ranks_sel[order] = np.arange(len(tiles))
        decile = np.minimum((ranks_sel*10)//max(len(tiles),1), 9)
        sel = []
        for dec in range(10):
            pool = np.where(decile==dec)[0]
            if len(pool)==0: continue
            take = max(1, n_sel//10)
            sel.append(rng.choice(pool, size=min(take,len(pool)), replace=False))
        sel = np.concatenate(sel) if sel else np.array([],dtype=np.int64)"""
assert old_sel in s
s=s.replace(old_sel,new_sel)
open(p,'w',encoding='utf-8').write(s)
print("patched OK")
