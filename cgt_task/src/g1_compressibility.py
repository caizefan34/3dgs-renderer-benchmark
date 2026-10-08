#!/usr/bin/env python3
"""G1 real-scene cost-weighted compressibility for CGT-Splat.

Work unit = one gsplat tile (tile x Gaussian group). For each tile we:
  1. take the Gaussians whose 2D support intersects the tile, depth-sorted;
  2. compute the teacher alpha-composite transfer response (E,T) over 16x16;
  3. fit a rank-R CGT Gaussian-transfer mixture (R in {4,8,16}) to (E,T);
  4. measure held-out fidelity (PSNR, max abs color err, ray-Jacobian rel-L2);
  5. weight by the tile's Gaussian-intersection count (actual GPU work).

q_weighted = work in tiles whose rank-R fit passes the fidelity gate
             / total work over sampled tiles.
Preregistered gate decided before fitting: PSNR>=30dB, max_abs<=0.02, jac<=0.05.
"""
from __future__ import annotations
import argparse, json, struct, time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import expit, logit, softmax

TILE = 16
GATE_PSNR_DB = 30.0
GATE_MAX_ABS = 0.02
GATE_JAC_REL = 0.05
RANKS = (4, 8, 16)
POWER_CLIP = -4.5   # ~3 sigma

# ---------------- COLMAP ----------------
def read_cameras_bin(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; out = {}
        for _ in range(n):
            cid = struct.unpack("<i", f.read(4))[0]; mid = struct.unpack("<i", f.read(4))[0]
            w = struct.unpack("<Q", f.read(8))[0]; h = struct.unpack("<Q", f.read(8))[0]
            npar = {0:3,1:4,2:3,3:4,4:4,5:5,6:6,7:8}.get(mid,4)
            p = struct.unpack(f"<{npar}d", f.read(npar*8)); out[cid] = (mid,w,h,p)
    return out

def read_images_bin(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; out = []
        for _ in range(n):
            iid = struct.unpack("<i", f.read(4))[0]; qv = struct.unpack("<4d", f.read(32)); tv = struct.unpack("<3d", f.read(24)); cam = struct.unpack("<i", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00": break
                name += c
            np_ = struct.unpack("<Q", f.read(8))[0]; f.read(np_*24)
            out.append((iid, name.decode(), cam, qv, tv))
    return out

def q2R(q):
    x,y,z,w = q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

# ---------------- CGT 2D mixture ----------------
def expert_values(p):
    t, color = expit(p[:,6:7]), expit(p[:,7:10])
    return np.concatenate([(1-t)*color, t], axis=1)

def predict_and_vjp(p, q, cotangent=None):
    dx, dy = q[:,0:1]-p[None,:,0], q[:,1:2]-p[None,:,1]
    l00, l10, l11 = np.exp(p[:,2]), p[:,3], np.exp(p[:,4])
    z0, z1 = dx*l00+dy*l10, dy*l11
    weights = softmax(p[None,:,5]-0.5*(z0*z0+z1*z1), axis=1)
    expert = expert_values(p)
    pred = weights @ expert
    if cotangent is None:
        return pred
    gl = weights*(cotangent@expert.T - np.sum(cotangent*pred, axis=1, keepdims=True))
    grad = np.zeros_like(p)
    grad[:,0] = np.sum(gl*z0*l00, axis=0)
    grad[:,1] = np.sum(gl*(z0*l10+z1*l11), axis=0)
    grad[:,2] = np.sum(-gl*z0*dx*l00, axis=0)
    grad[:,3] = np.sum(-gl*z0*dy, axis=0)
    grad[:,4] = np.sum(-gl*z1*dy*l11, axis=0)
    grad[:,5] = gl.sum(axis=0)
    gv = weights.T @ cotangent
    t, color = expit(p[:,6:7]), expit(p[:,7:10])
    grad[:,6] = ((gv[:,3:4]-np.sum(gv[:,:3]*color, axis=1, keepdims=True))*t*(1-t))[:,0]
    grad[:,7:10] = gv[:,:3]*(1-t)*color*(1-color)
    return pred, grad

def initialize(k, q, y, seed):
    rng = np.random.default_rng(seed)
    p = np.zeros((k,10))
    p[:,:2] = q[rng.choice(len(q), k, replace=True)]
    p[:,2] = p[:,4] = np.log(np.sqrt(k)/1.4)
    p[:,6] = logit(np.clip(y[:,3].mean(), 0.005, 0.995))
    p[:,7:10] = logit(np.clip(y[:,:3].mean(0), 0.005, 0.995))
    return p

def objective(flat, k, q, y, derivative_data=None, derivative_weight=0.0):
    p = flat.reshape(k,10)
    pred = predict_and_vjp(p, q)
    err = pred - y
    loss = float(np.mean(err**2))
    _, grad = predict_and_vjp(p, q, 2*err/err.size)
    if derivative_data is not None and derivative_weight > 0:
        h, target_d = derivative_data
        for d in range(2):
            dq = np.zeros(2); dq[d] = h
            pp, pm = predict_and_vjp(p, q+dq), predict_and_vjp(p, q-dq)
            ed = (pp-pm)/(2*h) - target_d[:,:,d]
            loss += derivative_weight*float(np.mean(ed**2))/2
            gp = derivative_weight*ed/(ed.size*2*h)
            _, ga = predict_and_vjp(p, q+dq, gp)
            _, gb = predict_and_vjp(p, q-dq, -gp)
            grad += ga+gb
    return loss, grad.ravel()

BOUNDS_ONE = [(-3,3),(-3,3),(-3,4),(-15,15),(-3,4),(-12,12),(-12,12),(-12,12),(-12,12),(-12,12)]

def fit_cgt(q, y, rank, seed, iters, dy=None, dweight=0.02):
    p0 = initialize(rank, q, y, seed)
    res = minimize(objective, p0.ravel(), args=(rank, q, y, (1e-3, dy) if dy is not None else None, dweight),
                   method="L-BFGS-B", jac=True, bounds=BOUNDS_ONE*rank,
                   options={"maxiter":iters, "ftol":1e-13, "gtol":1e-8, "maxls":30})
    return res.x.reshape(rank,10)

# ---------------- teacher for one tile ----------------
def tile_teacher(px, py, gids, means2d, conics, opacities, colors):
    """px,py: (P,1). Front-to-back alpha composite over tile's Gaussians.
    Returns E (P,3), T (P,1)."""
    mu = means2d[gids]; co = conics[gids]; op = opacities[gids]; col = colors[gids]
    dx = px[:,None,:] - mu[None,:,:]          # (P,m,2)
    a = co[:,0]; b = co[:,1]; c = co[:,2]
    Ax = dx[:,:,0]; Ay = dx[:,:,1]
    power = -0.5*(a[None,:]*Ax*Ax + 2*b[None,:]*Ax*Ay + c[None,:]*Ay*Ay)
    alpha = np.where(power > POWER_CLIP, op[None,:]*np.exp(np.clip(power,None,0)), 0.0)
    alpha = np.clip(alpha, 0.0, 0.999)
    one_minus = 1.0 - alpha
    Tprefix = np.cumprod(one_minus, axis=1)                 # (P,m)
    Tbefore = np.concatenate([np.ones((alpha.shape[0],1)), Tprefix[:,:-1]], axis=1)
    E = np.einsum('pi,pi,ic->pc', Tbefore, alpha, col)
    T = Tprefix[:,-1:]
    return E.astype(np.float64), T.astype(np.float64)

def teacher_jac(px, py, gids, means2d, conics, opacities, colors, h=1e-3):
    P = px.shape[0]; J = np.empty((P,4,2))
    for d in range(2):
        dq = np.zeros(2); dq[d] = h
        Ep,Tp = tile_teacher(px+dq[0], py+dq[1], gids, means2d, conics, opacities, colors)
        Em,Tm = tile_teacher(px-dq[0], py-dq[1], gids, means2d, conics, opacities, colors)
        J[:,:,d] = (np.concatenate([Ep,Tp],axis=1)-np.concatenate([Em,Tm],axis=1))/(2*h)
    return J

def psnr_from_mse(mse):
    return float(-10*np.log10(max(mse,1e-30)))

def evaluate_tile(px, py, gids, means2d, conics, opacities, colors, rank, seed, iters, dweight=0.02, h=1e-3):
    E, T = tile_teacher(px, py, gids, means2d, conics, opacities, colors)
    y = np.concatenate([E, T], axis=1)                      # (P,4)
    P = y.shape[0]; tr = np.arange(0,P,2); te = np.arange(1,P,2)
    qt = np.column_stack([px[tr,0], py[tr,0]]); yt = y[tr]
    dy = teacher_jac(px[tr], py[tr], gids, means2d, conics, opacities, colors)
    p = fit_cgt(qt, yt, rank, seed, iters, dy=dy, dweight=dweight)
    qte = np.column_stack([px[te,0], py[te,0]])
    pred_tr = predict_and_vjp(p, qt); pred_te = predict_and_vjp(p, qte); y_te = y[te]
    mse = float(np.mean((pred_te[:,:3]-y_te[:,:3])**2))
    bgs = np.array([[0,0,0],[1,1,1],[0.2,0.6,0.9]])
    exp = y_te[:,:3][:,None,:] + y_te[:,3:4][:,:,None]*bgs[None,:,:]
    pr  = pred_te[:,:3][:,None,:] + pred_te[:,3:4][:,:,None]*bgs[None,:,:]
    max_abs = float(np.max(np.abs(pr-exp)))
    psnr = psnr_from_mse(mse)
    jt = np.empty((len(te),4,2))
    for d in range(2):
        dq=np.zeros(2); dq[d]=h
        jt[:,:,d]=(predict_and_vjp(p,qte+dq)-predict_and_vjp(p,qte-dq))/(2*h)
    dyt = teacher_jac(px[te], py[te], gids, means2d, conics, opacities, colors)
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
                mse=mse, n_gauss=int(len(gids)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', required=True); ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data', required=True); ap.add_argument('--views', type=int, default=8)
    ap.add_argument('--iterations', type=int, default=300); ap.add_argument('--max-tiles-per-view', type=int, default=400)
    ap.add_argument('--out', required=True); ap.add_argument('--seed', type=int, default=20261004)
    args = ap.parse_args()
    dev = 'cuda'
    cams = read_cameras_bin(f'{args.data}/datasets/mipnerf360/{args.scene}/sparse/0/cameras.bin')
    imgs = read_images_bin(f'{args.data}/datasets/mipnerf360/{args.scene}/sparse/0/images.bin')
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    means = ck['xyz'].to(dev).float(); scales = torch.exp(ck['scaling'].to(dev).float())
    quats = ck['rotation'].to(dev).float(); opacities = torch.sigmoid(ck['opacity'].to(dev).float()); shs = ck['shs'].to(dev).float()
    deg = int(ck['active_sh_degree'])
    rng = np.random.default_rng(args.seed)
    view_idx = rng.choice(len(imgs), size=min(args.views, len(imgs)), replace=False)
    records = []
    for vi in view_idx:
        iid, name, cam, qv, tv = imgs[vi]; mid, W, H, par = cams[cam]; fx, fy, cx, cy = par
        R_w2c = q2R(qv); t_w2c = np.array(tv); t_c2w = -R_w2c.T @ t_w2c
        viewmat = torch.eye(4, dtype=torch.float32, device=dev)[None]
        viewmat[0,:3,:3] = torch.as_tensor(R_w2c, dtype=torch.float32, device=dev)
        viewmat[0,:3,3] = torch.as_tensor(t_w2c, dtype=torch.float32, device=dev)
        K = torch.as_tensor([[[fx,0,cx],[0,fy,cy],[0,0,1]]], dtype=torch.float32, device=dev)
        campos = torch.as_tensor(t_c2w, dtype=torch.float32, device=dev)[None]
        t0 = time.time()
        rc, ra, meta = gsplat.rasterization(means, quats, scales, opacities, shs, viewmat, K, width=W, height=H, sh_degree=deg, render_mode='RGB', packed=True, tile_size=TILE)
        dirs = torch.nn.functional.normalize(means - campos, dim=-1)
        colors = (gsplat.spherical_harmonics(deg, dirs, shs) + 0.5).clamp(0,1).detach().cpu().numpy()[meta['gaussian_ids'].detach().cpu().numpy()]  # remap original->packed index
        def flat2(x):
            x = meta[x].detach().cpu().numpy();  return x[0] if x.ndim==3 else x
        means2d = flat2('means2d'); conics = flat2('conics'); opac = flat2('opacities'); depths = flat2('depths')
        io = meta['isect_offsets'].detach().cpu().numpy().reshape(-1)
        fi = meta['flatten_ids'].detach().cpu().numpy()
        th, tw = int(meta['tile_height']), int(meta['tile_width']); total = len(fi)
        starts = io.astype(np.int64)
        def start_of(ti,tj): return int(starts[ti*tw+tj])
        def end_of(ti,tj):
            if tj+1 < tw: return int(starts[ti*tw+tj+1])
            if ti+1 < th: return int(starts[(ti+1)*tw+0])
            return total
        tiles = [(ti,tj) for ti in range(th) for tj in range(tw) if end_of(ti,tj)-start_of(ti,tj) > 0]
        counts = np.array([end_of(ti,tj)-start_of(ti,tj) for ti,tj in tiles])
        if len(tiles)==0: continue
        n_sel = min(args.max_tiles_per_view, len(tiles))
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
        sel = np.concatenate(sel) if sel else np.array([],dtype=np.int64)
        for idx in sel:
            ti,tj = tiles[idx]; s=start_of(ti,tj); e=end_of(ti,tj); gids=fi[s:e]
            if len(gids)<1: continue
            ox=tj*TILE; oy=ti*TILE
            xs=ox+np.arange(TILE)+0.5; ys=oy+np.arange(TILE)+0.5
            gx,gy=np.meshgrid(xs,ys,indexing='xy')
            px=np.stack([gx.ravel(),gy.ravel()],axis=1).astype(np.float64)
            cx0,cy0=px.mean(0); px[:,0]-=cx0; px[:,1]-=cy0
            for rank in RANKS:
                seed=args.seed+ti*1000+tj*7+rank
                try:
                    r=evaluate_tile(px[:,0:1],px[:,1:2],gids,means2d,conics,opac,colors,rank,seed,args.iterations)
                except Exception as ex:
                    import traceback as _tb; r=dict(psnr=0.0,max_abs=9.0,jac_rel=9.0,mse=9.0,n_gauss=int(len(gids)),err=_tb.format_exc()[-400:])
                r.update(scene=args.scene,view=name,tile_ti=int(ti),tile_tj=int(tj),rank=int(rank),tile_work=int(e-s))
                records.append(r)
        print(f"[{args.scene}] view {name}: tiles={len(tiles)} sel={n_sel} inter={total} t={time.time()-t0:.1f}s",flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(scene=args.scene,ckpt=args.ckpt,n_records=len(records),
                   gate=dict(psnr=GATE_PSNR_DB,max_abs=GATE_MAX_ABS,jac_rel=GATE_JAC_REL),records=records),
              open(args.out,'w'),indent=1)
    print("WROTE",args.out,"records",len(records),flush=True)

if __name__ == '__main__':
    import gsplat
    main()





