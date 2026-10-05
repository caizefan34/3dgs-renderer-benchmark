#!/usr/bin/env python3
"""Debug: is the per-tile AABB reject exact? float64 numpy check vs baseline alpha.

For each (tile,gaussian) pair the kernel would SKIP (AABB misses tile), check whether
any pixel in that tile has baseline alpha >= 1/255 (a wrong skip). Report counts + magnitude.
"""
import sys, os, json, math, argparse
import numpy as np
PACKAGE_SCRIPTS = os.environ["BES0_PACKAGE_ROOT"] + "/scripts"
sys.path.insert(0, PACKAGE_SCRIPTS)
import mx_guard
cfg = mx_guard.assert_mx_host()
import torch
import scene_adapter as sa
TS = 16

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--gpu-index", type=int, default=7)
    ap.add_argument("--max-cams", type=int, default=1)
    ap.add_argument("--max-gauss", type=int, default=4000)
    args = ap.parse_args()
    for x in mx_guard.checked_inventory(cfg):
        if x["index"] == args.gpu_index and x["index"] in cfg["allowed_gpu_indices"]:
            gpu_uuid = x["uuid"]; break
    torch.set_num_threads(1); torch.cuda.set_device(0)
    man = json.load(open(args.manifest)); rec = man["scenes"][args.scene]
    dev_ids = list(rec["dev_camera_ids"])[:args.max_cams]
    scene = sa.load_scene(rec, "cuda")
    op_all = sa.opacity_activation(scene.opacity).reshape(-1).cpu().numpy()
    CUTOFF = 1.0/255.0
    for cid in dev_ids:
        cam = scene.cameras[cid]
        mu2d, conic, depth, radii = sa.project_gaussians_2d(scene, cam)
        mu = mu2d.cpu().numpy().astype(np.float64)
        cc = conic.cpu().numpy().astype(np.float64)
        dd = depth.cpu().numpy().astype(np.float64)
        rr = radii.cpu().numpy().astype(np.float64)
        op = op_all.astype(np.float64)
        W, H = cam.width, cam.height
        tiles_x = (W+TS-1)//TS; tiles_y = (H+TS-1)//TS
        vis = (dd > 0.05) & (rr > 0)
        idx = np.nonzero(vis)[0]
        mu=mu[idx]; cc=cc[idx]; dd=dd[idx]; rr=rr[idx]; op=op[idx]
        V = mu.shape[0]
        rng = np.random.default_rng(0)
        if V > args.max_gauss:
            sub = rng.choice(V, size=args.max_gauss, replace=False)
        else:
            sub = np.arange(V)
        mu=mu[sub]; cc=cc[sub]; dd=dd[sub]; rr=rr[sub]; op=op[sub]
        V = mu.shape[0]
        # covariance diagonal from conic
        c0=cc[:,0]; c1=cc[:,1]; c2=cc[:,2]
        det_conic = np.maximum(c0*c2-c1*c1, 1e-30)
        s11 = c2/det_conic; s22 = c0/det_conic
        T = 2.0*(np.log(255.0)+np.log(np.maximum(op,1e-12)))
        Tpos = np.maximum(T,0.0)
        dx_max = np.sqrt(Tpos*s11)*(1+1e-6)+1e-3; dy_max = np.sqrt(Tpos*s22)*(1+1e-6)+1e-3
        # tile assignment (3 sigma_max square)
        tx0=np.clip(np.floor((mu[:,0]-rr)/TS).astype(np.int64),0,tiles_x-1)
        tx1=np.clip(np.floor((mu[:,0]+rr)/TS).astype(np.int64),0,tiles_x-1)
        ty0=np.clip(np.floor((mu[:,1]-rr)/TS).astype(np.int64),0,tiles_y-1)
        ty1=np.clip(np.floor((mu[:,1]+rr)/TS).astype(np.int64),0,tiles_y-1)
        # per-tile AABB overlap
        ax0=np.clip(np.floor((mu[:,0]-dx_max)/TS).astype(np.int64),0,tiles_x-1)
        ax1=np.clip(np.floor((mu[:,0]+dx_max)/TS).astype(np.int64),0,tiles_x-1)
        ay0=np.clip(np.floor((mu[:,1]-dy_max)/TS).astype(np.int64),0,tiles_y-1)
        ay1=np.clip(np.floor((mu[:,1]+dy_max)/TS).astype(np.int64),0,tiles_y-1)
        keep_tile = (ax1>=tx0)&(ax0<=tx1)&(ay1>=ty0)&(ay0<=ty1)  # AABB overlaps the 3sig square tiles
        # For each gaussian, for each tile in its square range, decide skip and check pixels
        n_wrong=0; wrong_alpha_sum=0.0; n_checked=0; nan_dx=0
        nan_dx = int(np.sum(~np.isfinite(dx_max)|~np.isfinite(dy_max)))
        max_wrong_alpha=0.0
        for g in range(V):
            for ty in range(ty0[g], ty1[g]+1):
                for tx in range(tx0[g], tx1[g]+1):
                    n_checked+=1
                    # per-tile AABB overlap test (as in kernel)
                    x0=tx*TS; x1=x0+TS-1; y0=ty*TS; y1=y0+TS-1
                    overlaps = (mu[g,0]+dx_max[g]>=x0 and mu[g,0]-dx_max[g]<=x1 and
                                mu[g,1]+dy_max[g]>=y0 and mu[g,1]-dy_max[g]<=y1)
                    if overlaps:
                        continue
                    # skipped: check if any pixel in tile has alpha>=CUTOFF
                    pxs=np.arange(x0,min(x1+1,W)); pys=np.arange(y0,min(y1+1,H))
                    if pxs.size==0 or pys.size==0: continue
                    dpx=pxs[:,None]-mu[g,0]; dpy=pys[None,:]-mu[g,1]
                    q=cc[g,0]*dpx*dpx+2*cc[g,1]*dpx*dpy+cc[g,2]*dpy*dpy
                    q=np.maximum(q,0.0); q=np.minimum(q,700.0)
                    alpha=op[g]*np.exp(-0.5*q)
                    amax=alpha.max()
                    if amax>=CUTOFF:
                        n_wrong+=1; wrong_alpha_sum+=amax
                        if amax>max_wrong_alpha: max_wrong_alpha=amax
                        if n_wrong<=3:
                            gg=sub[g]
                            am=alpha.argmax(); iy,ix=np.unravel_index(am,alpha.shape)
                            print("  WRONG g=%d mu=(%.2f,%.2f) dx_max=%.3f dy_max=%.3f T=%.3f s11=%.4f s22=%.4f det_conic=%.3e | pix dx=%.2f dy=%.2f q=%.3f alpha=%.4f op=%.4f cc=%s" % (
                                gg, mu[g,0], mu[g,1], dx_max[g], dy_max[g], T[g], s11[g], s22[g], det_conic[g],
                                dpx[iy,0], dpy[0,ix], q[iy,ix], alpha[iy,ix], op[g], cc[g]), flush=True)
        print("[cam %d] V=%d checked_pairs=%d nan_dx=%d WRONG_skips=%d wrong_alpha_sum=%.3f max_wrong_alpha=%.4f" % (
            cid, V, n_checked, nan_dx, n_wrong, wrong_alpha_sum, max_wrong_alpha), flush=True)

if __name__=="__main__":
    main()
