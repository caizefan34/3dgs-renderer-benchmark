#!/usr/bin/env python3
"""Diagnostic: why is per-occurrence R empty? Radius distribution + loose-cutoff lo."""
import sys, os, json, argparse
PACKAGE_SCRIPTS = os.environ["BES0_PACKAGE_ROOT"] + "/scripts"
sys.path.insert(0, PACKAGE_SCRIPTS)
import mx_guard
cfg = mx_guard.assert_mx_host()
import numpy as np
import torch
import scene_adapter as sa

ap = argparse.ArgumentParser()
ap.add_argument("--scene", default="room")
ap.add_argument("--cam", type=int, default=1)
ap.add_argument("--manifest", required=True)
args = ap.parse_args()

for x in mx_guard.checked_inventory(cfg):
    if x["index"] == 7 and x["index"] in cfg["allowed_gpu_indices"]:
        uuid = x["uuid"]; break
mx_guard.wait_idle_window(cfg, uuid)
torch.set_num_threads(1); torch.cuda.set_device(0)
man = json.load(open(args.manifest)); rec = man["scenes"][args.scene]
scene = sa.load_scene(rec, "cuda")
cam = scene.cameras[args.cam]
mu2d, conic, depth, radii = sa.project_gaussians_2d(scene, cam)
mu = mu2d.detach().cpu().numpy(); cc = conic.detach().cpu().numpy()
dd = depth.detach().cpu().numpy(); rr = radii.detach().cpu().numpy()
op = sa.opacity_activation(scene.opacity).reshape(-1).cpu().numpy()
vis = (dd > 0.05) & (rr > 0)
r = rr[vis].astype(np.float64)
print("scene=%s cam=%d W=%d H=%d visible=%d" % (args.scene, args.cam, cam.width, cam.height, vis.sum()))
print("radius px: mean=%.2f median=%.2f p90=%.2f p99=%.2f max=%.1f" % (
    r.mean(), np.median(r), np.percentile(r, 90), np.percentile(r, 99), r.max()))
# implied sigma_2d ~ radius/3
print("sigma2d px ~ radius/3: median=%.2f p90=%.2f" % (np.median(r)/3, np.percentile(r,90)/3))

# per-occurrence lo with NO cutoff (raw), distribution + fraction above thresholds
a = cc[:,0].astype(np.float64); b = cc[:,1].astype(np.float64); c = cc[:,2].astype(np.float64)
cx = mu[:,0].astype(np.float64); cy = mu[:,1].astype(np.float64); opv = op.astype(np.float64)
tiles_x = (cam.width+15)//16; tiles_y=(cam.height+15)//16
RS = 128
r_span = np.minimum(r, RS)
tx0 = np.clip(np.floor((cx-r_span)/16).astype(np.int64),0,tiles_x-1)
tx1 = np.clip(np.floor((cx+r_span)/16).astype(np.int64),0,tiles_x-1)
tx1 = np.minimum(tx1, tx0+7)
ty0 = np.clip(np.floor((cy-r_span)/16).astype(np.int64),0,tiles_y-1)
ty1 = np.clip(np.floor((cy+r_span)/16).astype(np.int64),0,tiles_y-1)
ty1 = np.minimum(ty1, ty0+7)
sx=tx1-tx0+1; sy=ty1-ty0+1; no=sx*sy
n=vis.sum()
los=[]; his=[]
CH=40000
for cs in range(0,n,CH):
    ce=min(n,cs+CH)
    ac=a[cs:ce];bc=b[cs:ce];cc2=c[cs:ce];cxc=cx[cs:ce];cyc=cy[cs:ce];opc=opv[cs:ce]
    sxc=sx[cs:ce];syc=sy[cs:ce];nxc=tx0[cs:ce];nyc=ty0[cs:ce];noc=no[cs:ce]
    tot=int(noc.sum())
    if tot==0: continue
    gr=np.repeat(np.arange(ce-cs),noc)
    o=np.arange(tot,dtype=np.int64); sxr=np.repeat(sxc,noc)
    ox=o%sxr; oy=o//sxr
    tx=np.repeat(nxc,noc)+ox; ty=np.repeat(nyc,noc)+oy
    px0=tx*16; px1=np.minimum(cam.width,px0+16); py0=ty*16; py1=np.minimum(cam.height,py0+16)
    ag=ac[gr];bg=bc[gr];cg=cc2[gr];cgx=cxc[gr];cgy=cyc[gr];opg=opc[gr]
    def qc(px,py):
        dx=px-cgx; dy=py-cgy; return ag*dx*dx+2*bg*dx*dy+cg*dy*dy
    qlo=np.maximum(np.maximum(qc(px0,py0),qc(px0,py1-1)),np.maximum(qc(px1-1,py0),qc(px1-1,py1-1)))
    nx=np.clip(np.rint(cgx).astype(np.int64),px0,px1-1); ny=np.clip(np.rint(cgy).astype(np.int64),py0,py1-1)
    dxh=nx-cgx; dyh=ny-cgy; qhi=ag*dxh*dxh+2*bg*dxh*dyh+cg*dyh*dyh
    with np.errstate(over='ignore',invalid='ignore'):
        lo=opg*np.exp(-0.5*qlo); hi=opg*np.exp(-0.5*qhi)
    los.append(lo); his.append(hi)
lo=np.concatenate(los); hi=np.concatenate(his)
print("occurrences=%d" % lo.size)
for thr in (1e-8,1e-6,1e-4,1e-3,1e-2):
    print("  frac lo>%.0e : %.4f" % (thr, (lo>thr).mean()))
ratio = hi/np.maximum(lo,1e-300)
pos = lo>1e-8
print("  among lo>1e-8: frac ratio<=2=%.4f <=10=%.4f <=100=%.4f ; median ratio=%.2f" % (
    (ratio[pos]<=2).mean() if pos.any() else 0, (ratio[pos]<=10).mean() if pos.any() else 0,
    (ratio[pos]<=100).mean() if pos.any() else 0, np.median(ratio[pos]) if pos.any() else -1))
print("  hi max=%.4f lo max=%.4f" % (hi.max(), lo.max()))
