#!/usr/bin/env python3
"""Diagnostic v2: why is per-occurrence R empty? radius + opacity + center-tile lo/hi."""
import sys, os, json
PACKAGE_SCRIPTS = os.environ["BES0_PACKAGE_ROOT"] + "/scripts"
sys.path.insert(0, PACKAGE_SCRIPTS)
import mx_guard
cfg = mx_guard.assert_mx_host()
import numpy as np
import torch
import scene_adapter as sa
ap = argparse.ArgumentParser() if False else None
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--scene", default="room"); ap.add_argument("--cam", type=int, default=1)
ap.add_argument("--manifest", required=True)
args = ap.parse_args()
uuid=None
for x in mx_guard.checked_inventory(cfg):
    if x["index"] == 7 and x["index"] in cfg["allowed_gpu_indices"]: uuid=x["uuid"]; break
print("[diag] skip idle gate (read-only)")
torch.set_num_threads(1); torch.cuda.set_device(0)
man = json.load(open(args.manifest)); rec = man["scenes"][args.scene]
scene = sa.load_scene(rec, "cuda"); cam = scene.cameras[args.cam]
mu2d, conic, depth, radii = sa.project_gaussians_2d(scene, cam)
mu = mu2d.detach().cpu().numpy(); cc = conic.detach().cpu().numpy()
dd = depth.detach().cpu().numpy(); rr = radii.detach().cpu().numpy()
op = sa.opacity_activation(scene.opacity).reshape(-1).cpu().numpy()
vis = (dd > 0.05) & (rr > 0)
g = np.nonzero(vis)[0]
cx = mu[g,0]; cy = mu[g,1]; a = cc[g,0]; b = cc[g,1]; c = cc[g,2]
r = rr[g].astype(np.float64); opv = op[g].astype(np.float64)
print("scene=%s cam=%d visible=%d" % (args.scene, args.cam, g.size))
print("radius px: mean=%.1f median=%.1f p90=%.1f p99=%.1f" % (r.mean(), np.median(r), np.percentile(r,90), np.percentile(r,99)))
print("opacity(active): median=%.4f mean=%.4f p10=%.4f p90=%.4f" % (np.median(opv), opv.mean(), np.percentile(opv,10), np.percentile(opv,90)))
tiles_x=(cam.width+15)//16; tiles_y=(cam.height+15)//16
# center-tile occurrence for each visible gaussian: tile containing the peak
txc = np.clip(np.floor(cx/16).astype(np.int64),0,tiles_x-1)
tyc = np.clip(np.floor(cy/16).astype(np.int64),0,tiles_y-1)
px0=txc*16; px1=np.minimum(cam.width,px0+16); py0=tyc*16; py1=np.minimum(cam.height,py0+16)
def qc(px,py):
    dx=px-cx; dy=py-cy; return a*dx*dx+2*b*dx*dy+c*dy*dy
qlo=np.maximum(np.maximum(qc(px0,py0),qc(px0,py1-1)),np.maximum(qc(px1-1,py0),qc(px1-1,py1-1)))
nx=np.clip(np.rint(cx).astype(np.int64),px0,px1-1); ny=np.clip(np.rint(cy).astype(np.int64),py0,py1-1)
dxh=nx-cx; dyh=ny-cy; qhi=a*dxh*dxh+2*b*dxh*dyh+c*dyh*dyh
with np.errstate(over='ignore',invalid='ignore'):
    lo=opv*np.exp(-0.5*qlo); hi=opv*np.exp(-0.5*qhi)
lo_raw=lo.copy()
lo=np.where(lo<sa.ALPHA_CUTOFF,0.0,lo); lo=np.clip(lo,0,sa.ALPHA_CAP); hi=np.clip(hi,0,sa.ALPHA_CAP)
print("center-tile occurrence: frac lo_raw>cut=%.4f ; frac lo_raw>1e-6=%.4f" % ((lo_raw>sa.ALPHA_CUTOFF).mean(), (lo_raw>1e-6).mean()))
pos=lo>0; ratio=hi/np.maximum(lo,1e-300)
print("  among lo>0: frac ratio<=2=%.4f <=10=%.4f <=100=%.4f median=%.3f" % (
    (ratio[pos]<=2).mean() if pos.any() else 0,(ratio[pos]<=10).mean() if pos.any() else 0,
    (ratio[pos]<=100).mean() if pos.any() else 0, np.median(ratio[pos]) if pos.any() else -1))
# opacity-stratified: for gaussians with opacity>0.05, what is center-tile lo?
big = opv>0.05
print("  gaussians opacity>0.05: n=%d frac(center lo>0)=%.4f" % (big.sum(), (lo[big]>0).mean() if big.any() else 0))
# how many visible gaussians have radius in [8,64] (tile-sized) AND opacity>0.05?
mid = (r>=8)&(r<=64)&(opv>0.05)
print("  radius in [8,64] & op>0.05: n=%d ; their center lo>0 frac=%.4f" % (mid.sum(), (lo[mid]>0).mean() if mid.any() else 0))

