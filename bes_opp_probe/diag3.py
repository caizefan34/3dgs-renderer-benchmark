#!/usr/bin/env python3
import sys, os, json, argparse
PACKAGE_SCRIPTS = os.environ["BES0_PACKAGE_ROOT"] + "/scripts"
sys.path.insert(0, PACKAGE_SCRIPTS)
import mx_guard; cfg = mx_guard.assert_mx_host()
import numpy as np, torch, scene_adapter as sa
ap = argparse.ArgumentParser()
ap.add_argument("--scene", default="room"); ap.add_argument("--cam", type=int, default=1)
ap.add_argument("--manifest", required=True)
args = ap.parse_args()
man = json.load(open(args.manifest)); rec = man["scenes"][args.scene]
scene = sa.load_scene(rec, "cuda"); cam = scene.cameras[args.cam]
mu2d, conic, depth, radii = sa.project_gaussians_2d(scene, cam)
mu = mu2d.detach().cpu().numpy(); cc = conic.detach().cpu().numpy()
dd = depth.detach().cpu().numpy(); rr = radii.detach().cpu().numpy()
op = sa.opacity_activation(scene.opacity).reshape(-1).cpu().numpy()
vis = (dd > 0.05) & (rr > 0); g = np.nonzero(vis)[0]
cx = mu[g,0]; cy = mu[g,1]; a = cc[g,0]; b = cc[g,1]; c = cc[g,2]
r = rr[g].astype(np.float64); opv = op[g].astype(np.float64)
print("W=%d H=%d vis=%d" % (cam.width, cam.height, g.size))
print("conic a: median=%.3e p90=%.3e max=%.3e" % (np.median(a), np.percentile(a,90), a.max()))
print("radius: median=%.1f" % np.median(r))
# sigma from conic: for axis-aligned, sigma_x=1/sqrt(a). implied radius=3*sigma
sig = 1.0/np.sqrt(np.maximum(a,1e-30))
print("implied sigma_x=1/sqrt(a): median=%.1f ; 3*sigma median=%.1f (vs radius median %.1f)" % (np.median(sig), 3*np.median(sig), np.median(r)))
# center-tile lo for first 8 visible gaussians
tiles_x=(cam.width+15)//16; tiles_y=(cam.height+15)//16
for i in range(8):
    txc=int(np.clip(np.floor(cx[i]/16),0,tiles_x-1)); tyc=int(np.clip(np.floor(cy[i]/16),0,tiles_y-1))
    px0=txc*16; px1=min(cam.width,px0+16); py0=tyc*16; py1=min(cam.height,py0+16)
    xs=[px0,px0,px1-1,px1-1]; ys=[py0,py1-1,py0,py1-1]
    qcs=[a[i]*(x-cx[i])**2+2*b[i]*(x-cx[i])*(y-cy[i])+c[i]*(y-cy[i])**2 for x,y in zip(xs,ys)]
    qlo=max(qcs)
    lo=opv[i]*np.exp(-0.5*qlo)
    print("g%d cx=%.1f cy=%.1f r=%.1f op=%.3f a=%.3e b=%.3e c=%.3e | qlo=%.3f lo=%.3e" % (
        i, cx[i], cy[i], r[i], opv[i], a[i], b[i], c[i], qlo, lo))
