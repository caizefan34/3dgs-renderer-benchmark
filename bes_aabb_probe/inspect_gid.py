import sys, os, json, math
os.environ.setdefault("CUDA_HOME", "/mnt/storage_pool/liaoyuanjun_mx0qual_cuda118")
sys.path.insert(0, os.environ["BES0_PACKAGE_ROOT"] + "/scripts")
import mx_guard
cfg = mx_guard.assert_mx_host()
import torch
import scene_adapter as sa
torch.set_num_threads(1); torch.cuda.set_device(0)
man = json.load(open(os.environ["BES0_PACKAGE_ROOT"] + "/config/scene_manifest.json"))
rec = man["scenes"]["room"]
scene = sa.load_scene(rec, "cuda")
cam = scene.cameras[1]
mu2d, conic, depth, radii = sa.project_gaussians_2d(scene, cam)
op = sa.opacity_activation(scene.opacity).reshape(-1)
g = 137982
W, H = cam.width, cam.height
TS=16
m = mu2d[g]; cc = conic[g]; r = radii[g]; o = op[g]; d = depth[g]
gthr = 2.0*(math.log(max(o.item(),1e-12)) + math.log(255.0))
# f64 gdx/gdy
c0=cc[0].item(); c1=cc[1].item(); c2=cc[2].item()
det_conic = c0*c2 - c1*c1
s11 = c2/det_conic; s22 = c0/det_conic
Tpos = max(gthr, 0.0)
dxm = math.sqrt(Tpos*s11)*(1+1e-6)+1e-3
dym = math.sqrt(Tpos*s22)*(1+1e-6)+1e-3
print("gid=%d W=%d H=%d" % (g, W, H))
print("mu2d=(%.4f,%.4f) conic=(%.6f,%.6f,%.6f) det_conic=%.6e" % (m[0].item(), m[1].item(), c0, c1, c2, det_conic))
print("s11=%.6f s22=%.6f radii=%d depth=%.4f op=%.6f gthr=%.6f Tpos=%.6f" % (s11, s22, int(r.item()), d.item(), o.item(), gthr, Tpos))
print("gdx(f64+margin)=%.6f gdy(f64+margin)=%.6f" % (dxm, dym))
# tile of pixel 6 (px=6,py=0)
px, py = 6, 0
tx, ty = px//TS, py//TS
print("pixel6=(px=%d,py=%d) tile=(%d,%d) tile_x=[%d,%d] tile_y=[%d,%d]" % (px,py,tx,ty,tx*TS,tx*TS+TS-1,ty*TS,ty*TS+TS-1))
# per-tile AABB overlap test (kernel logic)
tx0=tx*TS; tx1=tx*TS+TS-1; ty0=ty*TS; ty1=ty*TS+TS-1
overlap = (m[0].item()+dxm >= tx0 and m[0].item()-dxm <= tx1 and m[1].item()+dym >= ty0 and m[1].item()-dym <= ty1)
print("kernel_keep_tile0 =", overlap)
# q at pixel 6
dx = px - m[0].item(); dy = py - m[1].item()
q = c0*dx*dx + 2*c1*dx*dy + c2*dy*dy
alpha = min(o.item()*math.exp(-0.5*q), 0.999)
print("pixel6 dx=%.4f dy=%.4f q=%.6f alpha=%.6f (cutoff=%.6f)" % (dx, dy, q, alpha, 1/255))
# 3-sigma square tile assignment
import math as _m
sigmax = _m.sqrt((s11+ (s11*s22- (c1/det_conic)**2))/2 + _m.sqrt(((s11-(s11*s22-(c1/det_conic)**2))/2)**2))
print("sigma_x=%.4f sigma_y=%.4f 3sigmax=%.4f" % (_m.sqrt(s11), _m.sqrt(s22), 3*_m.sqrt(max(s11,s22))))
