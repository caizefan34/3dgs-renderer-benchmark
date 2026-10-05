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
W,H = cam.width, cam.height
TS=16; r_cap=256
g=137982
# replicate build_tile_lists tile assignment for gaussian g
r = min(int(radii[g].item()), r_cap)
cx=mu2d[g,0].item(); cy=mu2d[g,1].item()
tx0=max(0,min((cx-r)//TS, (W-1)//TS)); tx1=max(0,min((cx+r)//TS,(W-1)//TS))
ty0=max(0,min((cy-r)//TS,(H-1)//TS)); ty1=max(0,min((cy+r)//TS,(H-1)//TS))
print("gid=%d center=(%.1f,%.1f) r_cap_used=%d  tile_range x[%d,%d] y[%d,%d]" % (g,cx,cy,r,tx0,tx1,ty0,ty1))
print("Is tile (0,0) in range? %s" % (tx0<=0<=tx1 and ty0<=0<=ty1))
# Now build full tile lists and check if g is in tile 0
import importlib
# build via the same logic as cull_probe build_tile_lists
V=mu2d.shape[0]; dev=mu2d.device
tiles_x=(W+TS-1)//TS; tiles_y=(H+TS-1)//TS; num_tiles=tiles_x*tiles_y
rr=torch.clamp(radii,max=r_cap)
tx0t=torch.clamp(torch.floor((mu2d[:,0]-rr)/TS).long(),0,tiles_x-1)
tx1t=torch.clamp(torch.floor((mu2d[:,0]+rr)/TS).long(),0,tiles_x-1)
ty0t=torch.clamp(torch.floor((mu2d[:,1]-rr)/TS).long(),0,tiles_y-1)
ty1t=torch.clamp(torch.floor((mu2d[:,1]+rr)/TS).long(),0,tiles_y-1)
w=(tx1t-tx0t+1).clamp(min=1); h=(ty1t-ty0t+1).clamp(min=1)
counts=w*h; total=int(counts.sum())
seg=torch.cumsum(counts,0)-counts
ar=torch.arange(total,device=dev)
seg_rep=torch.repeat_interleave(seg,counts)
local=ar-seg_rep
gids_rep=torch.repeat_interleave(torch.arange(V,device=dev),counts)
w_rep=torch.repeat_interleave(w,counts); tx0_rep=torch.repeat_interleave(tx0t,counts); ty0_rep=torch.repeat_interleave(ty0t,counts)
ltx=local%w_rep; lty=local//w_rep
tx=tx0_rep+ltx; ty=ty0_rep+lty
tile_id=ty*tiles_x+tx
depth_rep=torch.repeat_interleave(depth,counts)
dmin=depth_rep.min(); drange=(depth_rep.max()-dmin).clamp(min=1e-6); dn=(depth_rep-dmin)/drange
composite=tile_id.double()+dn
order=torch.argsort(composite,stable=True)
tile_id_sorted=tile_id[order]; gids_sorted=gids_rep[order]
cpt=torch.bincount(tile_id_sorted,minlength=num_tiles)
tile_ptr=torch.cat([torch.zeros(1,dtype=torch.long,device=dev),torch.cumsum(cpt,0)])
begin=int(tile_ptr[0]); end=int(tile_ptr[1])
tile0_gids=gids_sorted[begin:end]
print("tile0 list length=%d  contains gid %d? %s" % (end-begin, g, bool((tile0_gids==g).any())))
