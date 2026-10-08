#!/usr/bin/env python3
import sys, os, time, math
sys.path.insert(0, "/root/hm0_pkg")
sys.path.insert(0, "/root/epic05_moonshot/a0")
import hm0_common as C
import a0_lib as A
import torch
import torch.nn.functional as F

# bootstrap FIRST to pre-register gsplat.csrc and avoid JIT rebuild
env = C.bootstrap_c0()
from gsplat.experimental.render.functional.gaussian_inference import (
    rasterize_gaussian_higs_dynamic, _HigsDynamicScene)
print("bootstrap OK, no rebuild", flush=True)

def sync(): torch.cuda.synchronize()

def isolated_render(means, quats, scales, opac, colors, cams, sh):
    viewmats = torch.stack([c.viewmatrix for c in cams]).unsqueeze(0)
    Ks = torch.stack([c.K for c in cams]).unsqueeze(0)
    H, W = cams[0].image_height, cams[0].image_width
    bg = torch.zeros(3, device="cuda")
    scene = _HigsDynamicScene()
    out = rasterize_gaussian_higs_dynamic(
        means, quats, scales, opac, colors, sh_degree=sh,
        viewmats=viewmats, Ks=Ks, width=W, height=H, background=bg,
        render_mode="RGB", enable_culling=True, camera_model="pinhole",
        backward_mode="higs_native", eps2d=0.3, scene=scene)
    return out["frame"], out["alpha"]

scene, it = "room", 10000
ds = C.load_dataset(scene, device="cuda")
cam_ids = C.frozen_camera_ids(len(ds))
m, opt_state, ident, osd, slrs = C.load_model(scene, it, device="cuda")
sh = min(ident["active_sh_degree"], C.SH_DEGREE)
N = ident["num_points"]
cams = [ds.get_camera(ci) for ci in cam_ids]
print(f"N={N} sh={sh}", flush=True)

info_uni, uninstall = A.install_capture_patch(); A.A0_CAPTURE["enabled"]=True
isect = torch.zeros(N, device="cuda")
for ci in cam_ids:
    A.A0_CAPTURE["records"].clear()
    cam,gt = ds.get_item(ci)
    frame,info,out = C.render_c0(m,cam,sh); C.train_loss(frame,gt,C.LAMBDA_DSSIM).backward()
    cap = A.parse_capture(A.A0_CAPTURE["records"][-1])
    isect += torch.bincount(cap["visible_ids"][cap["flatten_ids_f"].reshape(-1)], minlength=N).float()
uninstall(); sync()
print("isect done", flush=True)

xyz = m["xyz"]
vis = (isect>0).nonzero().reshape(-1)
order = vis[torch.argsort(isect[vis], descending=True)]
seeds = [int(order[int(f*len(order))]) for f in (0.02,0.1,0.3,0.5,0.8)]
K = 16

def knn(seed_list, k):
    with torch.no_grad():
        qs = 0.04
        mins = xyz.min(0).values
        keys = torch.floor((xyz - mins)/qs).long()
        from collections import defaultdict
        d = defaultdict(list)
        kl = keys.tolist()
        for idx,key in enumerate(kl):
            d[(key[0],key[1],key[2])].append(idx)
        out = {}
        for s in seed_list:
            base = keys[s].tolist()
            found = None
            for R in (1,2,3,5,8):
                neigh=[]
                for dx in range(-R,R+1):
                    for dy in range(-R,R+1):
                        for dz in range(-R,R+1):
                            neigh.extend(d.get((base[0]+dx,base[1]+dy,base[2]+dz),[]))
                if len(set(neigh))>=k+1:
                    found=list(dict.fromkeys(neigh)); break
            nt = torch.tensor(found,device="cuda")
            dd = torch.cdist(xyz[s:s+1], xyz[nt]).reshape(-1)
            ii = torch.topk(dd, min(k+1,len(dd)), largest=False).indices[1:]
            out[s]=nt[ii]
        return out

t0=time.perf_counter(); nbrs = knn(seeds, K); sync()
print(f"knn {len(seeds)} seeds: {time.perf_counter()-t0:.2f}s", flush=True)

s = seeds[0]; gidx = nbrs[s]
g = {"means":m["xyz"][gidx].detach().clone().float().requires_grad_(True),
     "quats":m["rotation"][gidx].detach().clone().float().requires_grad_(True),
     "scales":m["scaling"][gidx].detach().clone().float().requires_grad_(True),
     "opac":m["opacity"][gidx].detach().clone().float().requires_grad_(True),
     "colors":m["shs"][gidx].detach().clone().float().requires_grad_(True)}

def activated(g):
    return (g["means"], F.normalize(g["quats"],dim=-1), torch.exp(g["scales"]),
            torch.sigmoid(g["opac"]), g["colors"])

t1=time.perf_counter()
for _ in range(10):
    me,qu,sc,op,co = activated(g)
    fr,al = isolated_render(me,qu,sc,op,co,cams,sh)
    (fr.sum()+al.sum()).backward()
sync()
print(f"group render+grad K={K} 8cams: {1000*(time.perf_counter()-t1)/10:.1f} ms", flush=True)

# prototype macro fit timing
def macro_init(g):
    with torch.no_grad():
        me,qu,sc,op,co = activated(g)
        w = op.clamp(1e-4,1-1e-4)
        M = (w[:,None]*me).sum(0)/w.sum()
        dm = me - M
        cov = (w[:,None,None]*(dm[:,:,None]*dm[:,None,:])).sum(0)/w.sum()
        ev,evec = torch.linalg.eigh(cov)
        scale = torch.sqrt(ev.clamp(min=1e-10))
        Rm = evec.T
        tr = Rm[0,0]+Rm[1,1]+Rm[2,2]
        q0 = torch.sqrt(torch.clamp(1+tr,0))/2
        q = torch.stack([q0,(Rm[2,1]-Rm[1,2])/(4*q0+1e-12),(Rm[0,2]-Rm[2,0])/(4*q0+1e-12),(Rm[1,0]-Rm[0,1])/(4*q0+1e-12)])
        q = F.normalize(q,dim=-1)
        opm = float(torch.sigmoid(g["opac"]).mean())
        com = co.mean(0,keepdim=True)
        return dict(means=M.clone().requires_grad_(True), quats=q.clone().requires_grad_(True),
                    scales=torch.log(scale).clone().requires_grad_(True),
                    opacity=torch.tensor(float(torch.logit(torch.tensor(opm))),device="cuda").requires_grad_(True),
                    colors=com.clone().requires_grad_(True))

mp = macro_init(g)
opt = torch.optim.Adam([mp["means"],mp["quats"],mp["scales"],mp["opacity"],mp["colors"]], lr=0.02)
t3=time.perf_counter()
for step in range(60):
    opt.zero_grad()
    me,qu,sc,op,co = activated(mp)
    fr,al = isolated_render(me,qu,sc,op,co,cams,sh)
    (fr.sum()+al.sum()).backward()
    opt.step()
sync()
print(f"macro fit 60 steps: {time.perf_counter()-t3:.2f}s ({1000*(time.perf_counter()-t3)/60:.1f} ms/step)", flush=True)
print("PROBE2_DONE", flush=True)